#!/usr/bin/env python3
"""Compare a finite, inspectable latency/throughput switch with additive costs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clean_mechanism_data import clean_collection,read,sha
from audit_fine_mechanisms import audit_resources
from evaluate_mechanism_prediction import (
    DEFAULT_DATA,REPO,comparisons,metrics,prediction,row_scaling_contrasts,write,
)
from edmir.compiler.mechanism_prediction import HOSTS,MechanismModel
from edmir.compiler.regime_prediction import RegimeModel,VARIANTS

DEFAULT_DESIGN=REPO/"experiments/2026-10-25-latency-throughput-regimes/design.json"


def fit(rows,variant,*,include_flagged=False):
    cls=RegimeModel if variant in VARIANTS else MechanismModel
    return cls.fit(rows,variant=variant,include_flagged=include_flagged)


def grouped_folds(primary,variant,*,axis="shape",include_flagged=False,deadline=None):
    if axis not in ("shape","M"):
        raise ValueError("only whole shape or whole M-axis holdouts are defined")
    rows=primary if axis=="shape" else [r for r in primary if r["family"] not in HOSTS]
    key=lambda r: tuple(r["shape"]) if axis=="shape" else r["shape"][1]
    predictions,audits=[],[]
    for heldout in sorted({key(r) for r in rows}):
        if deadline is not None and time.monotonic()>deadline:
            raise TimeoutError("regime evaluation deadline exceeded")
        train=[r for r in rows if key(r)!=heldout and (r["eligible_for_calibration"] or include_flagged)]
        test=sorted((r for r in rows if key(r)==heldout),key=lambda r:r["config_id"])
        model=fit(train,variant,include_flagged=include_flagged)
        if set(model.training_config_ids)&{r["config_id"] for r in test}:
            raise ValueError("held-out identity leakage")
        if axis=="M" and any(s[1]==heldout for s in model.training_shapes):
            raise ValueError("held-out M leakage")
        output=[prediction(model,r,"whole_shape_leave_out" if axis=="shape" else "leave_M_out_stress") for r in test]
        predictions.extend(output)
        audits.append({"held_out":heldout,"axis":axis,"model":model.to_dict(),"predictions":output})
    return predictions,audits


def branch_stability(calibrated,folds,stress):
    records=[]
    for scope,items in (("calibration",[{"held_out":None,"model":calibrated}]),
                        ("shape",folds),("M",stress)):
        for item in items:
            f=item["model"]["models"]["fused"]
            c=f["coefficients_us"]
            tau=c[-2]/c[-1] if c[-1]>0 else None
            records.append({"scope":scope,"held_out":item["held_out"],
                "crossover_W_over_L":tau,"branch_counts":f.get("training_branch_counts"),
                "training_active_rank":f["unregularized_rank"],"parameters":f["feature_count"],
                "M_crossovers":[{"q":q,"g":g,"N":n,
                    "conditional_M":256*tau/(q*((n+128*g-1)//(128*g))) if tau is not None else None}
                    for n in (512,2048) for q in (2,4,8) for g in (1,2)]})
    return {"note":"K cancels from W/L=M*q*ceil(N/(128*g))/256. These are conditional algebraic branch crossings, not observed hardware saturation points. No confidence interval.",
            "fits":records}


def report(out,results,stress,models,elapsed):
    lines=["# 延迟下限与吞吐限制：受控假设检验","",
        "相同 8 个系数比较 common+aL+bW 与 common+max(aL,bW)。L=K/(2048q)，W=2MK·ceil(N/(128g))/MiB。",
        "common 保留固定项、grouped 深度、opt-in、SM 分布缺口、串行边界及额外输出字节。没有手写 M 阈值，也没有反复调神经网络。",
        "这些是旧冻结执行策略的统计代理；不是测得的设备延迟/带宽。四种 host 沿用上一版，未假设库内部 tile。","",
        "| 模型 | 平均形状误差 | 最大候选误差 | 错误方向/总对数 | >10% 损失错误对 | fused 最坏选择损失 |",
        "|---|---:|---:|---:|---:|---:|"]
    for v,r in results.items():
        lines.append(f"| {v} | {r['macro_shape_mean_absolute_percentage_error']:.2%} | {r['max_absolute_percentage_error']:.2%} | {r['wrong_pair_directions']}/{r['pairs']} | {r['wrong_pairs_over_10_percent_loss']} | {r['decision_metrics']['fused_only']['max_choice_loss_fraction']:.2%} |")
    lines += ["","| 留出的全部 M | 模型 | fused 平均形状误差 | fused 最大候选误差 | fused 最坏选择损失 |",
              "|---:|---|---:|---:|---:|"]
    for v,folds in stress.items():
        for f in folds:
            r=f["metrics"]
            lines.append(f"| {f['held_out']} | {v} | {r['macro_shape_mean_absolute_percentage_error']:.2%} | {r['max_absolute_percentage_error']:.2%} | {r['decision_metrics']['fused_only']['max_choice_loss_fraction']:.2%} |")
    lines += ["","max 模型在训练样本 W/L 的相邻值之间枚举切换区间，每个区间求非负凸二次最小值并检查 KKT；正则始终作用在同一组原始归一化系数上。",
        "所有区间共同覆盖非负系数空间，最小目标值是固定表达/损失下的数值全局解；不意味着硬件成因被识别。",
        "branch-stability.json 记录每折的系数、分支支持和交叉点。交叉点由模型而来，不能据此宣称真实设备在该 M 饱和。",
        "同一批数据已经用于多轮开发，本轮留出仍不构成独立验证。保留全部质量标记和错误；未进入自动编译器。",
        "本轮不运行 GPU，不改变旧实测标签、执行策略或 CUDA 源码。",
        f"本次完整分析 {elapsed:.2f} 秒。详见 metrics.json、pair-errors.json、leave-M-out.json、heldout-predictions.json。",""]
    (out/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--design",type=Path,default=DEFAULT_DESIGN)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("output directory must not exist")
    design=read(args.design)
    if design["variants"]!=["cutlass_depth","row_service","regime_sum","regime_max"]:
        raise ValueError("this declared four-model comparison must be preserved")
    started=time.monotonic(); deadline=started+180
    _,configs,manifest=clean_collection(args.data)
    resources=audit_resources(args.data,configs)
    primary=[r for r in configs if r["primary_scope"]]
    stable=[r for r in primary if r["eligible_for_calibration"]]
    models,heldout,folds,calibrated,results,stress={},{},{},{},{},{}
    for v in design["variants"]:
        model=fit(stable,v); models[v]=model.to_dict()
        heldout[v],folds[v]=grouped_folds(primary,v,deadline=deadline)
        calibrated[v]=[prediction(model,r,"in_sample_development") for r in primary]
        results[v]=metrics(heldout[v],design["development_acceptance"])
        _,stress[v]=grouped_folds(primary,v,axis="M",deadline=deadline)
        for f in stress[v]:
            f["metrics"]=metrics(f["predictions"],design["development_acceptance"])
        print(json.dumps({"completed":v,"mean_error":results[v]["macro_shape_mean_absolute_percentage_error"],
                          "severe_pairs":results[v]["wrong_pairs_over_10_percent_loss"]}),flush=True)
    sensitivity,sensitivity_models=grouped_folds(primary,"regime_max",include_flagged=True,deadline=deadline)
    args.out.mkdir(parents=True)
    for name,value in {
        "design":design,"dataset-manifest":manifest,"resource-audit":resources,"configurations":configs,
        "models":models,"heldout-predictions":heldout,"fold-models":folds,"calibration-predictions":calibrated,
        "metrics":results,"leave-M-out":stress,
        "calibration-metrics":{v:metrics(p,design["development_acceptance"]) for v,p in calibrated.items()},
        "decisions":{v:comparisons(p)[0] for v,p in heldout.items()},
        "pair-errors":{v:comparisons(p)[1] for v,p in heldout.items()},
        "row-scaling-contrasts":{v:row_scaling_contrasts(primary,p) for v,p in heldout.items()},
        "branch-stability":branch_stability(models["regime_max"],folds["regime_max"],stress["regime_max"]),
        "quality-sensitivity":{"scope":"Includes flagged configurations in calibration; scored on the same stable subset, not primary evidence.",
            "predictions":sensitivity,"folds":sensitivity_models,
            "metrics":metrics(sensitivity,design["development_acceptance"])},
    }.items():write(args.out/(name+".json"),value)
    elapsed=time.monotonic()-started
    report(args.out,results,stress,models,elapsed)
    sources=[Path(__file__),args.design,
        *[Path(__file__).with_name(f) for f in ("evaluate_mechanism_prediction.py","clean_mechanism_data.py","audit_fine_mechanisms.py","independent_selector_validation.py","run_mechanism_gpu.py","run_selector_gpu.py")],
        *[REPO/"EDM-IR/edmir/compiler"/f for f in ("mechanism_prediction.py","regime_prediction.py")]]
    write(args.out/"analysis-seal.json",{"source_sha256":{str(f.relative_to(REPO)):sha(f) for f in sources},
        "output_sha256":{f.name:sha(f) for f in sorted(args.out.iterdir()) if f.is_file()},
        "elapsed_seconds":elapsed,"gpu_runs":0,"independent_validation":False,"compiler_admission":False})
    print(json.dumps({"out":str(args.out.resolve()),"elapsed_seconds":elapsed}),flush=True)


if __name__=="__main__":
    main()
