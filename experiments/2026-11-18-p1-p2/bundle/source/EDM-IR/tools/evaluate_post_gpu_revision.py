#!/usr/bin/env python3
"""One bounded revision: metrics first, held-out cost/selection second.

No GPU, runtime changes, hidden label imputation or automatic model promotion.
Every rho is reported; these repeatedly examined datasets are development data.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import statistics as st
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_gpu_evidence import REPO, load_evidence, coverage, sha, read
from evaluate_mechanism_prediction import prediction, metrics, comparisons
from edmir.compiler.mechanism_prediction import HOSTS, MechanismModel
from edmir.compiler.regime_prediction import RegimeModel
from edmir.compiler.residual_regime_prediction import ResidualRegimeModel
from edmir.compiler.retained_regime_prediction import RetainedRegimeModel
from edmir.compiler.metric_evidence import metric_evidence
from edmir.compiler.candidate_budget import rank_predictions, shortlist, score, uniform_random_reference, training_only_order

DESIGN=REPO/"experiments/2026-10-25-post-gpu-revision/design.json"
MERGED=("host_merged_cutlass@merged", "host_merged_cublaslt@merged")


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(path)


def fit(rows, variant):
    if variant.startswith("retained_"):
        return RetainedRegimeModel.fit(rows,retained_fraction=float(variant.split("_")[1]))
    cls={"regime_max":RegimeModel,"residual_depth":ResidualRegimeModel,"row_service":MechanismModel}[variant]
    return cls.fit(rows,variant=variant)


def score_scopes(rows, acceptance):
    if not rows:
        return {"status":"unavailable", "reason":"no supported predictions"}
    all_rows=[dict(r,quality_flags=[]) for r in rows]
    return {"all_complete":metrics(all_rows,acceptance), "stable":metrics(rows,acceptance),
            "flagged_retained":[r["config_id"] for r in rows if r["quality_flags"]]}


def evaluate_fold(rows, variant, axis, heldout, repeat=None):
    key=lambda r:tuple(r["shape"]) if axis=="shape" else r["shape"][1 if axis=="M" else 3]
    eligible=lambda r:axis=="shape" or r["family"] not in HOSTS
    train=[r for r in rows if eligible(r) and key(r)!=heldout and r["eligible_for_calibration"]]
    test=[r for r in rows if eligible(r) and key(r)==heldout]
    model=fit(train,variant)
    fitted_kinds=model.baseline.models if isinstance(model,RegimeModel) else model.models
    if any(key(r)==heldout for r in train) or set(model.training_config_ids)&{r["config_id"] for r in test}:
        raise ValueError("held-out group leakage")
    output, skipped, repeated = [], [], []
    for label,source,target in (("primary",test,output),("repeat",repeat or [],repeated)):
        for row in source:
            if not eligible(row) or key(row)!=heldout:
                continue
            kind=row["family"] if row["family"] in HOSTS else "fused"
            if kind not in fitted_kinds:
                skipped.append({"evaluation":label,"config_id":row["config_id"],"shape":row["shape"],
                                "candidate_id":row["candidate_id"],"reason":"fewer than two training shapes for implementation"})
                continue
            target.append(prediction(model,row,"development_leave_"+axis+"_out"))
    fixed=[]
    if axis=="shape":
        fixed=training_only_order(train,heldout)["order"]
    return {"axis":axis,"held_out":heldout,"model":model.to_dict(),"predictions":output,
            "repeat_predictions":repeated,"skipped":skipped,"training_only_fixed_order":fixed}


def choices(rows, fixed_orders=None):
    groups=defaultdict(list)
    for row in rows:
        if not row["quality_flags"]:
            groups[tuple(row["shape"])].append(row)
    output=[]
    for shape, group in sorted(groups.items()):
        if len(group)<2:
            continue
        ids={r["candidate_id"] for r in group}
        order=rank_predictions(group)
        if order!=rank_predictions(group[::-1]):
            raise ValueError("candidate-order dependence")
        policies={f"model_top{k}":shortlist(order,k) for k in (1,2,3)}
        for cid in MERGED:
            if cid in ids:policies[cid]=[cid]
        if set(MERGED)<=ids:policies["two_merged_hosts"]=list(MERGED)
        for k in (1,2,3):
            if MERGED[1] in ids:
                policies[f"protected_lt_top{k}"]=shortlist(order,k,protected=MERGED[1])
        fixed=(fixed_orders or {}).get(shape,[])
        available_fixed=[c for c in fixed if c in ids]
        if available_fixed:
            policies["training_fixed_top2"]=available_fixed[:2]
        results={name:score(group,subset) for name,subset in policies.items()}
        results["random_top2"]=uniform_random_reference(group,2)
        output.append({"shape":shape,"candidate_count":len(group),"matched_two_hosts":set(MERGED)<=ids,
                       "policies":results,"unobserved_training_fixed_candidates":sorted(set(fixed)-ids),
                       "deployment_validated":False})
    return output


def acceptance_decision(results, counterexamples):
    baseline={label:results[f"{label}/shape/regime_max"]["stable"] for label in ("old","primary","repeat")}
    decisions={}
    for rho in (.25,.5,.75,1.):
        variant=f"retained_{rho:g}"; checks={}
        for label in baseline:
            r=results[f"{label}/shape/{variant}"]["stable"];b=baseline[label]
            for scope in ("all_measured_stable","fused_only"):
                for metric in ("mean_choice_loss_fraction","max_choice_loss_fraction"):
                    checks[f"{label}/{scope}/{metric}_no_regression"]=r["decision_metrics"][scope][metric]<=b["decision_metrics"][scope][metric]+1e-10
            checks[f"{label}/original_gates"]=results[f"{label}/shape/{variant}"]["all_complete"]["development_acceptance_passed"]
            for axis in ("M","N"):
                stress=results[f"{label}/{axis}/{variant}"]["stable"]["decision_metrics"]["fused_only"]
                control=results[f"{label}/{axis}/regime_max"]["stable"]["decision_metrics"]["fused_only"]
                checks[f"{label}/{axis}/worst_loss_no_regression"]=stress["max_choice_loss_fraction"]<=control["max_choice_loss_fraction"]+1e-10
        targets=counterexamples[variant]
        checks["robust_counterexamples_repaired"]=len(targets)==4 and all(t["predicted_b_minus_a_us"]*t["observed_b_minus_a_us"]>0 for t in targets)
        decisions[variant]={"checks":checks,"local_gate_passed":all(checks.values())}
    return {"variants":decisions,"promoted":False,"compiler_admission":False,
            "independent_validation":False,"further_total_latency_feature_search":False,
            "next_action":"prepare targeted stage diagnostics" if not any(d["local_gate_passed"] for d in decisions.values())
                          else "review local gates and freeze a new independent GPU test"}


def run(out, design, started):
    datasets, provenance=load_evidence()
    write(out/"design.json",design);write(out/"provenance.json",provenance)
    write(out/"configurations.json",datasets);write(out/"candidate-coverage.json",coverage(datasets))
    unique=sorted({(tuple(r["shape"]),r["family"]) for rows in datasets.values() for r in rows})
    evidence=[metric_evidence(shape,family) for shape,family in unique]
    write(out/"metric-evidence.json",evidence)
    variants=design["controls"]+[f"retained_{r:g}" for r in design["rho_grid"]]
    results={};forecasts={};selection={};full_models={};calibration={};missing={};counterexamples={}
    for context in ("old","primary"):
        rows=datasets[context]
        for variant in variants:
            model=fit([r for r in rows if r["eligible_for_calibration"]],variant)
            full_models[f"{context}/{variant}"]=model.to_dict()
            calibration[f"{context}/{variant}"]=[prediction(model,r,"in_sample_development_only") for r in rows]
            for axis in design["axes"]:
                key=lambda r:tuple(r["shape"]) if axis=="shape" else r["shape"][1 if axis=="M" else 3]
                groups=sorted({key(r) for r in rows if axis=="shape" or r["family"] not in HOSTS})
                outputs=[];repeated=[];skipped=[];fixed_orders={}
                for i,heldout in enumerate(groups):
                    fold=evaluate_fold(rows,variant,axis,heldout,datasets["repeat"] if context=="primary" else None)
                    write(out/"folds"/context/variant/axis/f"{i:02d}.json",fold)
                    outputs.extend(fold["predictions"]);repeated.extend(fold["repeat_predictions"]);skipped.extend(fold["skipped"])
                    if axis=="shape":fixed_orders[tuple(heldout)]=fold["training_only_fixed_order"]
                    write(out/"progress.json",{"state":"running","context":context,"variant":variant,"axis":axis,
                          "completed_folds":i+1,"total_folds":len(groups),"elapsed_seconds":time.monotonic()-started})
                tag=f"{context}/{axis}/{variant}"
                results[tag]=score_scopes(outputs,design["acceptance"]);forecasts[tag]=outputs;missing[tag]=skipped
                if axis=="shape":selection[tag]=choices(outputs,fixed_orders)
                if context=="primary":
                    repeat_tag=f"repeat/{axis}/{variant}"
                    results[repeat_tag]=score_scopes(repeated,design["acceptance"]);forecasts[repeat_tag]=repeated
                    if axis=="shape":selection[repeat_tag]=choices(repeated,fixed_orders)
                write(out/"metrics.json",results)
            print(json.dumps({"context":context,"variant":variant,
                "elapsed_seconds":round(time.monotonic()-started,2),
                "mape":results[f"{context}/shape/{variant}"]["stable"]["macro_shape_mean_absolute_percentage_error"]}),flush=True)
    pairs={key:comparisons([dict(r,quality_flags=[]) for r in rows])[1] for key,rows in forecasts.items() if "/shape/" in key}
    target_keys={(4,256,8192,512):{"splitk2@prepared","splitk8@prepared"},
                 (4,64,4096,2048):{"host@prepared","splitk4_groupedn2@prepared"}}
    for variant in variants:
        counterexamples[variant]=[dict(p,collection=label) for label in ("primary","repeat")
            for p in pairs[f"{label}/shape/{variant}"] if target_keys.get(tuple(p["shape"]))=={p["a"],p["b"]}]
    decision=acceptance_decision(results,counterexamples)
    for name,value in {"models":full_models,"heldout-predictions":forecasts,"calibration-predictions":calibration,
        "unsupported-predictions":missing,"selection":selection,"pair-decomposition":pairs,
        "counterexamples":counterexamples,"decision":decision}.items():write(out/(name+".json"),value)
    lines=["# 结合中间量审查的受控模型修订", "", "本地开发结果；旧数据、新数据分别标定。复跑与附加基线未用于标定。没有运行 GPU，也未自动替换编译器模型。", "",
        "借鉴论文的指标/性能分层验证与候选质量评价；没有移植 stencil 缓存模型。静态枚举通过不等于真实流量或等待已被识别。", "",
        "| 数据 | 表达 | 留形状 MAPE | 最大 APE | 错序 | 严重错序 | 全候选最坏选择损失 | fused 最坏选择损失 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for context in ("old","primary","repeat"):
        for variant in variants:
            r=results[f"{context}/shape/{variant}"]["stable"]
            lines.append(f"| {context} | {variant} | {r['macro_shape_mean_absolute_percentage_error']:.2%} | {r['max_absolute_percentage_error']:.2%} | {r['wrong_pair_directions']}/{r['pairs']} | {r['wrong_pairs_over_10_percent_loss']} | {r['decision_metrics']['all_measured_stable']['max_choice_loss_fraction']:.2%} | {r['decision_metrics']['fused_only']['max_choice_loss_fraction']:.2%} |")
    lines.extend(["", "上表为按原质量规则取得的稳定子集；metrics.json 同时保留完整集。重复数据只是同模型复查，不是额外独立训练形状。", "",
        "- candidate-coverage.json：逐集合、逐形状的实测/异常/缺失身份。", "- metric-evidence.json：独立逻辑计数核对、资源上界及未测硬件指标。",
        "- counterexamples.json / pair-decomposition.json：留形状预测下的方向、每项贡献及原进程差值。", "- selection.json：相同支持集下的两种合并 host、模型及其他等预算策略。缺标签处不补值。",
        "- folds/：每一折训练身份、系数与预测；M/N 压力检查只评价 fused，库家族覆盖不足单独报告。", "",
        "决策："+decision["next_action"]+"。没有获得自动首选或独立泛化资格。", "",
        f"本次耗时 {time.monotonic()-started:.2f} 秒；输入/输出哈希见 analysis-seal.json。", ""])
    (out/"README.md").write_text("\n".join(lines))
    write(out/"progress.json",{"state":"complete","elapsed_seconds":time.monotonic()-started})
    source_files=[Path(__file__),DESIGN,Path(__file__).with_name("post_gpu_evidence.py"),Path(__file__).with_name("evaluate_mechanism_prediction.py"),
        *[REPO/"EDM-IR/edmir/compiler"/n for n in ("mechanism_prediction.py","regime_prediction.py","residual_regime_prediction.py",
                                                 "retained_regime_prediction.py","metric_evidence.py","candidate_budget.py")]]
    write(out/"analysis-seal.json",{"sources":{str(p.relative_to(REPO)):sha(p) for p in source_files},
        "outputs":{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        "elapsed_seconds":time.monotonic()-started,"independent_validation":False,"gpu_runs":0})
    print(json.dumps(decision,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():parser.error("fresh output directory required")
    design=read(DESIGN);started=time.monotonic();a.out.mkdir(parents=True)
    def timeout(*_):raise TimeoutError("whole analysis exceeded 15-minute budget")
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,design["whole_run_timeout_seconds"])
    try:run(a.out,design,started)
    except Exception as exc:
        write(a.out/"failure.json",{"state":"failed","error":repr(exc),"elapsed_seconds":time.monotonic()-started,"compiler_admission":False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=="__main__":main()
