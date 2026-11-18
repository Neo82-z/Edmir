#!/usr/bin/env python3
"""Four W1 development shapes only. No SSH, device calls or frozen-set reads."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"EDM-IR"))
from edmir.compiler import w1_candidate_selection as model
from prepare_selector_gpu import implementation_id, program_artifact

SHAPES = {"grouped":[8,64,8192,1024], "m512":[8,512,8192,512],
          "large_n":[8,64,4096,2048], "m256":[8,256,8192,512]}


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def query(row):
    # Explicit allowlist: labels, fold assignments and repeat measurements are
    # unavailable to prediction, including future auxiliary target columns.
    return {k:deepcopy(row[k]) for k in (*model.CONTRACT_FIELDS,"checks","features","sm_count")}


def run(out, assets):
    started=time.monotonic()
    out.mkdir(parents=True,exist_ok=False)  # every attempted review is preserved
    spec=importlib.util.spec_from_file_location("w1_audit",ROOT/"experiments/2026-11-11-w1-local-analysis/analyze.py")
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    sources=[Path(__file__),ROOT/"EDM-IR/edmir/compiler/w1_candidate_selection.py",
             ROOT/"EDM-IR/runtime/src/nccl_device_fused_cutlass.cu",ROOT/"EDM-IR/runtime/src/cutlass_sm80_gemm.cuh",
             ROOT/"EDM-IR/runtime/include/edmir/runtime/split_k_workspace.hpp",
             ROOT/"EDM-IR/runtime/tools/benchmark_selector.cpp",ROOT/"EDM-IR/edmir/compiler/target.py",
             ROOT/"EDM-IR/tools/prepare_selector_gpu.py"]
    write(out/"implementation-sources.json",{str(p.relative_to(ROOT)):audit.sha(p) for p in sources})
    run_dir=ROOT/"experiments/2026-11-11-shadeform-w1"
    audit.analyze(run_dir,assets/"bundle",ROOT/"experiments/2026-11-01-w1-readiness/prior-predictions.json",out/"audit")
    evidence=json.loads((out/"audit/analysis.json").read_text())
    plan=json.loads((run_dir/"protocol/plan.json").read_text())
    if plan["shapes"] != SHAPES:
        raise ValueError("development shape allowlist changed")
    required=json.loads((run_dir/"protocol/inputs.json").read_text())
    required={tag:{seed:{k:meta[k] for k in ("input_sha256","reference_sha256")}
                   for seed,meta in by_seed.items()} for tag,by_seed in required.items()}
    if set(required)!=set(SHAPES):raise ValueError("unexpected input shape")
    if any(set(v)!=set(map(str,plan["seeds"])) for v in required.values()):
        raise ValueError("prescribed correctness seeds changed")
    observations={(r["shape"],r["family"]):r for r in evidence["candidates"] if r["phase"]=="measure"}
    rows=[]
    for tag,shape in SHAPES.items():
        check_jobs=[(i,j) for i,j in enumerate(plan["jobs"]) if j["shape"]==tag and j["phase"]=="check"]
        families=[f for f in check_jobs[0][1]["order"] if not f.startswith("pipeline")]
        for family in families:
            host=family in model.HOSTS
            mode="merged" if family.startswith("host_merged") else "prepared"
            policy="resource_queues_v1" if host else "uniform_immutable_peer_v1"
            obs=observations.get((tag,family))
            impl=implementation_id(family,mode,dependency_policy=policy)
            if obs and any(r["implementation_id"]!=impl for r in obs["runs"]):
                raise ValueError("source/measurement implementation mismatch")
            row=dict(shape=shape,shape_tag=tag,candidate_id=family,execution_context=evidence["audit"]["context_sha256"],
                     implementation_id=impl,source_id=json.loads((run_dir/"recovered/results/context.json").read_text())["build"],
                     precision=model.precision(family),error_contract=model.ERROR_CONTRACT,phase="measure",
                     sm_count=108,checks=[],split="train",measurement_kind="observed")
            row["features"]=model.original_geometry_features(row)
            for i,job in check_jobs:
                path=run_dir/f"recovered/results/collect-{i:03d}.json"
                raw=json.loads(path.read_text());check=next(r for r in raw["rows"] if r["family"]==family)
                row["checks"].append(dict(seed=job["seed"],input=required[tag][str(job["seed"])],
                    passed=check["correctness_passed"],checked_elements=check["checked_elements"],
                    max_abs_error=check["max_abs_error"],max_scaled_error=check["max_rel_error"],
                    record_sha256=audit.sha(path),contract_sha256=model.contract(row),
                    identity_binding="sealed registry/build/context; legacy check rows omit implementation_id"))
            status=model.admission(row,required[tag]);row["admission"]=status
            row["latency_us"]=1000*obs["median_ms"] if obs else None
            row["repeat_medians_us"]=[1000*v for v in obs["repeat_medians_ms"]] if obs else []
            if (status=="passed") != (obs is not None):raise ValueError("eligibility/timing coverage mismatch")
            row["coupling_account"]=model.coupling(row)
            rows.append(row)
    eligible=[r for r in rows if r["admission"]=="passed"]
    write(out/"dataset.json",dict(rows=rows,required_inputs=required,scope="W1 four development shapes only"))
    # The design is written before seeing either model's held-out scores.
    design=dict(variants=["original_geometry","coupled_geometry"],ridge=.01,
        split="leave one entire development shape out; all seeds/processes/candidates stay together",
        loss="weighted squared log latency; fixed diagnostic, no optimizer or parameter search",
        training_mass="equal shape mass; equal host/fused mass within shape",
        no_online_admission=True,frozen_shapes_read=False,prior_labels_used=False,
        limitations=["All four shapes influenced prior design: this is not an independent test",
                     "No host-best shape; reverse host/fused accuracy cannot be claimed",
                     "Logical coupling counts do not identify actual occupancy, contention or overlap"])
    write(out/"design.json",design)
    evaluations={}
    for variant in design["variants"]:
        decisions=[]
        for tag in SHAPES:
            train=[r for r in eligible if r["shape_tag"]!=tag]
            test=[r for r in rows if r["shape_tag"]==tag]
            fitted=model.fit(train,required,variant=variant)
            pred=model.predict(fitted,[query(r) for r in test],required[tag])
            reverse=model.predict(fitted,[query(r) for r in reversed(test)],required[tag])
            if reverse!=pred:raise AssertionError("candidate order affected prediction")
            truth={r["candidate_id"]:r for r in test if r["admission"]=="passed"}
            if set(pred["scores_us"])!=set(truth):raise ValueError("missing eligible candidate scores")
            best=min(truth,key=lambda f:(truth[f]["latency_us"],f));chosen=pred["chosen"]
            errors=[];pairs=0
            names=sorted(truth)
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    measured=truth[a]["latency_us"]-truth[b]["latency_us"]
                    estimated=pred["scores_us"][a]-pred["scores_us"][b]
                    pairs+=1
                    if measured*estimated<0:
                        repeat=[x-y for x,y in zip(truth[a]["repeat_medians_us"],truth[b]["repeat_medians_us"])]
                        errors.append(dict(a=a,b=b,measured_a_minus_b_us=measured,predicted_a_minus_b_us=estimated,
                                           repeat_deltas_us=repeat,direction_consistent_all_repeats=all(x*measured>0 for x in repeat)))
            decision=dict(shape_tag=tag,measured_best=best,
                regret_pct=100*(truth[chosen]["latency_us"]/truth[best]["latency_us"]-1),
                worst_repeat_regret_pct=max(100*(truth[chosen]["repeat_medians_us"][i]/min(r["repeat_medians_us"][i] for r in truth.values())-1) for i in range(3)),
                mean_absolute_percentage_error=statistics.mean(100*abs(pred["scores_us"][f]/truth[f]["latency_us"]-1) for f in truth),
                pair_errors=errors,pairs=pairs,score_coverage=len(truth),**pred)
            decisions.append(decision)
            write(out/f"{variant}-{tag}.json",dict(model=fitted,evaluation=decision))
        evaluations[variant]=dict(decisions=decisions,wrong_choices=sum(d["chosen"]!=d["measured_best"] for d in decisions),
            mean_regret_pct=statistics.mean(d["regret_pct"] for d in decisions),max_regret_pct=max(d["regret_pct"] for d in decisions),
            pair_errors=sum(len(d["pair_errors"]) for d in decisions),pairs=sum(d["pairs"] for d in decisions),
            mean_absolute_percentage_error=statistics.mean(d["mean_absolute_percentage_error"] for d in decisions))
        write(out/f"{variant}-development-model.json",model.fit(eligible,required,variant=variant))
    # New FP32 storage implementations have NO old timings or correctness pass.
    pending=[]
    for tag,shape in SHAPES.items():
        for q in (8,16):
            for g in (1,2):
                name=f"splitk{q}"+("_groupedn2" if g==2 else "")+"_fp32reduce"
                artifact=program_artifact(shape,name,dependency_policy="uniform_immutable_peer_v1")
                target=out/"fp32-manifests"/f"{tag}-{name}.hpp"
                target.parent.mkdir(exist_ok=True)
                target.write_text(artifact.files["edmir_program.hpp"])
                pending.append(dict(shape_tag=tag,shape=shape,candidate_id=name,precision=model.precision(name),
                    implementation_id=implementation_id(name,"prepared",dependency_policy="uniform_immutable_peer_v1"),
                    correctness="unknown",latency_us=None,compiler_admission=False,manifest=str(target.relative_to(out)),
                    expected_additional_costs=["FP32 partial matrix storage/traffic", "one final conversion launch per fused GEMM"]))
    write(out/"pending-gpu.json",dict(candidates=pending,requirements=[
        "Compile new CUDA translation unit against pinned CUTLASS and NCCL before allocation of GPU time when possible",
        "Same P=8 A100 SXM topology and W1 host execute+synchronize timing, full 3-seed output checks",
        "Only correctness-passing candidates receive timing; include original valid controls; three process repetitions",
        "Separate profiler runs: CTA/warp activity, semaphore stalls, peer access, CPU submission and conversion cost",
        "No frozen shapes in debugging; no BF16 result from FP16 evidence"] ))
    summary=dict(eligible=len(eligible),failed=sum(r["admission"]=="failed" for r in rows),
                 unknown_new_candidates=len(pending),historical_eligible_coverage=evidence["audit"]["historical_scored_and_eligible"],
                 evaluations=evaluations,elapsed_seconds=time.monotonic()-started,compiler_admission=False)
    write(out/"summary.json",summary)
    lines=["# W1 本地收敛结果","","只使用四个开发形状；六个冻结形状未读取。旧数据、失败记录与旧模型均保留。", "",
           f"52 个主候选组合中，{len(eligible)} 个通过全部三个输入种子；10 个失败。4 个 pipeline 对照属于单独阶段，不混入评分。",
           "", "下表是相同固定 ridge 校准的整形留出诊断。original_geometry 不是旧模型复现；coupled_geometry 只新增可解释计数。未按验证损失调整系数、正则或候选集。", "",
           "|表达|形状|选择|实测最快|选择损失|最坏重复损失|方向错误|", "|---|---|---|---|---:|---:|---:|"]
    for variant,result in evaluations.items():
        for d in result["decisions"]:
            lines.append(f'|{variant}|{d["shape_tag"]}|{d["chosen"]}|{d["measured_best"]}|{d["regret_pct"]:.2f}%|{d["worst_repeat_regret_pct"]:.2f}%|{len(d["pair_errors"])}/{d["pairs"]}|')
    lines += ["", "每个候选的分数、错误方向、三轮差值、训练形状和系数保存在对应 JSON；所有失败都保留，不按波动删样本。",
        "", "FP32 跨片存储有独立实现与候选身份，16 个组合待 GPU 验证，没有借用原 FP16 的计时或通过状态。",
        "", "四个开发形状都以 fused 为优；现有数据无法证明双向 host/fused 泛化。新增计数不是实测占用率，完整端到端耗时也不能分解出通信与计算争用的因果贡献。"]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="evaluations"},ensure_ascii=False))
    for name,result in evaluations.items():print(name,json.dumps({k:v for k,v in result.items() if k!="decisions"}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=Path("/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01"))
    args=parser.parse_args()
    try:
        run(args.out.resolve(),args.assets.resolve())
    except Exception:
        import traceback
        if args.out.exists() and not (args.out/"failure.json").exists():
            write(args.out/"failure.json",dict(traceback=traceback.format_exc()))
        raise
