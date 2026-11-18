#!/usr/bin/env python3
"""Audit the sealed four-shape precision window; no GPU or frozen-set access.

One fixed, whole-shape ridge comparison diagnoses the existing expressions.
It does not select hyperparameters or promote a production scoring model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3
import statistics as st
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EDM-IR"))
from edmir.compiler import w1_candidate_selection as model
from run_w1_gpu import validate_rows, done_raw_changed, require_successful_pilot

SHAPES = {"m512": [8,512,8192,512], "large_n": [8,64,4096,2048],
          "m256": [8,256,8192,512], "grouped": [8,64,8192,1024]}
SEEDS = [20261020, 20261101, 20261102]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def read(path):
    return json.loads(path.read_text())


def checked_file(root, manifest, name):
    path = root / name
    if path.resolve().is_relative_to(root.resolve()) is False:
        raise ValueError("receipt path escapes recovery directory")
    info = manifest[name]
    if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
        raise ValueError("recovery hash/size mismatch: "+name)
    return path


def import_window(window, bundle):
    recovered = window / "recovered"
    manifest = read(recovered / "recovery-manifest.json")["files"]
    if not 1 <= len(manifest) <= 512:
        raise ValueError("bounded recovery manifest required")
    for name in manifest:
        checked_file(recovered, manifest, name)
    queue = read(recovered / "window/queue.json")
    source = read(bundle / "plan.json")
    if source["shapes"] != SHAPES or source["seeds"] != SEEDS:
        raise ValueError("only the four development shapes and prescribed seeds are allowed")
    if (queue["source_manifest_sha256"] != sha(bundle / "files.json") or
        queue["source_plan_sha256"] != sha(bundle / "plan.json")):
        raise ValueError("sealed plan identity changed")
    input_records = read(bundle / "inputs.json")
    required = {tag:{str(seed):{k:input_records[tag][str(seed)][k]
                    for k in ("input_sha256", "reference_sha256")} for seed in SEEDS} for tag in SHAPES}
    receipt = read(recovered / "build-precision/receipt.json")
    if receipt["package"] != queue["source_manifest_sha256"]:
        raise ValueError("build/package identity mismatch")
    for tag in SHAPES:
        if sha(recovered / f"build-precision/w1_{tag}_batch") != receipt["binaries"][tag]:
            raise ValueError("measured binary changed")
    rows, contexts, all_measurements = [], set(), {}
    if [s["name"] for s in queue["stages"]] != list(SHAPES):
        raise ValueError("stage order changed")
    for stage in queue["stages"]:
        tag=stage["name"]; folder=recovered/"results-precision"/tag
        context=read(folder/"context.json"); ctx_hash=sha(folder/"context.json")
        contexts.add(ctx_hash)
        if context["build"] != sha(recovered/"build-precision/receipt.json"):
            raise ValueError("context/build binding failed")
        require_successful_pilot(folder,ctx_hash,source,context)
        if stage["jobs"] != [source["jobs"][i] for i in stage["source_job_indices"]]:
            raise ValueError("stage differs from sealed jobs")
        checks, measurements, invalid = defaultdict(list), defaultdict(list), set()
        for i,job in enumerate(stage["jobs"]):
            path=folder/f"collect-{i:03d}.json"; record=read(path)
            if record["job"] != job or record["context"] != ctx_hash or done_raw_changed(path,folder):
                raise ValueError("raw record/context mismatch")
            if record["input"] != input_records[tag][str(job["seed"])]:
                raise ValueError("input receipt mismatch")
            order=[f for f in job["order"] if job["phase"] == "check" or f not in invalid]
            if record["order"] != order:
                raise ValueError("incorrect admission filter")
            validate_rows(record["rows"],job,order,SHAPES[tag],context["runtime"])
            for r in record["rows"]:
                item=dict(raw=r,seed=job["seed"],record_sha256=sha(path),
                          record=str(path.relative_to(window)),position=order.index(r["family"]),
                          process_seconds=record["process_seconds"])
                (checks if job["phase"] == "check" else measurements)[r["family"]].append(item)
                if not r["correctness_passed"]: invalid.add(r["family"])
        for family, observations in checks.items():
            if len(observations) != 3: raise ValueError("incomplete correctness protocol")
            ids={v["raw"]["implementation_id"] for v in observations}
            if len(ids) != 1: raise ValueError("mixed candidate implementation")
            row=dict(shape=SHAPES[tag],shape_tag=tag,candidate_id=family,
                     execution_context=ctx_hash,source_id=context["build"],
                     implementation_id=ids.pop(),precision=model.precision(family),
                     error_contract=model.ERROR_CONTRACT,phase="measure",sm_count=108,
                     split="train",measurement_kind="observed",checks=[])
            row["features"]=model.original_geometry_features(row)
            for v in observations:
                raw=v["raw"]
                row["checks"].append(dict(seed=v["seed"],input=required[tag][str(v["seed"])],
                    passed=raw["correctness_passed"],checked_elements=raw["checked_elements"],
                    max_abs_error=raw["max_abs_error"],max_scaled_error=raw["max_rel_error"],
                    contract_sha256=model.contract(row),record_sha256=v["record_sha256"]))
            row["admission"]=model.admission(row,required[tag])
            times=measurements.get(family,[])
            if (row["admission"] == "passed") != (len(times) == 3):
                raise ValueError("admission/timing coverage mismatch")
            if row["admission"] != "passed" and times:
                raise ValueError("failed candidate supplied a timing label")
            row["repeat_medians_us"]=[v["raw"]["median_ms"]*1000 for v in times]
            row["latency_us"]=st.median(row["repeat_medians_us"]) if times else None
            row["measurement_records"]=[{k:v[k] for k in ("seed","record","record_sha256","position")} for v in times]
            row["coupling_account"]=model.coupling(row)
            rows.append(row)
        all_measurements[tag]=dict(measurements)
    if len(contexts) != 1: raise ValueError("whole-shape diagnosis requires one execution condition")
    return dict(rows=rows,required_inputs=required,execution_context=contexts.pop(),
                recovered_files_verified=len(manifest),frozen_shapes_read=False),all_measurements


def query(row):
    return {k:deepcopy(row[k]) for k in (*model.CONTRACT_FIELDS,"checks","features","sm_count")}


def pair_directions(a,b):
    deltas=[x-y for x,y in zip(a,b)]
    return dict(deltas_us=deltas,consistent=all(x>0 for x in deltas) or all(x<0 for x in deltas))


def evaluate(dataset,out):
    rows=dataset["rows"]; required=dataset["required_inputs"]
    eligible=[r for r in rows if r["admission"] == "passed"]
    evaluations={}
    for variant in ("original_geometry","coupled_geometry"):
        decisions=[]
        for tag in SHAPES:
            train=[r for r in eligible if r["shape_tag"] != tag]
            test=[r for r in rows if r["shape_tag"] == tag]
            fitted=model.fit(train,required,variant=variant)
            pred=model.predict(fitted,[query(r) for r in test],required[tag])
            if pred != model.predict(fitted,[query(r) for r in reversed(test)],required[tag]):
                raise AssertionError("candidate permutation changed scoring")
            truth={r["candidate_id"]:r for r in test if r["admission"]=="passed"}
            if set(pred["scores_us"]) != set(truth): raise ValueError("incomplete eligible scores")
            best=min(truth,key=lambda f:(truth[f]["latency_us"],f));chosen=pred["chosen"]
            errors=[];stable=0; unstable=0
            for a,b in itertools.combinations(sorted(truth),2):
                pair=pair_directions(truth[a]["repeat_medians_us"],truth[b]["repeat_medians_us"])
                stable+=pair["consistent"];unstable+=not pair["consistent"]
                delta=truth[a]["latency_us"]-truth[b]["latency_us"]
                est=pred["scores_us"][a]-pred["scores_us"][b]
                if delta*est<0:
                    errors.append(dict(a=a,b=b,measured_delta_us=delta,predicted_delta_us=est,**pair))
            d=dict(shape_tag=tag,**pred,measured_best=best,
                regret_pct=100*(truth[chosen]["latency_us"]/truth[best]["latency_us"]-1),
                worst_block_regret_pct=max(100*(truth[chosen]["repeat_medians_us"][i]/min(r["repeat_medians_us"][i] for r in truth.values())-1) for i in range(3)),
                pair_errors=errors,stable_pairs=stable,unstable_pairs=unstable,
                stable_pair_errors=sum(e["consistent"] for e in errors),
                mape_pct=st.mean(100*abs(pred["scores_us"][f]/truth[f]["latency_us"]-1) for f in truth),
                design_rank=fitted["design_rank"],coefficient_count=fitted["coefficient_count"])
            write(out/f"{variant}-{tag}.json",dict(model=fitted,evaluation=d));decisions.append(d)
        evaluations[variant]=dict(decisions=decisions,wrong_choices=sum(d["chosen"]!=d["measured_best"] for d in decisions),
            mean_regret_pct=st.mean(d["regret_pct"] for d in decisions),max_regret_pct=max(d["regret_pct"] for d in decisions),
            pair_errors=sum(len(d["pair_errors"]) for d in decisions),
            stable_pair_errors=sum(d["stable_pair_errors"] for d in decisions),
            stable_pairs=sum(d["stable_pairs"] for d in decisions),unstable_pairs=sum(d["unstable_pairs"] for d in decisions))
    return evaluations


def union_length(intervals):
    total=0;stop=None
    for start,end in sorted(intervals):
        if end<start: raise ValueError("negative interval")
        total+=end-start if stop is None else max(0,end-max(start,stop))
        stop=end if stop is None else max(stop,end)
    return total


def trace_analysis(window,summary):
    cases=[]
    for db in sorted((window/"recovered/results-precision").glob("*/diagnostic/*/timeline.sqlite")):
        tag=db.parents[2].name;case=db.parent.name
        c=sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True)
        c.execute("PRAGMA query_only=ON")
        capture=c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='edmir.w1.capture'").fetchall()
        if len(capture)!=1 or capture[0][1]<=capture[0][0]:raise ValueError("invalid capture range")
        kernels=c.execute('''SELECT k.start,k.end,k.deviceId,s.value,k.correlationId
                 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName''').fetchall()
        apis=c.execute('''SELECT r.start,r.end,r.globalTid,r.correlationId,s.value
                 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId''').fetchall()
        ranges=c.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'edm.op:%' AND end>start").fetchall()
        launch={r[3]:r for r in apis if r[4].startswith("cudaLaunchKernel")}
        iterations=defaultdict(list)
        for k in kernels:
            r=launch[k[4]]
            matches=[v for v in ranges if v[2]==r[2] and v[0]<=r[0] and r[1]<=v[1]]
            if len(matches)!=1:raise ValueError("kernel launch does not map uniquely to operator NVTX range")
            op=matches[0][3];it=int(re.search(r"\.iter(\d+)$",op).group(1))
            iterations[it].append(k)
        if len(iterations)!=3:raise ValueError("trace must contain three complete executions")
        gpu=[]
        for d in range(8):
            intervals=[(k[0],k[1]) for k in kernels if k[2]==d]
            span=max(e for _,e in intervals)-min(b for b,_ in intervals)
            gpu.append(dict(device=d,kernel_count=len(intervals),span_us=span/1000,
                kernel_interval_union_us=union_length(intervals)/1000,
                kernel_present_fraction=union_length(intervals)/span))
        kernel_types={name:dict(count=sum(k[3]==name for k in kernels),
                    median_us=st.median((k[1]-k[0])/1000 for k in kernels if k[3]==name)) for name in sorted({k[3] for k in kernels})}
        per_iter=[dict(iteration=i,gpu_ids=sorted({k[2] for k in ks}),kernel_count=len(ks),
                  gpu_envelope_us=(max(k[1] for k in ks)-min(k[0] for k in ks))/1000) for i,ks in sorted(iterations.items())]
        queue=read(window/"recovered/window/queue.json")
        bundle=ROOT/"experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2"
        family=read(bundle/"plan.json")["trace_cases"][case][1]
        plain=summary["stages"][tag]["timings"][family]["median_ms"]*1000
        cases.append(dict(case=case,shape=tag,candidate=family,diagnostic_only=True,
                    capture_us=(capture[0][1]-capture[0][0])/1000,
                    capture_per_execution_over_unprofiled_median=(capture[0][1]-capture[0][0])/3000/plain,
                    note="Ratio includes profiler/NVTX and diagnostic workload effects; not a correction factor.",
                    kernel_types=kernel_types,per_gpu=gpu,iterations=per_iter,
                    runtime_api_counts=dict(Counter(v[4] for v in apis))))
        c.close()
    if len(cases)!=7:raise ValueError("missing trace cases")
    return dict(cases=cases,limitations=["No profiler time used as training label",
        "Kernel interval presence is not SM occupancy or useful compute utilization",
        "Gaps include submission, synchronization, other work and instrumentation; not attributed to CPU alone",
        "CUDA event tracing may perturb dependency arrows; no intrakernel overlap or bandwidth conclusion"])


def run(window,bundle,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design=dict(scope="four development shapes only; six frozen shapes not opened",variants=["original_geometry","coupled_geometry"],
       ridge=.01,fold="leave one entire development shape out",search=False,production_promotion=False,
       interval_rule="report all median direction errors AND all-three-block sign consistency; no discarded pairs",
       confounds=["each block changes input seed and candidate permutation together", "machine state is not randomized independently"],
       same_shape_evaluation_is_not_independent=True)
    write(out/"design.json",design)
    try:
        dataset,measurements=import_window(window,bundle);write(out/"dataset.json",dataset)
        eligible=[r for r in dataset["rows"] if r["admission"]=="passed"]
        descriptions={}
        for tag in SHAPES:
            rs={r["candidate_id"]:r for r in eligible if r["shape_tag"]==tag}
            collisions=[]
            for a,b in itertools.combinations(sorted(rs),2):
                if model.features(rs[a],"original_geometry")==model.features(rs[b],"original_geometry"):
                    collisions.append(dict(a=a,b=b,relative_latency_difference_pct=100*(rs[b]["latency_us"]/rs[a]["latency_us"]-1)))
            spreads={}
            for f,items in measurements[tag].items():
                spreads[f]=[dict(seed=x["seed"],position=x["position"],median_us=x["raw"]["median_ms"]*1000,
                    p10_us=st.quantiles(x["raw"]["samples_ms"],n=10)[0]*1000,
                    p90_us=st.quantiles(x["raw"]["samples_ms"],n=10)[8]*1000,
                    first_half_median_us=st.median(x["raw"]["samples_ms"][:50])*1000,
                    last_half_median_us=st.median(x["raw"]["samples_ms"][50:])*1000) for x in items]
            descriptions[tag]=dict(feature_collisions=collisions,raw_block_summary=spreads,
                block_winners=[min(rs,key=lambda f:rs[f]["repeat_medians_us"][i]) for i in range(3)])
        write(out/"measurement-analysis.json",descriptions)
        old=read(ROOT/"experiments/2026-11-13-w1-local-convergence/round-05/coupled_geometry-development-model.json")
        old_result={tag:model.predict(old,[query(r) for r in dataset["rows"] if r["shape_tag"]==tag],dataset["required_inputs"][tag]) for tag in SHAPES}
        write(out/"old-model-contract-check.json",dict(results=old_result,
            note="Different source/execution identity: no rebinding old correctness or silently transferring calibration"))
        evaluations=evaluate(dataset,out);write(out/"score-diagnostics.json",evaluations)
        trace=trace_analysis(window,read(window/"stage-summary.json"));write(out/"trace-analysis.json",trace)
        write(out/"receipt.json",dict(status="analysis_complete",elapsed_seconds=time.monotonic()-started,
            source_sha256=sha(__file__),model_source_sha256=sha(ROOT/"EDM-IR/edmir/compiler/w1_candidate_selection.py"),
            manifest_sha256=sha(window/"recovered/recovery-manifest.json"),
            rows=len(dataset["rows"]),eligible=len(eligible),failed=len(dataset["rows"])-len(eligible),
            frozen_shapes_read=False,GPU_used=False,optimizer_search=False,production_model_changed=False))
        for variant,v in evaluations.items():
            print(variant,json.dumps({k:v for k,v in v.items() if k!="decisions"}))
            for d in v["decisions"]:print(d["shape_tag"],d["chosen"],d["regret_pct"],len(d["pair_errors"]))
    except BaseException:
        import traceback
        write(out/"failure.json",dict(traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window",type=Path,default=ROOT/"experiments/2026-11-16-shadeform-precision")
    p.add_argument("--bundle",type=Path,default=ROOT/"experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.window.resolve(),a.bundle.resolve(),a.out.resolve())
