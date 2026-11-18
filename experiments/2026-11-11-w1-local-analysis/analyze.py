#!/usr/bin/env python3
"""Read-only W1 development analysis. No model training, device or network calls.

Reports use medians of three process medians, never a pooled 300-sample median.
Only the four explicitly named development shapes are permitted.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHAPES = {"grouped": [8, 64, 8192, 1024], "m512": [8, 512, 8192, 512],
          "large_n": [8, 64, 4096, 2048], "m256": [8, 256, 8192, 512]}
HOSTS = ("host", "host_merged_cutlass", "host_merged_cublaslt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rows):
    """Rows contain one candidate in three independently launched processes."""
    require(len(rows) == 3 and {r["repetition"] for r in rows} == {0, 1, 2},
            "expected exactly three distinct process repetitions")
    rows = sorted(rows, key=lambda r: r["repetition"])
    medians = [st.median(r["samples_ms"]) for r in rows]
    center = st.median(medians)
    return dict(median_ms=center, repeat_medians_ms=medians,
                process_range_pct=100 * (max(medians) - min(medians)) / center,
                sample_count=sum(len(r["samples_ms"]) for r in rows),
                runs=[dict(job_index=r["job_index"], repetition=r["repetition"],
                           order_position=r["order_position"],
                           median_ms=st.median(r["samples_ms"]),
                           first_quarter_ms=st.median(r["samples_ms"][:25]),
                           last_quarter_ms=st.median(r["samples_ms"][-25:]),
                           last_vs_first_quarter_pct=100 * (
                               st.median(r["samples_ms"][-25:]) /
                               st.median(r["samples_ms"][:25]) - 1),
                           implementation_id=r["implementation_id"],
                           gemm_algorithm_id=r["gemm_algorithm_id"])
                      for r in rows])


def paired(a, b):
    """Positive reduction means a is faster than b, within corresponding processes."""
    return dict(a=a["family"], b=b["family"],
                delta_a_minus_b_ms=[x-y for x, y in zip(a["repeat_medians_ms"], b["repeat_medians_ms"])],
                reduction_a_vs_b_pct=[100*(1-x/y) for x, y in zip(a["repeat_medians_ms"], b["repeat_medians_ms"])],
                ratio_of_centers=b["median_ms"]/a["median_ms"])


def allowed_families(checks, seeds):
    by_candidate = defaultdict(list)
    for r in checks:
        by_candidate[r["family"]].append(r)
    for family, rows in by_candidate.items():
        require(len(rows) == len(seeds) and {r["input_seed"] for r in rows} == set(seeds),
                f"incomplete/duplicate correctness seeds: {family}")
    return {f for f, rows in by_candidate.items() if all(r["correctness_passed"] for r in rows)}


def analyze(run, bundle, prior_path, out):
    # Hash every recovered artifact, including raw outputs, binaries and session records.
    recovered = run / "recovered"
    manifest = read(recovered / "recovery-manifest.json")
    for name, entry in manifest["files"].items():
        path = (recovered / name).resolve()
        require(path.is_relative_to(recovered.resolve()), "manifest path leaves recovery root")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                f"recovered artifact changed: {name}")
    results = recovered / "results"
    plan = read(run / "protocol/plan.json")
    require(plan["shapes"] == SHAPES, "only four development shapes are allowed")
    require(sha(bundle / "plan.json") == sha(run / "protocol/plan.json"), "plan mismatch")
    require(sha(bundle / "files.json") == sha(run / "protocol/files.json"), "package mismatch")
    require(sha(bundle / "inputs.json") == sha(run / "protocol/inputs.json"), "input manifest mismatch")
    files = read(bundle / "files.json")
    for name in ("plan.json", "inputs.json", "predictions.json"):
        require(sha(bundle/name) == files[name], f"sealed protocol artifact changed: {name}")
    require(sha(prior_path) == sha(bundle/"predictions.json"), "historical predictions differ from sealed comparator")
    # Verify sealed source before importing its existing data validator. No CLI is executed.
    source_files = [n for n in files if n.startswith("source/")]
    for name in source_files:
        require(sha(bundle / name) == files[name], f"sealed analysis source changed: {name}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle / "source/EDM-IR/tools"))
    from run_w1_gpu import validate_rows, done_raw_changed
    context = read(results / "context.json")
    receipt = read(recovered / "build/receipt.json")
    require(context["build"] == sha(recovered / "build/receipt.json"), "build receipt changed")
    require(context["package"] == receipt["package"] == sha(bundle / "files.json"), "build/package mismatch")
    for tag, digest in receipt["binaries"].items():
        require(sha(recovered / f"build/w1_{tag}_batch") == digest, "binary mismatch")
    records, checks, measured = [], defaultdict(list), defaultdict(list)
    inputs = read(bundle / "inputs.json")
    for index, job in enumerate(plan["jobs"]):
        path = results / f"collect-{index:03d}.json"
        record = read(path)  # Missing process groups fail closed.
        require(record["job"] == job, "job differs from frozen development protocol")
        require(record["context"] == sha(results / "context.json"), "mixed execution identity")
        require(record["input"] == inputs[job["shape"]][str(job["seed"])], "input metadata mismatch")
        require(not done_raw_changed(path, results), "parsed rows differ from raw stdout")
        validate_rows(record["rows"], job, record["order"], SHAPES[job["shape"]], context["runtime"])
        require(len(record["order"]) == len(set(record["order"])), "duplicate candidate")
        record["job_index"] = index
        records.append(record)
        for position, row in enumerate(record["rows"]):
            item = dict(row, job_index=index, repetition=job["repetition"], order_position=position)
            if job["phase"] == "check":
                checks[job["shape"]].append(item)
            else:
                require(item["correctness_passed"], "measurement contains a numerical failure")
                measured[(job["shape"], row["family"])].append(item)
    eligible = {tag: allowed_families(checks[tag], plan["seeds"]) for tag in SHAPES}
    for record in records:
        job = record["job"]
        expected = job["order"] if job["phase"] == "check" else [f for f in job["order"] if f in eligible[job["shape"]]]
        require(record["order"] == expected, "illegal omission/inclusion or candidate reordering")
        excluded = [] if job["phase"] == "check" else [f for f in job["order"] if f not in eligible[job["shape"]]]
        require(set(record["excluded_by_numerical_check"]) == set(excluded), "incorrect exclusion receipt")
    invalid = sorted({(tag, r["family"]) for tag in SHAPES for r in checks[tag] if not r["correctness_passed"]})
    complete = read(results / "collect-complete.json")
    require(complete["attempted_groups"] == len(plan["jobs"]) and complete["valid_set_only"], "incomplete collection")
    require(complete["invalid"] == [list(i) for i in invalid], "invalid set mismatch")

    candidates, shapes, contrasts = [], {}, []
    for (tag, family), rows in measured.items():
        require(family in eligible[tag], "failed seed has leaked into timing")
        require(len({r["implementation_id"] for r in rows}) == 1, "mixed implementation")
        candidates.append(dict(shape=tag, family=family, phase="pair" if family.startswith("pipeline") else "measure", **summarize(rows)))
    by_key = {(r["shape"], r["family"]): r for r in candidates}
    for tag, shape in SHAPES.items():
        main = sorted((r for r in candidates if r["shape"] == tag and r["phase"] == "measure"), key=lambda r: (r["median_ms"], r["family"]))
        best = main[0]
        host = min((r for r in main if r["family"] in HOSTS), key=lambda r: r["median_ms"])
        fused = min((r for r in main if r["family"] not in HOSTS), key=lambda r: r["median_ms"])
        repeat_winners = [min(main, key=lambda r: (r["repeat_medians_ms"][i], r["family"]))["family"] for i in range(3)]
        per_repeat_best_host = [min(r["repeat_medians_ms"][i] for r in main if r["family"] in HOSTS) for i in range(3)]
        shapes[tag] = dict(shape_PMKN=shape, ranking=[r["family"] for r in main], repeat_winners=repeat_winners,
                           best_fused=fused["family"], best_host=host["family"],
                           best_fused_ms=fused["median_ms"], best_host_ms=host["median_ms"],
                           host_over_fused_ratio=host["median_ms"]/fused["median_ms"],
                           fused_reduction_vs_best_host_pct=100*(1-fused["median_ms"]/host["median_ms"]),
                           fused_reduction_vs_each_repeat_best_host_pct=[100*(1-x/y) for x,y in zip(fused["repeat_medians_ms"],per_repeat_best_host)],
                           legacy_host_over_best_merged_ratio=by_key[(tag,"host")]["median_ms"]/host["median_ms"],
                           full_valid_primary_regret_pct={r["family"]:100*(r["median_ms"]/best["median_ms"]-1) for r in main})
        for q in (1, 2, 4, 8, 16):
            f = "direct" if q == 1 else f"splitk{q}"
            if (tag, f) in by_key and (tag, f+"_groupedn2") in by_key:
                contrasts.append(dict(shape=tag, kind="g2_vs_g1", q=q,
                                      **paired(by_key[(tag,f+"_groupedn2")], by_key[(tag,f)])))
        if (tag,"pipeline") in by_key:
            contrasts.append(dict(shape=tag, kind="pipeline_vs_serial", **paired(by_key[(tag,"pipeline")],by_key[(tag,"pipeline_serial")])))
        for g in (1, 2):
            suffix = "_groupedn2" if g == 2 else ""
            for low, high in ((1,2),(2,4),(4,8)):
                a = f"splitk{high}"+suffix
                b = ("direct" if low == 1 else f"splitk{low}")+suffix
                if (tag,a) in by_key and (tag,b) in by_key:
                    contrasts.append(dict(shape=tag, kind="split_depth", g=g, q_from=low, q_to=high,
                                          **paired(by_key[(tag,a)],by_key[(tag,b)])))

    # Historical predictions are never renamed as calibrated W1 predictions.
    prior = read(prior_path)
    require(all(r["w1_calibrated_prediction_us"] is None for r in prior["rows"]), "unexpected W1 model: explicitly review provenance")
    transfer = []
    for tag, shape in SHAPES.items():
        old = [r for r in prior["rows"] if r["shape"] == shape and r["historical_predicted_us"] is not None]
        raw_choice = min(old, key=lambda r: (r["historical_predicted_us"],r["family"]))["family"]
        common = [r for r in old if (tag,r["family"]) in by_key]
        choice = min(common, key=lambda r: (r["historical_predicted_us"],r["family"]))["family"]
        best_common = min(common, key=lambda r: (by_key[(tag,r["family"])]["median_ms"],r["family"]))["family"]
        transfer.append(dict(shape=tag, interpretation="historical_identity_transfer_only_not_W1_validation",
                             raw_historical_choice=raw_choice, raw_choice_eligible=raw_choice in eligible[tag],
                             choice_after_current_numerical_gate=choice, common_candidate_count=len(common),
                             best_measured_common=best_common,
                             common_regret_pct=100*(by_key[(tag,choice)]["median_ms"]/by_key[(tag,best_common)]["median_ms"]-1),
                             expanded_primary_regret_pct=shapes[tag]["full_valid_primary_regret_pct"][choice],
                             rows=[dict(family=r["family"], predicted_us=r["historical_predicted_us"],
                                        measured_us=1000*by_key[(tag,r["family"])]["median_ms"],
                                        signed_transfer_error_pct=100*(r["historical_predicted_us"]/(1000*by_key[(tag,r["family"])]["median_ms"])-1)) for r in common]))
    failures = [dict(shape=tag, **r) for tag in SHAPES for r in checks[tag] if not r["correctness_passed"]]
    comparisons = []
    for tag in SHAPES:
        lookup = {(r["input_seed"],r["family"]): r for r in checks[tag]}
        for seed in plan["seeds"]:
            for q in (1,2,4,8,16):
                fam = "direct" if q == 1 else f"splitk{q}"
                a,b = lookup[(seed,fam)],lookup[(seed,fam+"_groupedn2")]
                comparisons.append(dict(shape=tag,seed=seed,q=q,
                    identical_error_summary=all(a[k]==b[k] for k in ("max_abs_error","max_rel_error","correctness_passed")),
                    # Check-only success prints default six significant figures
                    # until an earlier failure sets cout precision to 17.
                    compatible_at_recorded_precision=(a["correctness_passed"]==b["correctness_passed"] and
                        all(format(a[k],".6g")==format(b[k],".6g") for k in ("max_abs_error","max_rel_error")))))
    drift = sorted([dict(shape=r["shape"],family=r["family"],**run) for r in candidates for run in r["runs"]],key=lambda r:abs(r["last_vs_first_quarter_pct"]),reverse=True)
    arithmetic = None
    diagnostic_path = out/"rounding-full-m512-seed20261101.json"
    if diagnostic_path.exists():
        diagnostic = read(diagnostic_path)
        require(diagnostic["kind"] == "CPU_arithmetic_diagnostic_not_GPU_measurement" and
                diagnostic["shape_PMKN"] == SHAPES["m512"] and diagnostic["seed"] == 20261101 and
                diagnostic["fraction_of_full_output"] == 1.0 and diagnostic["checked_elements"] == 16777216,
                "unexpected arithmetic replay scope")
        require(diagnostic["reference_max_abs_difference"] == 0, "arithmetic reference mismatch")
        require(diagnostic["diagnostic_source_sha256"] == sha(out/"check_rounding.py"), "arithmetic diagnostic source changed")
        expected_input = inputs["m512"]["20261101"]
        require(diagnostic["input_hashes"] == {"inputs.bin": expected_input["input_sha256"],
                "reference.bin":expected_input["reference_sha256"]}, "arithmetic input mismatch")
        matched = []
        for q in (1,2,4,8,16):
            family = "direct" if q==1 else f"splitk{q}"
            sim = diagnostic["q_results"][str(q)]
            for suffix in ("","_groupedn2"):
                gpu = next(r for r in checks["m512"] if r["family"]==family+suffix and r["input_seed"]==20261101)
                agrees = (format(sim["max_abs_error"],".6g")==format(gpu["max_abs_error"],".6g") and
                          format(sim["max_scaled_error"],".6g")==format(gpu["max_rel_error"],".6g"))
                require(agrees, "arithmetic replay no longer matches GPU summaries")
                matched.append(dict(family=family+suffix,recorded_error_summary_agrees=agrees,
                                    raw_numeric_summary_exact=(sim["max_abs_error"]==gpu["max_abs_error"] and
                                        sim["max_scaled_error"]==gpu["max_rel_error"])))
        arithmetic = dict(kind=diagnostic["kind"],sha256=sha(diagnostic_path),matched_GPU_summaries=matched,
                          limitation="aggregate errors agree; saved GPU element arrays unavailable; no timing simulation")
    audit = dict(scope="four_development_shapes_only_no_fit_no_frozen_shapes_no_GPU",
                 recovery_manifest_sha256=sha(recovered / "recovery-manifest.json"),
                 verified_recovered_files=len(manifest["files"]), verified_sealed_source_files=len(source_files),
                 context_sha256=sha(results / "context.json"),plan_sha256=sha(bundle/"plan.json"),
                 prior_prediction_sha256=sha(prior_path),analysis_source_sha256=sha(Path(__file__)),
                 hardware=context["hardware"],completed_groups=len(records),correctness_rows=sum(map(len, checks.values())),
                 numerical_failure_rows=len(failures),ineligible_combinations=len(invalid),
                 valid_combinations=len(candidates),valid_timing_rows=sum(map(len,measured.values())),
                 raw_timing_samples=sum(len(r["samples_ms"]) for rows in measured.values() for r in rows),
                 timing_seed=20261020,correctness_seeds=plan["seeds"],
                 historical_scored_combinations=sum(r["historical_predicted_us"] is not None for r in prior["rows"]),
                 historical_unscored_combinations=sum(r["historical_predicted_us"] is None for r in prior["rows"]),
                 historical_scored_and_eligible=sum(len(r["rows"]) for r in transfer),
                 forbidden_claims=["held-out generalization", "BF16 results", "serving speedup", "measured overlap fraction", "calibrated W1 selector accuracy"])
    out.mkdir(parents=True,exist_ok=True)
    write(out/"analysis.json",dict(audit=audit,shapes=shapes,candidates=candidates,contrasts=contrasts,
         correctness=dict(ineligible=invalid,failures=failures,checks=checks,g1_g2_comparisons=comparisons),
         historical_transfer=transfer,ordered_drift_diagnostics=drift,CPU_arithmetic_attribution=arithmetic))
    lines = ["# W1 开发数据计算附表", "", "由 analyze.py 生成。所有耗时单位为 ms；中心值为三轮进程中位数的中位数。", "",
             "前三轮分别列出；波动范围 = (最大进程中位数 − 最小进程中位数) / 中心值。", ""]
    for tag in SHAPES:
        lines += [f"## {tag} — [P,M,K,N]={SHAPES[tag]}", "", "|候选|阶段|轮 1|轮 2|轮 3|中心|波动范围|", "|---|---|---:|---:|---:|---:|---:|"]
        for r in sorted((r for r in candidates if r["shape"]==tag),key=lambda r:r["median_ms"]):
            values="|".join(f"{v:.6f}" for v in r["repeat_medians_ms"])
            lines.append(f'|{r["family"]}|{r["phase"]}|{values}|{r["median_ms"]:.6f}|{r["process_range_pct"]:.2f}%|')
        lines += ["", "独立进程最快候选："+", ".join(shapes[tag]["repeat_winners"])+"。", ""]
    lines += ["## 同进程配对：正值表示前者缩短耗时", "", "|形状|比较|轮 1|轮 2|轮 3|", "|---|---|---:|---:|---:|"]
    for r in contrasts:
        if r["kind"] == "split_depth":continue
        values="|".join(f"{v:+.2f}%" for v in r["reduction_a_vs_b_pct"])
        lines.append(f'|{r["shape"]}|{r["a"]} vs {r["b"]}|{values}|')
    lines += ["", "同进程配对仍是按候选顺序先后执行，不代表同时测量或消除了漂移。pipeline 的 pair 阶段晚于 measure，跨阶段对比仅作描述。", ""]
    (out/"TABLES.md").write_text("\n".join(lines))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=REPO/"experiments/2026-11-11-shadeform-w1")
    parser.add_argument("--bundle",type=Path,default=Path("/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01/bundle"))
    parser.add_argument("--prior",type=Path,default=REPO/"experiments/2026-11-01-w1-readiness/prior-predictions.json")
    parser.add_argument("--out",type=Path,default=HERE)
    args=parser.parse_args()
    require(not args.out.resolve().is_relative_to((args.run/"recovered").resolve()), "analysis must not overwrite raw data")
    analyze(args.run.resolve(),args.bundle.resolve(),args.prior.resolve(),args.out.resolve())
