#!/usr/bin/env python3
"""Freeze decisions before new GPU labels; score only a complete matching run.

This tool does not run kernels. Merged host collectors must be implemented and
reviewed before measurement. Synthetic tests exercise the protocol, not speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.host_baseline import (
    PER_SOURCE_CUTLASS, PER_SOURCE_CUBLAS, MERGED_CUTLASS, MERGED_CUBLASLT, WALL_TIMER,
)
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.two_stage_selection import _require_finite, _require_int, parse_schedule_family
from evaluate_factorized_selector import METHODS, fit_models, predict_methods

REPO = Path(__file__).resolve().parents[2]
FAMILIES = ("host", "direct", "splitk2", "splitk4", "splitk8", "splitk16",
            "direct_groupedn2", "splitk2_groupedn2", "splitk4_groupedn2",
            "splitk8_groupedn2", "splitk16_groupedn2")
BASELINES = {"host": PER_SOURCE_CUTLASS, "host_compiler_cublas": PER_SOURCE_CUBLAS,
             "host_merged_cutlass": MERGED_CUTLASS,
             "host_merged_cublaslt": MERGED_CUBLASLT}
ALL_FAMILIES = (*FAMILIES, "host_compiler_cublas", "host_merged_cutlass", "host_merged_cublaslt")
SOURCE_FILES = (
    "EDM-IR/edmir/compiler/host_baseline.py",
    "EDM-IR/edmir/compiler/factorized_selection.py",
    "EDM-IR/edmir/compiler/relative_selection.py",
    "EDM-IR/edmir/compiler/two_stage_selection.py",
    "EDM-IR/edmir/compiler/measurement_dataset.py",
    "EDM-IR/tools/evaluate_factorized_selector.py",
    "EDM-IR/tools/independent_selector_validation.py",
    "EDM-IR/runtime/tools/validate_generated.cpp",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def candidate(shape, family, environment_id):
    p, m, k, n = shape
    q, g = parse_schedule_family(family)
    rho = 1 if family == "host" else (math.ceil(n/128)+g-1)//g
    row = ScheduleCandidate(environment_id, family, p, m, k, n, 2*p*(p-1)*m*k,
                            rho, q, g, 0 if family == "host" else math.ceil(m/64)*rho*q)
    row.validate()
    return row


def freeze(directory: Path, world_size: int):
    if type(world_size) is not int or world_size not in (4, 8):
        raise ValueError("this protocol supports P=4 or P=8")
    p4, calibration, old_scoring, _ = load_identified_selector_dataset(directory)
    # A new run has a new environment identity. The old P8 offset cannot be
    # silently reused on another node/session; this freeze uses no target fit.
    models = fit_models(p4)
    # All previously examined shapes are development data, including former holdouts.
    known = {r.shape[1:] for d in (p4, calibration, old_scoring) for r in d.rows}
    shapes = [(world_size, m, k, n) for m, k, n in product(
        (48, 64, 96, 192, 384), (1536, 3072, 6144), (384, 768, 1536))]
    if any(s[1:] in known for s in shapes):
        raise ValueError("proposed independent workload overlaps examined M/K/N")
    environment_id = f"independent-a100-sxm4-80gb-p{world_size}-v1"
    workload_records = []
    rng = random.Random(20261020)
    for shape in shapes:
        candidates = {name: candidate(shape, name, environment_id) for name in FAMILIES}
        choices, details = predict_methods(models, candidates, PER_SOURCE_CUTLASS)
        orders = []
        for _ in range(5):
            order = list(ALL_FAMILIES)
            rng.shuffle(order)
            orders.append(order)
        workload_records.append({
            "shape": list(shape), "choices": choices, "factorized_details": details,
            "trial_orders": orders, "grouped_scope": "training_M" if shape[1] == 64 else "unseen_M",
        })
    payload = {
        "schema": "edmir.independent-selector-plan.v1",
        "status": "awaiting_new_gpu_measurements", "independent_evidence_available": False,
        "eligible_for_compiler_admission": False,
        "environment_id": environment_id, "world_size": world_size,
        "target_environment_calibration": "none; new-node calibration would require a separately frozen protocol",
        "hardware": {"gpu": "NVIDIA A100-SXM4-80GB", "arch": "sm_80", "sm_count": 108,
                     "link_label": "NV12", "full_peer_access": True},
        "dataset_manifest_sha256": sha256(directory / "manifest.json"),
        "source_sha256": {p: sha256(REPO/p) for p in SOURCE_FILES},
        "baseline_contracts": {n: b.to_dict() for n, b in BASELINES.items()},
        "timing_scope": WALL_TIMER, "latency_unit": "ms", "dtype": "float16",
        "measurement": {"repetitions": 5, "minimum_warmup": 20, "minimum_iterations": 100,
                        "fresh_process_per_repetition": True, "setup_outside_timer": True,
                        "required_families": list(ALL_FAMILIES)},
        "correctness": {"atol": 0.02, "rtol": 0.02, "input_seed": 20261020,
                        "reference": "CPU_FP32_on_identical_FP16_inputs",
                        "relative_denominator": "max(abs(reference), 1)",
                        "rule": "max_abs_error <= atol OR max_rel_error <= rtol; all outputs finite",
                        "checked_elements": "P*P*M*N"},
        "precision": "same FP16 inputs/output and accumulation contract for all candidates",
        "models": {"factorized": models[0].to_dict(), "relative": models[1].to_dict(),
                   "train_best_fixed": models[2]},
        "examined_MKN": [list(s) for s in sorted(known)],
        "workloads": workload_records,
        "predeclared_analysis": {
            "bootstrap_seed": 20261020, "bootstrap_resamples": 2000,
            "interval": "95% percentile, hierarchical resampling of workloads and paired repetitions",
            "primary": "regret against all 14 observed candidates and regression against the fastest of 4 materialized hosts",
            "secondary": "same eligible oracle, grouped eligibility coverage, strata by M and N",
            "promotion": "none automatic; full correctness and matched identities are necessary, not proof of reliability",
            "stop_on_missing_candidate": True,
            "baseline_implementation_status": "legacy measured; merged collectors not implemented by this tool",
        },
    }
    return {"plan_sha256": digest(payload), "plan": payload}


def verify_plan(envelope):
    plan = envelope["plan"]
    if plan["schema"] != "edmir.independent-selector-plan.v1" or digest(plan) != envelope["plan_sha256"]:
        raise ValueError("frozen plan hash mismatch")
    if set(plan["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("frozen source manifest is incomplete")
    for path, expected in plan["source_sha256"].items():
        if sha256(REPO/path) != expected:
            raise ValueError(f"source changed after freezing; preserve/review the old run before creating another plan: {path}")
    if plan["baseline_contracts"] != {n: b.to_dict() for n, b in BASELINES.items()}:
        raise ValueError("baseline contract changed")
    return plan


def read_run(envelope, run_path):
    plan = verify_plan(envelope)
    run = json.loads(run_path.read_text())
    if run.get("schema") != "edmir.independent-selector-run.v1" or run.get("plan_sha256") != envelope["plan_sha256"]:
        raise ValueError("run does not match frozen plan")
    for field in ("timing_scope", "latency_unit", "dtype", "baseline_contracts", "environment_id"):
        if run.get(field) != plan[field]:
            raise ValueError(f"run contract mismatch: {field}")
    env = run["environment"]
    for field, expected in {**plan["hardware"], "world_size": plan["world_size"]}.items():
        if type(env.get(field)) is not type(expected) or env.get(field) != expected:
            raise ValueError(f"run environment mismatch: {field}")
    for field in ("driver_version", "cuda_version", "nccl_version", "nccl_commit",
                  "cutlass_version", "cutlass_commit", "cublas_version", "collector_sha256"):
        if not isinstance(env.get(field), str) or not env[field].strip():
            raise ValueError(f"missing environment fingerprint: {field}")
    if len(env["collector_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in env["collector_sha256"]):
        raise ValueError("invalid collector source hash")
    required = plan["measurement"]
    if run.get("fresh_process_per_repetition") is not True:
        raise ValueError("independent timing repetitions require fresh processes")
    expected = {}
    for work in plan["workloads"]:
        shape = tuple(work["shape"])
        if list(shape[1:]) in plan["examined_MKN"]:
            raise ValueError("independent workload was already examined")
        for repetition, order in enumerate(work["trial_orders"]):
            for position, family in enumerate(order):
                expected[(shape, family, repetition)] = position
    rows, raw_cache, input_hashes = {}, {}, {}
    for row in run["rows"]:
        shape = tuple(row["shape"])
        for v in shape:
            _require_int("shape dimension", v)
        _require_int("repetition", row["repetition"], minimum=0)
        key = (shape, row["family"], row["repetition"])
        if key not in expected or key in rows:
            raise ValueError("unexpected or duplicate measurement")
        _require_int("trial order", row["trial_order"], minimum=0)
        if row["trial_order"] != expected[key]:
            raise ValueError("trial order differs from frozen randomized order")
        if row.get("correctness_passed") is not True:
            raise ValueError("candidate failed or omitted correctness")
        for field in ("max_abs_error", "max_rel_error", "atol", "rtol", "median_ms"):
            _require_finite(field, row[field])
            if row[field] < 0 or (field == "median_ms" and row[field] == 0):
                raise ValueError("invalid accuracy/timing value")
        accuracy = plan["correctness"]
        for field in ("atol", "rtol", "input_seed"):
            if type(row.get(field)) is not type(accuracy[field]) or row.get(field) != accuracy[field]:
                raise ValueError(f"correctness contract mismatch: {field}")
        if row.get("correctness_reference") != accuracy["reference"]:
            raise ValueError("correctness reference differs")
        if row["max_abs_error"] > accuracy["atol"] and row["max_rel_error"] > accuracy["rtol"]:
            raise ValueError("reported numerical errors exceed frozen tolerance")
        _require_int("checked elements", row["checked_elements"])
        _require_int("nonfinite outputs", row["nonfinite_outputs"], minimum=0)
        if row["nonfinite_outputs"] or row["checked_elements"] != shape[0]**2*shape[1]*shape[3]:
            raise ValueError("correctness did not check all finite outputs")
        input_hash = row.get("input_sha256")
        if not isinstance(input_hash, str) or len(input_hash) != 64 or any(c not in "0123456789abcdef" for c in input_hash):
            raise ValueError("input content hash is missing")
        if input_hashes.setdefault(shape, input_hash) != input_hash:
            raise ValueError("candidates/repetitions did not use identical inputs")
        if row["family"] in BASELINES:
            implementation = BASELINES[row["family"]].baseline_id
        else:
            q, g = parse_schedule_family(row["family"])
            implementation = f"direct_peer_cutlass_sm80_q{q}_g{g}"
        if row.get("implementation_id") != implementation:
            raise ValueError("measured implementation does not match the candidate")
        if row["family"] == "host_merged_cublaslt" and (
                not isinstance(row.get("gemm_algorithm_id"), str) or not row["gemm_algorithm_id"].strip()):
            raise ValueError("cuBLASLt algorithm identity is missing")
        for field in ("warmup", "iterations"):
            _require_int(field, row[field])
            if row[field] < required[f"minimum_{field}"]:
                raise ValueError("insufficient warmup/iterations")
        # Check a raw JSONL source line, not just a user-supplied aggregate.
        source = row["source"]
        raw_path = (run_path.parent / source["path"]).resolve()
        if raw_path not in raw_cache:
            raw_cache[raw_path] = (sha256(raw_path), raw_path.read_text().splitlines())
        raw_hash, lines = raw_cache[raw_path]
        if raw_hash != source["sha256"]:
            raise ValueError("raw measurement hash mismatch")
        _require_int("raw line", source["line"])
        if source["line"] > len(lines):
            raise ValueError("raw measurement line is missing")
        raw = json.loads(lines[source["line"]-1])
        if raw != {k: v for k, v in row.items() if k != "source"}:
            raise ValueError("normalized measurement differs from raw result")
        rows[key] = row["median_ms"]
    if rows.keys() != expected.keys():
        raise ValueError(f"incomplete candidate surface: received {len(rows)}, expected {len(expected)}")
    return plan, rows, env


def timing_summary(plan, rows, *, resampled=None):
    result = {method: {"regrets": [], "host_ratios": [], "strong_host_ratios": [],
                       "eligible_regrets": [], "baseline_ratios": {n: [] for n in BASELINES}}
              for method in METHODS}
    jobs = resampled or [(w, list(range(plan["measurement"]["repetitions"]))) for w in plan["workloads"]]
    for work, reps in jobs:
        shape = tuple(work["shape"])
        times = {n: statistics.median(rows[(shape, n, rep)] for rep in reps) for n in ALL_FAMILIES}
        oracle = min(times.values())
        strong_host = min(times[n] for n in BASELINES)
        eligible_oracle = min(times[n] for n, p in work["factorized_details"]["predictions"].items() if p["eligible"])
        for method, selected in work["choices"].items():
            stats = result[method]
            stats["regrets"].append(times[selected]/oracle-1)
            stats["host_ratios"].append(times[selected]/times["host"])
            stats["strong_host_ratios"].append(times[selected]/strong_host)
            stats["eligible_regrets"].append(times[selected]/eligible_oracle-1)
            for name in BASELINES:
                stats["baseline_ratios"][name].append(times[selected]/times[name])
    return {m: {
        "mean_regret_all_candidates": statistics.mean(d["regrets"]),
        "max_regret_all_candidates": max(d["regrets"]),
        "legacy_host_regressions": sum(x > 1+1e-12 for x in d["host_ratios"]),
        "strong_host_regressions": sum(x > 1+1e-12 for x in d["strong_host_ratios"]),
        "mean_ratio_to_strong_host": statistics.mean(d["strong_host_ratios"]),
        "against_each_host": {n: {"mean_latency_ratio": statistics.mean(ratios),
                                  "regressions": sum(x > 1+1e-12 for x in ratios)}
                              for n, ratios in d["baseline_ratios"].items()},
        **({"mean_regret_within_eligible": statistics.mean(d["eligible_regrets"])}
           if m.endswith("_supported") else {}),
    } for m, d in result.items()}


def score(envelope, run_path):
    plan, rows, env = read_run(envelope, run_path)
    point = timing_summary(plan, rows)
    rng = random.Random(plan["predeclared_analysis"]["bootstrap_seed"])
    estimates = {m: {"mean_regret_all_candidates": [], "mean_ratio_to_strong_host": []} for m in METHODS}
    for _ in range(plan["predeclared_analysis"]["bootstrap_resamples"]):
        jobs = [(rng.choice(plan["workloads"]), [rng.randrange(plan["measurement"]["repetitions"])
                 for _ in range(plan["measurement"]["repetitions"])]) for _ in plan["workloads"]]
        metrics = timing_summary(plan, rows, resampled=jobs)
        for method in METHODS:
            for metric in estimates[method]:
                estimates[method][metric].append(metrics[method][metric])
    for method in METHODS:
        point[method]["bootstrap_95pct"] = {}
        for metric, values in estimates[method].items():
            values.sort()
            point[method]["bootstrap_95pct"][metric] = [
                values[int(0.025*(len(values)-1))], values[int(0.975*(len(values)-1))]]
    strata = {}
    for index, name in ((1, "M"), (3, "N")):
        strata[name] = {str(v): timing_summary(
            {**plan, "workloads": [w for w in plan["workloads"] if w["shape"][index] == v]}, rows)
            for v in sorted({w["shape"][index] for w in plan["workloads"]})}
    return {
        "schema": "edmir.independent-selector-score.v1", "status": "new_measurements_scored",
        "eligible_for_compiler_admission": False, "plan_sha256": envelope["plan_sha256"],
        "run_sha256": sha256(run_path), "environment": env,
        "workloads": len(plan["workloads"]), "timing_rows": len(rows),
        "grouped_eligible_workloads": sum(w["factorized_details"]["grouped_eligible"] for w in plan["workloads"]),
        "metrics": point, "strata": strata,
        "limitations": [
            "Intervals describe this grid and timing procedure, not all workloads or architectures.",
            "Collector implementation and correctness tolerances need review; metadata alone cannot certify them.",
            "Run provenance does not prove labels were never viewed; the frozen-before-measurement procedure is required.",
            "Repeated tuning on this run converts it to development data; a revised model needs another untouched test.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("freeze")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--world-size", type=int, choices=(4, 8), required=True)
    build.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("score")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new file")
    result = freeze(args.dataset, args.world_size) if args.command == "freeze" else score(
        json.loads(args.plan.read_text()), args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(args.out)
