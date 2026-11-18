#!/usr/bin/env python3
"""Run the sealed precision package in fixed stages, with diagnostics between them."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MAX_BUDGET_USD = 15  # User's explicit limit for this window, 2026-11-14.
ORDER = ("m512", "large_n", "m256", "grouped")
TRACES = {
    "m512": ["m512-fp32-q8-g2"],
    "large_n": ["large-n-q2", "large-n-q8-g2"],
    "m256": [f"m256-q{q}-g{g}" for q in (4, 8) for g in (1, 2)],
    "grouped": [],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_plan(plan, source, bundle):
    if (plan["source_manifest_sha256"] != sha(bundle / "files.json") or
            plan["source_plan_sha256"] != sha(bundle / "plan.json")):
        raise ValueError("sealed source changed")
    if plan["shapes"] != source["shapes"] or plan["seeds"] != source["seeds"]:
        raise ValueError("shape or seed changed")
    if [s["name"] for s in plan["stages"]] != list(ORDER):
        raise ValueError("stage order changed")
    indices = []
    for stage in plan["stages"]:
        chosen = stage["source_job_indices"]
        if stage["jobs"] != [source["jobs"][i] for i in chosen]:
            raise ValueError("job differs from sealed source")
        if any(j["shape"] != stage["name"] for j in stage["jobs"]):
            raise ValueError("stage shape mismatch")
        if stage["traces"] != TRACES[stage["name"]]:
            raise ValueError("trace order changed")
        indices.extend(chosen)
    if sorted(indices) != list(range(len(source["jobs"]))):
        raise ValueError("source jobs missing or duplicated")


def cutoff_from_guard(guard, now):
    if guard.get("authorization_confirmed") is not True or guard.get("platform_verified") is not True:
        raise ValueError("current authorization and platform stop rule must be verified")
    if not guard.get("instance_id"):
        raise ValueError("missing current instance identity")
    names = ("billed_from_epoch", "hourly_total_usd", "budget_usd", "platform_stop_epoch")
    if any(type(guard.get(k)) not in (int, float) or not math.isfinite(guard[k]) for k in names):
        raise ValueError("missing or nonfinite cost/time value")
    start, hourly, budget, end = (guard[k] for k in names)
    if hourly <= 0 or not 0 < budget <= MAX_BUDGET_USD or not start <= now < end <= start + 3600:
        raise ValueError("invalid or excessive time/cost window")
    if (end - start) * hourly / 3600 > budget:
        raise ValueError("platform time rule exceeds authorized budget")
    cutoff = end - 600
    if now >= cutoff:
        raise TimeoutError("recovery reserve reached")
    return cutoff


def run_stage(w1, args, source, records, stage, clock, context):
    stage_args = SimpleNamespace(out=args.out / stage["name"], inputs=args.inputs,
                                build_dir=args.build_dir, action="pilot", trace_case=None)
    stage_plan = dict(source, jobs=stage["jobs"])
    w1.collect(stage_args, stage_plan, records, clock, context)
    stage_args.action = "collect"
    w1.collect(stage_args, stage_plan, records, clock, context)
    traces = {}
    for case in stage["traces"]:
        dest = stage_args.out / "diagnostic" / case
        failure = stage_args.out / f"{case}-failure.json"
        if dest.exists() or failure.exists():
            receipt, report = dest / "receipt.json", dest / "timeline.nsys-rep"
            valid = receipt.is_file() and report.is_file()
            if valid:
                valid = json.loads(receipt.read_text())["report_sha256"] == sha(report)
            traces[case] = "captured" if valid else "failed_preserved_no_overwrite"
            continue
        if clock.end - time.time() < 125:
            traces[case] = "not_started_time_budget"
            continue
        stage_args.trace_case = case
        try:
            w1.trace(stage_args, stage_plan, clock, context)
            traces[case] = "captured_requires_local_review"
        except Exception as error:
            traces[case] = "failed_or_candidate_not_admitted"
            w1.save(failure,
                    {"case": case, "error": str(error), "diagnostic_only": True})
            base = {k: v for k, v in context.items() if k != "window_identity"}
            if w1.preflight(args, source, clock) != base:
                raise ValueError("environment changed after trace failure")
    return traces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "inputs", "build-dir", "out", "guard"):
        p.add_argument("--" + name, type=Path, required=name == "bundle")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, Path): setattr(args, k, v.resolve())
    plan = json.loads((ROOT / "queue.json").read_text())
    source = json.loads((args.bundle / "plan.json").read_text())
    validate_plan(plan, source, args.bundle)
    if args.dry_run:
        print(json.dumps({"status": "plan_validated_gpu_not_run", "stages": list(ORDER),
                          "jobs": sum(len(s["jobs"]) for s in plan["stages"]),
                          "trace_cases": sum(len(s["traces"]) for s in plan["stages"])}))
        return
    if any(getattr(args, k) is None for k in ("inputs", "build_dir", "out", "guard")):
        p.error("execution requires inputs, build-dir, out, and verified current guard")
    guard = json.loads(args.guard.read_text())
    cutoff = cutoff_from_guard(guard, time.time())
    sys.path.insert(0, str(args.bundle / "source/EDM-IR/tools"))
    import run_w1_gpu as w1
    clock = w1.Deadline(cutoff)
    _, records = w1.verify(args.bundle, args.inputs)
    context = w1.preflight(args, source, clock)
    context["window_identity"] = {"queue": sha(ROOT / "queue.json"), "runner": sha(__file__),
                                  "instance_id": guard["instance_id"]}
    args.out.mkdir(parents=True, exist_ok=True)
    intent = {"plan": plan, "context": context, "guard": guard, "cutoff_epoch": cutoff}
    path = args.out / "intent.json"
    if path.exists() and json.loads(path.read_text()) != intent:
        raise ValueError("resume identity differs")
    w1.save(path, intent)
    status = {"state": "running", "completed_stages": [], "traces": {}, "GPU_started_by_runner": False}
    try:
        for stage in plan["stages"]:
            if clock.end - time.time() < 30: raise TimeoutError("collection cutoff reached")
            traces = run_stage(w1, args, source, records, stage, clock, context)
            status["completed_stages"].append(stage["name"])
            status["traces"].update(traces)
            w1.save(args.out / "status.json", status)
            print("Completed stage:", stage["name"], traces, flush=True)
        status["state"] = "timings_complete_diagnostics_require_review"
    except BaseException as error:
        status.update(state="interrupted", error=str(error))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        w1.save(args.out / "status.json", status)


if __name__ == "__main__": main()
