#!/usr/bin/env python3
"""Verify input/build identity and export only complete, admitted comparisons."""
import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from make_design import DESIGN_ID, ROOT, read, sha
from hw_gate import load_passing_gate
sys.path.append(str(ROOT.parent / "2026-11-20-final-round" / "scripts"))
from common import classify, intervals, write_csv  # noqa: E402

FAILURES = {"numerical_failure", "fallback_detected", "timeout", "oom", "runtime_error",
            "unsupported", "not_run", "infrastructure_failure", "lease_cutoff"}


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


RETRYABLE = {"runtime_error", "timeout", "infrastructure_failure"}
# Pre-registered: a row whose host enqueue time reaches 90% of its wall time is launch-bound.
LAUNCH_BOUND = 0.9


def first_attempt_failed(rows, arms):
    """The one retry rule: the first attempt left an arm unrecorded or hit an infrastructure error."""
    first = [r for r in rows if r.get("attempt", 1) == 1]
    return len(first) < arms or any(r["status"] in RETRYABLE for r in first)


def numerically_valid(row):
    return (row.get("status") == "ok" and finite(row.get("scaled_max_err"))
            and 0 <= row["scaled_max_err"] <= 0.02
            and type(row.get("nonfinite")) is int and row["nonfinite"] == 0
            and row.get("fingerprint_ok") is True)


def check_rows(design, run, source, admission, metrics):
    """Return admitted timings keyed by immutable (process, arm)."""
    d = json.loads((design / "DESIGN.json").read_text())
    if d["design_id"] != DESIGN_ID or any(sha(design/name) != h for name, h in d["files"].items()):
        raise ValueError("design tables changed")
    if run.get("design_sha256") != sha(design / "DESIGN.json") or run.get("origin") not in (
            "device_measurement", "synthetic_test"):
        raise ValueError("run does not belong to this design")
    if not run.get("run_id") or not digest(run.get("hardware_sha256")) or not run.get("harness"):
        raise ValueError("missing run/hardware/harness identity")
    if run["origin"] == "device_measurement":
        gate_path = source / "hw_gate.json"
        if sha(gate_path) != run["hardware_sha256"]:
            raise ValueError("hardware gate digest mismatch")
        gate = load_passing_gate(gate_path)
        if run.get("gpu_uuids")!=gate["gpu_uuids"]:
            raise ValueError("run GPU inventory differs from the hardware gate")
        for name,digest_ in run["artifacts"].items():
            if Path(name).name!=name or sha(source/name)!=digest_:
                raise ValueError(f"run artifact changed: {name}")
        for folder in ("prediction","prediction_calibrated"):
            for name,digest_ in run.get(folder,{}).items():
                if Path(name).name!=name or sha(source/folder/name)!=digest_:
                    raise ValueError(f"frozen prediction changed: {folder}/{name}")
    blocks = read(design / "blocks.csv")
    policy = json.loads((design / "policies.json").read_text())
    planned = {(b["process_id"], b["arm_id"]): b for b in blocks}
    admission_keys = {(b["unit_id"], b["rep"] if b["stage"] == "balanced" else "real", b["arm_id"])
                      for b in blocks}

    def identity(row, unit, seed, arm):
        h = run["inputs"].get(f"{unit}:{seed}")
        b = run["builds"].get(arm)
        if (not digest(h) or not digest(b) or row.get("input_sha256") != h
                or row.get("build_sha256") != b or row.get("run_id") != run["run_id"]
                or row.get("hardware_sha256") != run["hardware_sha256"]
                or row.get("harness") != run["harness"]):
            raise ValueError(f"identity mismatch for {unit}/{seed}/{arm}")

    admitted = {}
    for row in admission:
        unit, seed, arm = row["unit_id"], str(row["input_id"]), row["arm_id"]
        key = (unit, seed, arm)
        if key not in admission_keys or key in admitted:
            raise ValueError(f"invalid or duplicate admission {key}")
        identity(row, unit, seed, arm)
        if row.get("status") not in FAILURES | {"ok"}:
            raise ValueError(f"invalid admission status {key}")
        admitted[key] = numerically_valid(row)
    timings, seen, excluded = {}, set(), set()
    arm_order = defaultdict(int)
    by_process = defaultdict(list)
    for row in metrics:
        key = row["process_id"], row["arm_id"]
        attempt = row.get("attempt", 1)
        if key not in planned or attempt not in (1, 2) or (key, attempt) in seen:
            raise ValueError(f"invalid or duplicate timing {key}")
        seen.add((key, attempt))
        by_process[key[0]].append(row)
        b = planned[key]
        unit, seed, arm = b["unit_id"], b["rep"] if b["stage"] == "balanced" else "real", key[1]
        identity(row, unit, seed, arm)
        if any(str(row.get(f)) != b[f] for f in ("unit_id", "rep", "orientation", "arm_position")):
            raise ValueError(f"timing does not match block {key}")
        arm_order[key[0], attempt] += 1
        if str(arm_order[key[0], attempt]) != b["arm_position"]:
            raise ValueError(f"timing rows violate scheduled arm order {key}")
        if row["status"] in FAILURES:
            if row["status"] in {"numerical_failure", "fallback_detected"}:
                excluded.add((unit, arm))
            continue
        if row["status"] != "ok":
            raise ValueError(f"invalid timing status {key}")
        if row.get("warmup") != policy["warmup"] or row.get("timed") != policy["timed"]:
            raise ValueError(f"sample counts violate protocol {key}")
        value = verify_samples(source,row,"samples","median_wall_ms",policy,key)
        if run["origin"] == "device_measurement":
            verify_samples(source,row,"events","median_event_ms",policy,key)
            verify_samples(source,row,"enqueue","median_enqueue_ms",policy,key)
            counts = row.get("rank_counts",[])
            required = {"input_rows","live_assignments","projection_rows","projection_live_rows","padding_rows"}
            if len(counts)!=8 or any(not required<=c.keys() or
                    any(type(c[f]) is not int or c[f]<0 for f in required) or
                    c["projection_live_rows"]+c["padding_rows"]!=c["projection_rows"] or
                    c["live_assignments"]!=c["projection_live_rows"] for c in counts):
                raise ValueError(f"invalid per-rank row counts {key}")
        if "phases_file" in row:
            verify_phases(source,row,policy,key)
        if admitted.get((unit, seed, arm)):
            timings[key, attempt] = value
    arms_in = defaultdict(int)
    for b in blocks:
        arms_in[b["process_id"]] += 1
    retried = retried_processes(metrics)
    for pid in retried:
        if not first_attempt_failed(by_process[pid], arms_in[pid]):
            raise ValueError(f"retry of {pid} without an infrastructure failure of the first attempt")
    # A retried process contributes only its retry, so arms in one process stay one block.
    timings = {key: v for (key, attempt), v in timings.items()
               if attempt == (2 if key[0] in retried else 1)}
    # A unit that failed any correctness seed cannot provide any timing for that arm.
    excluded |= {(u, a) for u, s, a in admission_keys if not admitted.get((u, s, a))}
    return blocks, policy, {key: v for key, v in timings.items()
                           if (planned[key]["unit_id"], key[1]) not in excluded}


def retried_processes(metrics):
    return {r["process_id"] for r in metrics if r.get("attempt", 1) == 2}


def verify_samples(source,row,label,median_field,policy,key):
    rel = Path(row[label+"_file"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("raw samples must stay under the run directory")
    raw = source / rel
    if not raw.resolve().is_relative_to(source.resolve()) or sha(raw)!=row.get(label+"_sha256"):
        raise ValueError(f"raw {label} hash mismatch {key}")
    samples = json.loads(raw.read_text())
    if (len(samples)!=policy["timed"] or
            any(not isinstance(t,list) or len(t)!=8 or any(not finite(x) or x<=0 for x in t) for t in samples)):
        raise ValueError(f"each {label} iteration requires eight positive rank samples {key}")
    value = statistics.median(map(max,samples))
    if not finite(row.get(median_field)) or not math.isclose(value,row[median_field],rel_tol=1e-7):
        raise ValueError(f"{median_field} does not match max-rank samples {key}")
    return value


def verify_phases(source,row,policy,key):
    """iteration -> rank -> one device-timeline duration per named phase."""
    rel,names = Path(row["phases_file"]),row.get("phase_names",[])
    raw = source/rel
    if (rel.is_absolute() or ".." in rel.parts or not raw.resolve().is_relative_to(source.resolve())
            or sha(raw)!=row.get("phases_sha256")):
        raise ValueError(f"raw phases hash mismatch {key}")
    samples = json.loads(raw.read_text())
    if (not names or names[-1]!="release" or len(samples)!=policy["timed"] or any(
            len(it)!=8 or any(len(r)!=len(names) or any(not finite(x) or x<0 for x in r) for r in it)
            for it in samples)):
        raise ValueError(f"each iteration requires eight ranks with one duration per phase {key}")
    for j,name in enumerate(names):
        value = statistics.median(max(r[j] for r in it) for it in samples)
        if not math.isclose(value,row["phase_ms"][name],rel_tol=1e-7,abs_tol=1e-12):
            raise ValueError(f"phase_ms does not match max-rank phases {key}")


def collect_pairs(blocks):
    units = defaultdict(dict)
    for row in blocks:
        ids = units[row["unit_id"]].setdefault(row["pair_id"], {})
        ids[row["orientation"]] = row["process_id"]
    return units


def paired_values(pairs, arms, timings):
    values = []
    for p in pairs.values():
        if set(p) != {"forward", "reverse"}:
            continue
        if not all((pid, a) in timings for pid in p.values() for a in arms):
            continue
        values.append({a: math.exp(statistics.fmean(math.log(timings[pid, a]) for pid in p.values()))
                       for a in arms})
    return values


def effect_rows(design, blocks, policy, timings):
    units = collect_pairs(blocks)
    stage_of = {b["unit_id"]: b["stage"] for b in blocks}
    out = []
    for c in read(design / "comparisons.csv"):
        a, b = c["baseline_arm"], c["candidate_arm"]
        for unit, pairs in units.items():
            if c["stage"] != stage_of[unit]:
                continue
            v = paired_values(pairs, [a, b], timings)
            ratios = [x[a]/x[b] for x in v]
            directions = {classify(x, policy["tie_band_upper"]) for x in ratios}
            ready = len(v) == policy["pairs_required"]
            outcome = ("incomplete" if not ready else directions.pop() if len(directions) == 1 else "undetermined")
            out.append(dict(comparison_id=c["comparison_id"], stage=c["stage"], unit_id=unit,
                            baseline_arm=a, candidate_arm=b, pairs_complete=len(v), outcome=outcome,
                            ratio=statistics.median(ratios) if ready else None,
                            low=min(ratios) if ready else None, high=max(ratios) if ready else None))
    return out


def crossing_rows(design, effects):
    m = {r["cell_id"]: int(r["m_per_source_destination"]) for r in read(design/"cells.csv")}
    ids = {r["comparison_id"] for r in read(design/"comparisons.csv") if r["crossover"] == "1"}
    groups = defaultdict(list)
    for u in effects:
        if u["comparison_id"] in ids:
            groups[u["comparison_id"]].append((m[u["unit_id"]], u["outcome"]))
    out = []
    for cid, seq in sorted(groups.items()):
        seq.sort()
        all_bins = set(c for _, c in seq)
        found = intervals(seq)
        if "incomplete" in all_bins:
            coverage = "incomplete"
        elif "undetermined" in all_bins or any(x[0] == "loose" for x in found):
            coverage = "unresolved"
        else:
            coverage = "complete"
        out.append(dict(comparison_id=cid, group="GLM", coverage=coverage,
                        intervals=[dict(kind=k, m_low=a, m_high=b, direction=d) for k,a,b,d in found]))
    return out


def score_crossing(measured, pred_rows, bins):
    """Score strict and loose forecasts separately against strict measured reversals."""
    if not pred_rows:
        return "unsupported"
    if measured["coverage"]!="complete":
        return measured["coverage"]
    expected = sorted((int(r["m_low"]),int(r["m_high"]),r["direction"],r["kind"])
                      for r in pred_rows if r["kind"] in {"strict","loose"})
    actual = sorted((r["m_low"],r["m_high"],r["direction"])
                    for r in measured["intervals"] if r["kind"]=="strict")
    if not expected and not any(r["kind"]=="none" for r in pred_rows):
        return "unsupported"
    index = {m:i for i,m in enumerate(bins)}
    if any(lo not in index or hi not in index or index[hi]<=index[lo] or
           (kind=="strict" and index[hi]-index[lo]!=1) for lo,hi,_,kind in expected):
        return "unsupported"
    if len(expected)!=len(actual):
        return "miss"
    for (plo,phi,pd,kind),(alo,ahi,ad) in zip(expected,actual,strict=True):
        if pd!=ad or (kind=="loose" and not plo<=alo<ahi<=phi) or (
                kind=="strict" and abs(index[plo]-index[alo])>1):
            return "miss"
    return "hit_loose" if any(kind=="loose" for *_,kind in expected) else "hit"


def predicted_classes(path):
    """Per-cell predicted class of a prediction already validated by load_prediction."""
    seal = json.loads((path/"PREDICTION_SEAL.json").read_text())
    if sha(path/"predicted_comparisons.csv")!=seal["outputs"].get("predicted_comparisons.csv"):
        raise ValueError("prediction output hash mismatch")
    return {(r["comparison_id"],r["unit_id"]):r["predicted_class"] for r in read(path/"predicted_comparisons.csv")}


def bracket(seq):
    """seq: (m, class). The crossover lies between the last clear win on one side and the
    first clear win on the other; ties and undecided cells may sit inside it."""
    wins = [m for m,c in seq if c=="candidate_wins"]
    losses = [m for m,c in seq if c=="baseline_wins"]
    if not wins and not losses:
        return dict(kind="no_clear_cell")
    if not wins or not losses:
        return dict(kind="none",winner="candidate" if wins else "baseline")
    if max(wins)<min(losses):
        return dict(kind="bracket",m_low=max(wins),m_high=min(losses),direction="candidate_wins->baseline_wins")
    if max(losses)<min(wins):
        return dict(kind="bracket",m_low=max(losses),m_high=min(wins),direction="baseline_wins->candidate_wins")
    return dict(kind="interleaved")


def score_bracket(measured, predicted):
    kinds = {measured["kind"],predicted["kind"]}
    if kinds & {"no_clear_cell","interleaved"}:
        return "unscorable"
    if kinds=={"none"}:
        return "agree_none" if measured["winner"]==predicted["winner"] else "miss_winner"
    if "none" in kinds:
        return "miss_existence"
    if measured["direction"]!=predicted["direction"]:
        return "miss_direction"
    if predicted["m_low"]<=measured["m_low"] and measured["m_high"]<=predicted["m_high"]:
        return "contained"
    if measured["m_low"]<predicted["m_high"] and predicted["m_low"]<measured["m_high"]:
        return "overlap"
    return "disjoint"


def secondary_r1(design, effects, classes):
    """Pre-registered secondary scores: they stay defined when a tie or an undecided cell
    sits at the crossover, where the primary strict-crossing verdict is 'unresolved'."""
    m = {r["cell_id"]:int(r["m_per_source_destination"]) for r in read(design/"cells.csv")}
    out = []
    for c in read(design/"comparisons.csv"):
        if c["crossover"]!="1":
            continue
        cid = c["comparison_id"]
        cells = sorted((m[e["unit_id"]],classes.get((cid,e["unit_id"])),e["outcome"])
                       for e in effects if e["comparison_id"]==cid)
        decided = [(pred,got) for _,pred,got in cells if got in ("candidate_wins","baseline_wins","tie")]
        clear = [(pred,got) for pred,got in decided if "tie" not in (pred,got)]
        measured = bracket([(x,got) for x,_,got in cells])
        predicted = bracket([(x,pred) for x,pred,_ in cells])
        out.append(dict(comparison_id=cid,cells=len(cells),decided=len(decided),
            class_agree=sum(pred==got for pred,got in decided),
            both_clear=len(clear),direction_agree=sum(pred==got for pred,got in clear),
            measured_bracket=measured,predicted_bracket=predicted,
            bracket_verdict=score_bracket(measured,predicted),
            by_cell=[dict(m=x,predicted=pred,measured=got) for x,pred,got in cells]))
    return out


def release_sensitivity(design, blocks, policy, timings, phase):
    """Same-consumer comparisons with the direct arm's release barrier removed. The barrier
    protects buffer reuse; a deployment with double buffering would not pay it."""
    adjusted = {key:(v-phase[key]["release"] if key[1].startswith("D_") and key in phase else v)
                for key,v in timings.items()}
    pairs = {c["comparison_id"] for c in read(design/"comparisons.csv")
             if c["candidate_arm"].startswith("D_") and c["baseline_arm"].startswith("BNEW_")}
    rows = [r for r in effect_rows(design,blocks,policy,adjusted) if r["comparison_id"] in pairs]
    return dict(effects=rows,crossings=crossing_rows(design,rows))


def phase_rows(blocks, timings, phase):
    """Per unit and arm: each phase's median over the processes that count."""
    groups = defaultdict(list)
    unit = {(b["process_id"],b["arm_id"]):b["unit_id"] for b in blocks}
    for key in timings:
        if key in phase:
            groups[unit[key],key[1]].append(phase[key])
    names = ("pack","transport","consumer","release")
    return [dict(unit_id=u,arm_id=arm,processes=len(v),
                 **{n+"_ms":statistics.median(x[n] for x in v) if all(n in x for x in v) else None for n in names})
            for (u,arm),v in sorted(groups.items())]


def load_prediction(path, design, stage):
    if path is None:
        return [], {}
    seal = json.loads((path / "PREDICTION_SEAL.json").read_text())
    if seal.get("stage") != stage:
        raise ValueError("prediction stage mismatch")
    archived = ROOT.parent/"2026-11-20-final-round"
    for name in ("predict.py","common.py"):
        hashes = {h for p,h in seal["inputs"].items() if Path(p).name==name}
        if hashes!={sha(archived/"scripts"/name)}:
            raise ValueError(f"prediction used a different frozen {name}")
    files = ("predicted_crossover.csv", "policy_choice.csv")
    for name in files:
        if sha(path / name) != seal["outputs"].get(name):
            raise ValueError("prediction output hash mismatch")
    for name in ("policies.json", "comparisons.csv", "cells.csv" if stage == "balanced" else "snapshots.csv"):
        expected = {h for p,h in seal["inputs"].items() if Path(p).name == name}
        if expected != {sha(design / name)}:
            raise ValueError("prediction was not made for this design")
    return read(path / files[0]), {r["unit_id"]: r["arm_id"] for r in read(path / files[1]) if r["policy"] == "P0"}


def selection_rows(blocks, policy, timings, choices, overhead):
    rows = []
    units = collect_pairs(blocks)
    stages = {b["unit_id"]: b["stage"] for b in blocks}
    for unit, pairs in units.items():
        stage = stages[unit]
        cfg = policy[stage]
        arms = cfg["selection_set"]
        v = paired_values(pairs, arms, timings)
        for label, rule in cfg["policies"].items():
            arm = choices.get(stage, {}).get(unit) if rule == "sealed_prediction" else rule
            extra = overhead.get(unit,0.) if rule == "sealed_prediction" else 0.
            eligible = len(v) == policy["pairs_required"] and arm in arms and finite(extra) and extra >= 0
            regret = statistics.fmean((p[arm]+extra)/min(p.values())-1 for p in v) if eligible else None
            rows.append(dict(stage=stage, unit=unit, policy=label, chosen_arm=arm, paired=len(v), regret=regret))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, default=ROOT / "design")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pred-cells", type=Path)
    ap.add_argument("--pred-snapshots", type=Path)
    ap.add_argument("--pred-calibrated", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run = json.loads((args.run / "RUN.json").read_text())
    metrics_path = args.run/"metrics.jsonl"
    metrics = jsonl(metrics_path) if metrics_path.exists() else []
    blocks, policy, timings = check_rows(args.design, run, args.run,
        jsonl(args.run / "admission.jsonl"),metrics)
    prediction_dir = args.run/"prediction" if run["origin"]=="device_measurement" else args.pred_cells
    if run["origin"]=="device_measurement" and (args.pred_cells or args.pred_snapshots):
        ap.error("device runs use the prediction frozen in RUN.json")
    pred, c = load_prediction(prediction_dir,args.design,"balanced")
    _, s = load_prediction(args.pred_snapshots, args.design, "snapshot")
    calibrated_dir = (args.run/"prediction_calibrated" if run.get("prediction_calibrated") else None
                      ) if run["origin"]=="device_measurement" else args.pred_calibrated
    if run["origin"]=="device_measurement" and args.pred_calibrated:
        ap.error("device runs use the calibrated prediction frozen in RUN.json")
    pred_cal, c_cal = load_prediction(calibrated_dir,args.design,"balanced")
    effects = effect_rows(args.design, blocks, policy, timings)
    cross = crossing_rows(args.design, effects)
    bins = [int(r["m_per_source_destination"]) for r in read(args.design / "cells.csv")]
    # Two sealed predictions from one frozen predictor: pre-boot scenario parameters, and
    # parameters measured by the node microbenchmark before any formal timing.
    r1 = {}
    for label,rows_,folder in (("scenario",pred,prediction_dir),("calibrated",pred_cal,calibrated_dir)):
        if folder is None:
            continue
        for group in cross:
            support = [r for r in rows_ if all(r[k] == group[k] for k in ("comparison_id","group"))]
            verdict = score_crossing(group, support, bins)
            group["prediction_verdict" if label=="scenario" else "calibrated_verdict"] = verdict
        r1[label] = secondary_r1(args.design,effects,predicted_classes(folder))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "effects.csv", list(effects[0]), effects)
    retried = retried_processes(metrics)
    phase = {(row["process_id"],row["arm_id"]):row["phase_ms"] for row in metrics if "phase_ms" in row
             and row.get("attempt",1)==(2 if row["process_id"] in retried else 1)}
    phases = phase_rows(blocks,timings,phase)
    if phases:
        write_csv(args.out/"phases.csv",list(phases[0]),phases)
    selection = selection_rows(blocks, policy, timings, {"balanced": c, "snapshot": s},
                               run.get("selection_overhead_ms", {}))
    if calibrated_dir is not None:
        selection += [dict(row,policy="P0_CALIBRATED") for row in selection_rows(
            blocks,policy,timings,{"balanced":c_cal},{}) if row["policy"]=="P0" and row["stage"]=="balanced"]
    diagnostics = []
    for row in metrics:
        used = row.get("attempt",1)==(2 if row["process_id"] in retried else 1)
        if not used or (row["process_id"],row["arm_id"]) not in timings or "rank_counts" not in row:
            continue
        counts = row["rank_counts"]
        diagnostics.append(dict(**{k:row[k] for k in ("process_id","unit_id","arm_id",
                "median_wall_ms","median_event_ms","median_enqueue_ms")},
            **{n+"_ms":row.get("phase_ms",{}).get(n) for n in ("pack","transport","consumer","release")},
            enqueue_over_wall=row["median_enqueue_ms"]/row["median_wall_ms"],
            launch_bound=row["median_enqueue_ms"]>=LAUNCH_BOUND*row["median_wall_ms"],
            total_live_rows=sum(c["projection_live_rows"] for c in counts),
            total_dp_pad_rows=sum(c["padding_rows"] for c in counts),
            total_nonlocal_zero_rows=sum(c.get("zeroed_nonlocal_rows",0) for c in counts),
            max_projection_launches=max(c.get("projection_launches",0) for c in counts),
            max_merge_launches=max(c.get("merge_launches",0) for c in counts),
            launch_mode=",".join(sorted({c.get("launch_mode","eager") for c in counts})),
            native_config=counts[0].get("native_tune",{}).get("kind","")))
    if diagnostics:
        write_csv(args.out/"timing_diagnostics.csv",list(diagnostics[0]),diagnostics)
    lock_path = args.run/"ADMISSION_LOCK.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    payload = dict(schema="edmir.glm-path-paper-results.v2", origin=run["origin"],
        run_id=run["run_id"], design_id=DESIGN_ID, design_sha256=run["design_sha256"],
        hardware_sha256=run["hardware_sha256"], effects=effects, crossings=cross,
        r1_secondary=r1,phases=phases,
        release_sensitivity=release_sensitivity(args.design,blocks,policy,timings,phase),
        mixed_launch_mode_rows=sorted(f"{r['process_id']}:{r['arm_id']}" for r in diagnostics
                                      if r["arm_id"]=="NATIVE" and r["launch_mode"]!="graph"),
        static_overhead_scope="presealed choices; excludes any online selection",
        p0_equals_always_slt=len(c)==len(bins) and all(arm=="SLT" for arm in c.values()),
        bridge_enabled=lock.get("bridge_enabled",False),
        retried_processes=sorted(retried),missing=run.get("missing",{}),
        # D/BNEW consumers replay as CUDA graphs; any that fell back to eager launches are listed.
        launch_bound_rows=sorted(f"{r['process_id']}:{r['arm_id']}" for r in diagnostics if r["launch_bound"]),
        eager_consumer_rows=sorted(f"{r['process_id']}:{r['arm_id']}" for r in diagnostics
                                   if r["arm_id"].startswith(("D_","BNEW_")) and r["launch_mode"]!="graph"),
        smoke_failures=run.get("smoke_failures",[]),
        native_config_source={name:r["native_tune"] for name,r in run.get("coverage",{}).items()},
        selection=selection)
    (args.out / "paper_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(f"{sum(r['outcome'] != 'incomplete' for r in effects)}/{len(effects)} complete comparisons; {run['origin']}")


if __name__ == "__main__":
    main()
