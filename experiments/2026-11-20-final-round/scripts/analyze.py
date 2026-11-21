#!/usr/bin/env python3
"""Paired effects, crossover intervals, prediction accuracy (R1) and selection regret (R2).

  python3 analyze.py --design ../design --metrics metrics.jsonl [--metrics more.jsonl] \
      --admission admission.jsonl --pred-cells pred_cells --pred-snapshots pred_snapshots --out analysis

metrics.jsonl, one row per (process_id, arm_id) that was planned in design/blocks.csv:
  process_id, arm_id, status, harness; on status "ok" also median_wall_ms > 0.
admission.jsonl, one row per numerics check:
  unit_id, arm_id, input_id (balanced: the input seed; snapshot: "real"), status.

Rules (fixed before data, see START_HERE.md section 6):
- A pair is one forward and one reverse process of the same (block type, unit, rep). Its log ratio is the
  mean of ln(T_baseline / T_candidate) over the two processes. Both processes must have both arms "ok",
  admitted, not excluded, and on the same harness. A half pair never counts.
- A unit is classified only with all planned pairs complete: all ratios > upper -> candidate_wins,
  all < 1/upper -> baseline_wins, all inside the band -> tie, otherwise undetermined.
- Any status other than "ok" or an operational status in PROCESS_ONLY (for example numerical_failure or
  fallback_detected) excludes that arm on that unit from every comparison.
- Without --admission the run is a preview and nothing is marked admitted.
- Sealed predictions are verified against PREDICTION_SEAL.json before use; a mismatch aborts.
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAR, classify, intervals, read_csv, sha256, write_csv  # noqa: E402

PROCESS_ONLY = {
    "timeout", "timed_out", "oom", "out_of_memory", "runtime_error", "build_failed", "compile_failed",
    "unsupported", "skipped", "not_run", "incomplete", "infrastructure_failure", "profiler_error",
    "lease_cutoff",
}


def jsonl(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{n}: expected a JSON object")
                    yield path, n, row


class Design:
    def __init__(self, d):
        self.blocks = read_csv(d / "blocks.csv")
        self.arms = {a["arm_id"]: a for a in read_csv(d / "arms.csv")}
        self.cells = {c["cell_id"]: c for c in read_csv(d / "cells.csv")}
        self.snapshots = {s["snapshot_id"]: s for s in read_csv(d / "snapshots.csv")}
        self.comparisons = read_csv(d / "comparisons.csv")
        self.policies = json.loads((d / "policies.json").read_text(encoding="utf-8"))
        self.process = {}
        self.planned = set()
        for b in self.blocks:
            meta = {k: b[k] for k in ("stage", "block_type", "round", "unit_id", "rep", "orientation", "pair_id")}
            if self.process.setdefault(b["process_id"], meta) != meta:
                raise ValueError(f"blocks.csv: conflicting metadata for {b['process_id']}")
            key = (b["process_id"], b["arm_id"])
            if key in self.planned:
                raise ValueError(f"blocks.csv: duplicate {key}")
            self.planned.add(key)
        self.pairs = defaultdict(dict)  # pair_id -> orientation -> process_id
        for pid, meta in self.process.items():
            if meta["orientation"] in self.pairs[meta["pair_id"]]:
                raise ValueError(f"blocks.csv: two {meta['orientation']} processes in {meta['pair_id']}")
            self.pairs[meta["pair_id"]][meta["orientation"]] = pid

    def inputs_for(self, unit):
        if unit in self.cells:
            return sorted({self.process[p]["rep"] for p in self.process if self.process[p]["unit_id"] == unit})
        return ["real"]


def read_metrics(design, paths):
    observed = {}
    for path, n, row in jsonl(paths):
        key = (row.get("process_id"), row.get("arm_id"))
        if key not in design.planned:
            raise ValueError(f"{path}:{n}: unplanned process/arm {key}")
        if key in observed:
            raise ValueError(f"{path}:{n}: duplicate process/arm {key}")
        meta = design.process[key[0]]
        for field in ("unit_id", "rep", "orientation"):
            if field in row and str(row[field]) != meta[field]:
                raise ValueError(f"{path}:{n}: {field} does not match blocks.csv for {key}")
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{path}:{n}: missing status")
        value = row.get("median_wall_ms")
        if status == "ok":
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}:{n}: ok requires positive finite median_wall_ms")
            if not row.get("harness"):
                raise ValueError(f"{path}:{n}: ok requires a harness identity")
        observed[key] = dict(status=status, ms=value if status == "ok" else None, harness=row.get("harness"))
    return observed


def read_admission(design, paths):
    if not paths:
        return None
    checks = defaultdict(dict)
    for path, n, row in jsonl(paths):
        try:
            unit, arm, inp, status = row["unit_id"], row["arm_id"], str(row["input_id"]), row["status"]
        except KeyError as e:
            raise ValueError(f"{path}:{n}: missing {e}") from e
        if arm not in design.arms or (unit not in design.cells and unit not in design.snapshots):
            raise ValueError(f"{path}:{n}: unknown unit/arm {unit}/{arm}")
        if inp not in design.inputs_for(unit):
            raise ValueError(f"{path}:{n}: unexpected input {inp} for {unit}")
        if inp in checks[(unit, arm)]:
            raise ValueError(f"{path}:{n}: duplicate admission {unit}/{arm}/{inp}")
        checks[(unit, arm)][inp] = status
    admitted = {}
    for unit in list(design.cells) + list(design.snapshots):
        for arm in design.arms:
            got = checks.get((unit, arm), {})
            admitted[(unit, arm)] = all(got.get(i) == "ok" for i in design.inputs_for(unit))
    return admitted


def exclusions(design, observed):
    out = defaultdict(set)
    for (pid, arm), rec in observed.items():
        if rec["status"] != "ok" and rec["status"] not in PROCESS_ONLY:
            out[(design.process[pid]["unit_id"], arm)].add(rec["status"])
    return out


def usable(design, observed, excluded, admitted, pid, arm):
    unit = design.process[pid]["unit_id"]
    if (pid, arm) not in observed:
        return None, "missing"
    rec = observed[(pid, arm)]
    if rec["status"] != "ok":
        return None, rec["status"]
    if excluded.get((unit, arm)):
        return None, "excluded:" + "|".join(sorted(excluded[(unit, arm)]))
    if admitted is not None and not admitted.get((unit, arm)):
        return None, "not_admitted"
    return rec, ""


def pair_effects(design, observed, excluded, admitted, upper, required):
    pair_rows, unit_rows = [], []
    for c in design.comparisons:
        base, cand = c["baseline_arm"], c["candidate_arm"]
        per_unit = defaultdict(list)
        for pair_id, sides in sorted(design.pairs.items()):
            meta = design.process[next(iter(sides.values()))]
            if meta["stage"] != c["stage"] or meta["block_type"] != c["block_type"]:
                continue
            logs, deltas, reasons, harness = [], [], [], set()
            for orientation in ("forward", "reverse"):
                pid = sides.get(orientation)
                if pid is None:
                    reasons.append(f"{orientation}:unplanned")
                    continue
                rb, why_b = usable(design, observed, excluded, admitted, pid, base)
                rc, why_c = usable(design, observed, excluded, admitted, pid, cand)
                if rb is None or rc is None:
                    reasons.append(f"{orientation}:" + ",".join(x for x in (why_b and f"{base}={why_b}",
                                                                           why_c and f"{cand}={why_c}") if x))
                    continue
                if rb["harness"] != rc["harness"]:
                    reasons.append(f"{orientation}:harness_mismatch")
                    continue
                harness.add(rb["harness"])
                logs.append(math.log(rb["ms"] / rc["ms"]))
                deltas.append(rb["ms"] - rc["ms"])
            complete = len(logs) == 2 and not reasons
            row = dict(comparison_id=c["comparison_id"], unit_id=meta["unit_id"], rep=meta["rep"], pair_id=pair_id,
                       complete=int(complete), harness="|".join(sorted(harness)), reason=" / ".join(reasons),
                       ratio="", delta_ms="")
            if complete:
                lr = statistics.fmean(logs)
                row.update(ratio=f"{math.exp(lr):.6f}", delta_ms=f"{statistics.fmean(deltas):.6f}")
                per_unit[meta["unit_id"]].append((lr, statistics.fmean(deltas)))
            else:
                per_unit.setdefault(meta["unit_id"], [])
            pair_rows.append(row)
        for unit, pairs in sorted(per_unit.items()):
            ratios = [math.exp(lr) for lr, _ in pairs]
            if len(pairs) < required:
                cls = "incomplete"
            else:
                classes = {classify(r, upper) for r in ratios}
                cls = classes.pop() if len(classes) == 1 else "undetermined"
            unit_rows.append(dict(
                comparison_id=c["comparison_id"], family=c["family"], stage=c["stage"], unit_id=unit,
                baseline_arm=base, candidate_arm=cand, pairs_complete=len(pairs), pairs_required=required,
                median_ratio=f"{math.exp(statistics.median(lr for lr, _ in pairs)):.6f}" if pairs else "",
                min_ratio=f"{min(ratios):.6f}" if pairs else "", max_ratio=f"{max(ratios):.6f}" if pairs else "",
                median_delta_ms=f"{statistics.median(d for _, d in pairs):.6f}" if pairs else "",
                unit_class=cls, admission_verified=int(admitted is not None)))
    return pair_rows, unit_rows


def measured_crossover(design, unit_rows):
    seqs = defaultdict(list)
    crossover_ids = {c["comparison_id"] for c in design.comparisons if c["crossover"] == "1"}
    for r in unit_rows:
        if r["comparison_id"] in crossover_ids and r["unit_id"] in design.cells:
            cell = design.cells[r["unit_id"]]
            seqs[(r["comparison_id"], cell["width_id"])].append((int(cell["m_per_source_destination"]), r["unit_class"]))
    rows = []
    for (cid, group), seq in sorted(seqs.items()):
        seq.sort()
        found = intervals(seq)
        counts = Counter(c for _, c in seq)
        summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if not found:
            rows.append(dict(comparison_id=cid, group=group, kind="none", m_low="", m_high="", direction="",
                             all_bins_complete=int("incomplete" not in counts), class_counts=summary))
        for kind, lo, hi, direction in found:
            rows.append(dict(comparison_id=cid, group=group, kind=kind, m_low=lo, m_high=hi, direction=direction,
                             all_bins_complete=int("incomplete" not in counts), class_counts=summary))
    return rows, seqs


def load_sealed(directory):
    if directory is None:
        return None
    seal = json.loads((directory / "PREDICTION_SEAL.json").read_text(encoding="utf-8"))
    for name, digest in seal["outputs"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"{directory / name} changed after it was sealed")
    return dict(seal=seal, comparisons=read_csv(directory / "predicted_comparisons.csv"),
                crossover=read_csv(directory / "predicted_crossover.csv"),
                choice={r["unit_id"]: r["arm_id"] for r in read_csv(directory / "policy_choice.csv") if r["policy"] == "P0"})


def r1_rows(measured_rows, seqs, pred):
    if pred is None:
        return []
    bins = sorted({m for seq in seqs.values() for m, _ in seq})
    index = {m: i for i, m in enumerate(bins)}
    predicted = defaultdict(list)
    for r in pred["crossover"]:
        if r["kind"] != "none":
            predicted[(r["comparison_id"], r["group"])].append(int(r["m_low"]))
    measured = defaultdict(list)
    complete = {}
    for r in measured_rows:
        key = (r["comparison_id"], r["group"])
        complete[key] = r["all_bins_complete"] == 1
        if r["kind"] == "strict":
            measured[key].append(int(r["m_low"]))
    rows = []
    for key in sorted(complete):
        p, m = predicted.get(key, []), measured.get(key, [])
        if not complete[key] and not m:
            verdict = "incomplete"
        elif not m:
            verdict = "hit" if not p else "miss"
        else:
            verdict = "hit" if any(abs(index[a] - index[b]) <= 1 for a in m for b in p) else "miss"
        rows.append(dict(comparison_id=key[0], group=key[1], predicted_m_low="|".join(map(str, p)) or "none",
                         measured_strict_m_low="|".join(map(str, m)) or "none", verdict=verdict))
    return rows


def sign_rows(unit_rows, preds):
    rows = []
    for stage, pred in preds.items():
        if pred is None:
            continue
        lookup = {(r["comparison_id"], r["unit_id"]): r["predicted_class"] for r in pred["comparisons"]}
        tally = defaultdict(Counter)
        for r in unit_rows:
            key = (r["comparison_id"], r["unit_id"])
            if key in lookup and r["unit_class"] in (*CLEAR, "tie"):
                tally[r["comparison_id"]]["agree" if lookup[key] == r["unit_class"] else "disagree"] += 1
        for cid, t in sorted(tally.items()):
            n = t["agree"] + t["disagree"]
            rows.append(dict(stage=stage, comparison_id=cid, units_classified=n, agree=t["agree"],
                             agreement=f"{t['agree'] / n:.4f}" if n else ""))
    return rows


def unit_arm_times(design, observed, excluded, admitted, stage):
    times = defaultdict(lambda: defaultdict(list))
    harness = defaultdict(set)
    for (pid, arm), rec in observed.items():
        meta = design.process[pid]
        if meta["stage"] != stage or meta["block_type"] != "core":
            continue
        r, _ = usable(design, observed, excluded, admitted, pid, arm)
        if r is not None:
            times[meta["unit_id"]][arm].append(r["ms"])
            harness[meta["unit_id"]].add(r["harness"])
    return {u: {a: statistics.median(v) for a, v in arms.items()} for u, arms in times.items()}, harness


def regret(design, observed, excluded, admitted, stage, pred):
    cfg = design.policies[stage]
    units = sorted(design.cells if stage == "balanced" else design.snapshots)
    est, harness = unit_arm_times(design, observed, excluded, admitted, stage)
    staged_rep = next((a for a in cfg["staged_representative_priority"]
                       if admitted is not None and all(admitted.get((u, a)) for u in units)), None)
    rows = []
    for unit in units:
        t = est.get(unit, {})
        have = all(a in t for a in cfg["selection_set"]) and len(harness.get(unit, ())) == 1
        oracle_arm = min(cfg["selection_set"], key=lambda a: t[a]) if have else ""
        for policy, rule in cfg["policies"].items():
            if rule == "sealed_prediction":
                arm = pred["choice"].get(unit, "") if pred else ""
            elif rule == "staged_representative":
                arm = staged_rep or ""
            else:
                arm = rule
            ok = have and arm in t
            rows.append(dict(stage=stage, unit_id=unit,
                             cohort=design.cells[unit]["cohort"] if unit in design.cells else "real",
                             policy=policy, chosen_arm=arm, oracle_arm=oracle_arm,
                             regret=f"{t[arm] / t[oracle_arm] - 1:.6f}" if ok else "",
                             complete=int(ok)))
    return rows


def regret_summary(rows, policies):
    out = []
    for stage in sorted({r["stage"] for r in rows}):
        cfg = policies[stage]
        scopes = (("all", lambda r: True), ("unseen_only", lambda r: r["cohort"] in ("unseen", "real")))
        for scope, keep in scopes:
            means = {}
            for policy in cfg["policies"]:
                sel = [r for r in rows if r["stage"] == stage and r["policy"] == policy and keep(r)]
                vals = [float(r["regret"]) for r in sel if r["complete"]]
                full = bool(sel) and len(vals) == len(sel)
                means[policy] = statistics.fmean(vals) if full else None
                out.append(dict(stage=stage, scope=scope, policy=policy, units=len(sel), complete=len(vals),
                                mean_regret=f"{means[policy]:.6f}" if full else "",
                                max_regret=f"{max(vals):.6f}" if full else "", verdict=""))
            p0 = [r for r in out if r["stage"] == stage and r["scope"] == scope and r["policy"] == "P0"][0]
            if means.get("P0") is None:
                p0["verdict"] = "incomplete"
            else:
                r2a = means["P0"] <= policies["r2a_mean_max"] and float(p0["max_regret"]) <= policies["r2a_max_max"]
                rivals = [means[c] for c in cfg["r2b_competitors"]]
                r2b = None if any(v is None for v in rivals) else means["P0"] < min(rivals)
                p0["verdict"] = f"R2a={'pass' if r2a else 'fail'} R2b={'incomplete' if r2b is None else ('pass' if r2b else 'fail')}"
    return out


def coverage(design, observed, excluded, admitted):
    planned = Counter()
    status = defaultdict(Counter)
    for pid, arm in design.planned:
        m = design.process[pid]
        key = (m["stage"], m["block_type"], m["unit_id"], arm)
        planned[key] += 1
        if (pid, arm) in observed:
            status[key][observed[(pid, arm)]["status"]] += 1
    rows = []
    for key in sorted(planned):
        stage, bt, unit, arm = key
        rows.append(dict(stage=stage, block_type=bt, unit_id=unit, arm_id=arm, planned=planned[key],
                         observed=sum(status[key].values()), ok=status[key]["ok"],
                         excluded="|".join(sorted(excluded.get((unit, arm), ()))),
                         admitted="" if admitted is None else int(bool(admitted.get((unit, arm)))),
                         statuses=json.dumps(status[key], sort_keys=True)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, action="append", required=True)
    ap.add_argument("--admission", type=Path, action="append", default=[])
    ap.add_argument("--pred-cells", type=Path)
    ap.add_argument("--pred-snapshots", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    design = Design(args.design)
    upper, required = design.policies["tie_band_upper"], design.policies["pairs_required"]
    observed = read_metrics(design, args.metrics)
    admitted = read_admission(design, args.admission)
    excluded = exclusions(design, observed)
    preds = {"balanced": load_sealed(args.pred_cells), "snapshot": load_sealed(args.pred_snapshots)}

    pairs, units = pair_effects(design, observed, excluded, admitted, upper, required)
    crossing, seqs = measured_crossover(design, units)
    r1 = r1_rows(crossing, seqs, preds["balanced"])
    signs = sign_rows(units, preds)
    reg = regret(design, observed, excluded, admitted, "balanced", preds["balanced"]) + \
        regret(design, observed, excluded, admitted, "snapshot", preds["snapshot"])
    summary = regret_summary(reg, design.policies)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pair_effects.csv", ("comparison_id", "unit_id", "rep", "pair_id", "complete", "harness",
                                         "reason", "ratio", "delta_ms"), pairs)
    write_csv(out / "unit_effects.csv", ("comparison_id", "family", "stage", "unit_id", "baseline_arm",
                                         "candidate_arm", "pairs_complete", "pairs_required", "median_ratio",
                                         "min_ratio", "max_ratio", "median_delta_ms", "unit_class",
                                         "admission_verified"), units)
    write_csv(out / "crossover_measured.csv", ("comparison_id", "group", "kind", "m_low", "m_high", "direction",
                                               "all_bins_complete", "class_counts"), crossing)
    write_csv(out / "r1_crossover.csv", ("comparison_id", "group", "predicted_m_low", "measured_strict_m_low",
                                         "verdict"), r1)
    write_csv(out / "r1_sign_agreement.csv", ("stage", "comparison_id", "units_classified", "agree", "agreement"), signs)
    write_csv(out / "r2_regret.csv", ("stage", "unit_id", "cohort", "policy", "chosen_arm", "oracle_arm", "regret",
                                      "complete"), reg)
    write_csv(out / "r2_summary.csv", ("stage", "scope", "policy", "units", "complete", "mean_regret",
                                       "max_regret", "verdict"), summary)
    write_csv(out / "coverage.csv", ("stage", "block_type", "unit_id", "arm_id", "planned", "observed", "ok",
                                     "excluded", "admitted", "statuses"), coverage(design, observed, excluded, admitted))

    lines = [f"observed {len(observed)}/{len(design.planned)} planned process/arm timings",
             "admission: " + ("PREVIEW (no admission file)" if admitted is None else "applied")]
    for stage, p in preds.items():
        lines.append(f"{stage} predictions: " + ("not supplied" if p is None else f"sealed, params {p['seal']['params_source']}"))
    lines.append("crossover (measured):")
    lines += [f"  {r['comparison_id']} {r['group']}: {r['kind']} {r['m_low']}-{r['m_high']} {r['direction']} [{r['class_counts']}]"
              for r in crossing]
    lines.append("R1:")
    lines += [f"  {r['comparison_id']} {r['group']}: predicted {r['predicted_m_low']} measured {r['measured_strict_m_low']} -> {r['verdict']}"
              for r in r1]
    lines.append("R2:")
    lines += [f"  {r['stage']} {r['scope']} {r['policy']}: mean {r['mean_regret'] or '-'} max {r['max_regret'] or '-'} "
              f"({r['complete']}/{r['units']}) {r['verdict']}" for r in summary]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
