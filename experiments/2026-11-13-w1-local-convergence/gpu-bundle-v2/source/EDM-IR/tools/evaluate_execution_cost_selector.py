#!/usr/bin/env python3
"""Finite local evaluation of execution geometry and symmetric host decisions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools"),
               str(ROOT/"experiments/2026-10-20-selector-decision-audit")]
from candidate_model import DecisionModel
from evaluate_factorized_selector import fit_models, predict_methods
from edmir.compiler.execution_cost_selection import ExecutionCostSelector, direction_diagnostics
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate

PRIORITY = {(4, 64, 1024, 512), (4, 64, 4096, 2048),
            (4, 64, 8192, 512), (4, 256, 8192, 512)}
CONTROLS = ("train_best_fixed", "relative_supported", "existing_waves_separate", "interactions_joint")


def summarize(rows, method):
    if not rows:
        raise ValueError("an empty evaluation cannot pass")
    selected = [r["diagnostics"][method] for r in rows]
    return {"workloads": len(rows),
            "mean_regret": sum(r["regret_all_observed"] for r in selected)/len(rows),
            "max_regret": max(r["regret_all_observed"] for r in selected),
            "host_regressions": sum(r["relative_to_host"] > 1e-12 for r in selected),
            "false_host_over_five_percent": sum(r["false_host"] for r in selected),
            "false_fused_over_five_percent": sum(r["false_fused"] for r in selected)}


def evaluate(data, variants, protocol):
    key = (lambda r: r.group_id) if protocol == "leave_group_out" else (lambda r: r.k)
    rows, fits = [], []
    for heldout in sorted({key(r) for r in data.rows}):
        train = replace(data, rows=tuple(r for r in data.rows if key(r) != heldout))
        test = replace(data, rows=tuple(r for r in data.rows if key(r) == heldout))
        assert not {r.shape for r in train.rows} & {r.shape for r in test.rows}
        models = {v["name"]: ExecutionCostSelector.fit(
            train, execution_terms=v["execution_terms"], balanced_host=v["balanced_host"]) for v in variants}
        reference = fit_models(train)
        previous = DecisionModel(train, joint=True, interactions=True)
        waves = DecisionModel(train, joint=False, interactions=False)
        fits.append({"heldout": heldout, "models": {n: m.to_dict() for n, m in models.items()}})
        for surface in test.validate().values():
            first = next(iter(surface.values()))
            candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in surface.items()}
            baseline_choices, _ = predict_methods(reference, candidates, data.baseline)
            choices = {n: baseline_choices[n] for n in CONTROLS[:2]}
            scores = {}
            choices["interactions_joint"], scores["interactions_joint"] = previous.choose(candidates)
            choices["existing_waves_separate"], scores["existing_waves_separate"] = waves.choose(candidates)
            predictions = {}
            for name, model in models.items():
                selected = model.select(candidates, baseline=data.baseline)
                choices[name], scores[name], predictions[name] = selected["choice"], selected["scores"], selected["predictions"]
            times = {n: r.measured_ms for n, r in surface.items()}
            # Controls have the same grouped support contract. A fixed policy
            # always uses a measured ungrouped schedule.
            eligible = tuple(scores["interactions_joint"])
            rows.append({"shape": list(first.shape), "heldout": heldout, "times_ms": times,
                         "choices": choices, "scores": scores, "predictions": predictions,
                         "diagnostics": {n: direction_diagnostics(times, c, tuple(scores.get(n, eligible)))
                                         for n, c in choices.items()}})
    return {"metrics": {n: summarize(rows, n) for n in (*CONTROLS, *(v["name"] for v in variants))},
            "decisions": rows, "fits": fits}


def decision_gate(primary, stress, method):
    metrics = primary["metrics"][method]
    priority = [r for r in primary["decisions"] if tuple(r["shape"]) in PRIORITY]
    if {tuple(r["shape"]) for r in priority} != PRIORITY:
        raise ValueError("all four priority workloads must be scored")
    comparators = [primary["metrics"][n] for n in ("train_best_fixed", "relative_supported", "interactions_joint")]
    checks = {
        "mean_no_worse_than_controls": all(metrics["mean_regret"] <= c["mean_regret"]+1e-12 for c in comparators),
        "max_no_worse_than_controls": all(metrics["max_regret"] <= c["max_regret"]+1e-12 for c in comparators),
        "host_regressions_no_more_than_old_relative": metrics["host_regressions"] <= comparators[1]["host_regressions"],
        "no_priority_false_host": not any(r["diagnostics"][method]["false_host"] for r in priority),
        "no_priority_false_fused": not any(r["diagnostics"][method]["false_fused"] for r in priority),
        "priority_regret_at_most_five_percent": all(r["diagnostics"][method]["regret_all_observed"] <= .05+1e-12 for r in priority),
        "leave_k_max_no_worse_than_fixed": stress["metrics"][method]["max_regret"] <= stress["metrics"]["train_best_fixed"]["max_regret"]+1e-12,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main(design_path, out):
    if out.exists():
        raise ValueError("choose a fresh output")
    start = time.monotonic()
    design = json.loads(design_path.read_text())
    data, _, _, _ = load_identified_selector_dataset(ROOT/design["dataset"])
    primary = evaluate(data, design["variants"], "leave_group_out")
    stress = evaluate(data, design["variants"], "leave_k_out")
    old = json.loads((ROOT/"experiments/2026-10-20-selector-decision-audit/certified_recheck/results.json").read_text())
    for protocol, result in (("primary", primary), ("stress", stress)):
        previous = {tuple(r["shape"]): r for r in old[protocol]["decisions"]}
        for row in result["decisions"]:
            reference = previous[tuple(row["shape"])]
            assert row["times_ms"] == reference["times_ms"]
            assert all(row["choices"][n] == reference["choices"][n] for n in CONTROLS)
        for name in CONTROLS:
            for metric in ("mean_regret", "max_regret", "host_regressions"):
                assert abs(result["metrics"][name][metric]-old[protocol]["metrics"][name][metric]) < 1e-12
    gates = {v["name"]: decision_gate(primary, stress, v["name"]) for v in design["variants"]}
    result = {
        "schema": "edmir.execution-cost-evaluation.v1", "status": "retrospective local development",
        "design_sha256": sha256(design_path),
        "dataset_manifest_sha256": sha256(ROOT/design["dataset"]/"manifest.json"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT/"EDM-IR/edmir/compiler/execution_cost_selection.py")},
        "proposed_model": design["proposed_model"], "controls_reproduced": True,
        "primary": primary, "stress": stress, "gates": gates,
        "local_gate_passed": gates[design["proposed_model"]]["passed"],
        "server_ready": False, "compiler_admission": False,
        "elapsed_seconds": time.monotonic()-start,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"metrics": primary["metrics"], "stress": stress["metrics"], "gates": gates,
                      "priority": [{"shape": r["shape"], "choices": r["choices"], "diagnostics": r["diagnostics"][design["proposed_model"]]}
                                   for r in primary["decisions"] if tuple(r["shape"]) in PRIORITY],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.design, args.out)
