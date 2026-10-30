#!/usr/bin/env python3
"""Finite CPU-only diagnosis and controls; never launch a GPU workload."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.splitk_revision import split_features
from evaluate_factorized_selector import fit_models, predict_methods
from candidate_model import DecisionModel

HERE = Path(__file__).resolve().parent
PRIORITY = {(4, 64, 1024, 512), (4, 64, 4096, 2048),
            (4, 64, 8192, 512), (4, 256, 8192, 512)}


def metrics(rows, method):
    regrets, slower = [], []
    for row in rows:
        t = row["times_ms"]
        chosen = t[row["choices"][method]]
        regrets.append(chosen/min(t.values())-1)
        slower.append(chosen/t["host"]-1)
    return {"mean_regret": sum(regrets)/len(regrets), "max_regret": max(regrets),
            "host_regressions": sum(v > 1e-12 for v in slower)}


def collisions(data):
    cells = defaultdict(list)
    for surface in data.validate().values():
        direct = surface["direct"].measured_ms
        for row in surface.values():
            if not row.is_host and row.n_tiles_per_cta == 1 and row.split_k_slices > 1:
                cells[split_features(row, "waves")].append({
                    "shape": list(row.shape), "q": row.split_k_slices,
                    "measured_ratio_to_direct": row.measured_ms/direct})
    result = []
    for vector, rows in cells.items():
        if len(rows) < 2:
            continue
        ratios = [r["measured_ratio_to_direct"] for r in rows]
        # Identical inputs force identical predictions in this component,
        # regardless of optimizer, ridge, or how many coefficients are fitted.
        spread = math.log(max(ratios)/min(ratios))
        result.append({"features": vector, "rows": rows, "log_ratio_spread": spread,
                       "unavoidable_max_abs_log_error_at_least": spread/2,
                       "opposite_split_vs_direct_winners": min(ratios) < 1 < max(ratios)})
    return sorted(result, key=lambda r: r["log_ratio_spread"], reverse=True)


def evaluate(data, variants, protocol):
    key = (lambda r: r.group_id) if protocol == "leave_group_out" else (lambda r: r.k)
    groups = sorted({key(r) for r in data.rows})
    records, fits = [], []
    for group in groups:
        train = replace(data, rows=tuple(r for r in data.rows if key(r) != group))
        test = replace(data, rows=tuple(r for r in data.rows if key(r) == group))
        if {r.shape for r in train.rows} & {r.shape for r in test.rows}:
            raise ValueError("training/test shape overlap")
        controls = fit_models(train)
        models = {v["name"]: DecisionModel(train, joint=v["joint"], interactions=v["depth_grid_interactions"])
                  for v in variants}
        fits.extend({"fold": group, "variant": name, "gradient_max_abs": model.fit.gradient_max_abs,
                     "design_rank": model.fit.design_rank, "columns": len(model.fit.coefficients)}
                    for name, model in models.items())
        for surface in test.validate().values():
            first = next(iter(surface.values()))
            candidates = {n: ScheduleCandidate.from_measurement(r, test.environment_id)
                          for n, r in surface.items()}
            reference_choices, _ = predict_methods(controls, candidates, test.baseline)
            choices = {n: reference_choices[n] for n in ("train_best_fixed", "relative_supported")}
            scores = {}
            for name, model in models.items():
                choices[name], scores[name] = model.choose(candidates)
            records.append({"shape": list(first.shape), "fold": group, "choices": choices,
                            "times_ms": {n: r.measured_ms for n, r in surface.items()}, "scores": scores})
    methods = ["train_best_fixed", "relative_supported", *(v["name"] for v in variants)]
    return {"metrics": {n: metrics(records, n) for n in methods}, "decisions": records, "fits": fits}


def gate(report, variant):
    score = report["metrics"][variant]
    controls = [report["metrics"][n] for n in ("train_best_fixed", "relative_supported")]
    checks = {
        "mean_le_both_controls": all(score["mean_regret"] <= c["mean_regret"]+1e-12 for c in controls),
        "worst_le_both_controls": all(score["max_regret"] <= c["max_regret"]+1e-12 for c in controls),
        "host_regressions_le_relative": score["host_regressions"] <= controls[1]["host_regressions"],
    }
    failures = []
    for row in report["decisions"]:
        if tuple(row["shape"]) not in PRIORITY:
            continue
        t = row["times_ms"]
        best = min((n for n in row["scores"][variant] if n != "host"), key=t.get, default=None)
        if best and t[best] <= .95*t["host"] and row["choices"][variant] == "host":
            failures.append(row["shape"])
    checks["no_priority_false_host"] = not failures
    return {"passed": all(checks.values()), "checks": checks, "priority_false_host_shapes": failures}


def component_replacement(previous):
    # Measured ratios here are diagnostic counterfactuals. They never enter
    # a training fit or a held-out candidate score above.
    records = []
    for row in previous:
        times = row["times_ms"]
        scores = {"original": {} , "replace_gate": {}, "replace_split": {}, "replace_group": {}}
        for name, pred in row["predictions"].items():
            if not pred["eligible"]:
                continue
            if name == "host":
                for method in scores:
                    scores[method][name] = 0.
                continue
            plain = name.removesuffix("_groupedn2")
            actual = {
                "gate": math.log(times["direct"]/times["host"]),
                "split": math.log(times[plain]/times["direct"]),
                "group": math.log(times[name]/times[plain]),
            }
            scores["original"][name] = pred["log_ratio"]
            for part in actual:
                scores["replace_"+part][name] = pred["log_ratio"]-pred[part]+actual[part]
        choices = {}
        for method, values in scores.items():
            best = min((n for n in values if n != "host"), key=lambda n: (values[n], n))
            choices[method] = best if values[best] < 0 else "host"
        records.append({"shape": row["shape"], "times_ms": times, "choices": choices})
    return {"status": "uses observed labels for error attribution only; not a valid predictor",
            "metrics": {n: metrics(records, n) for n in records[0]["choices"]}, "decisions": records}


def main(out):
    if out.exists():
        raise ValueError("choose a new output path")
    start = time.monotonic()
    design_path = HERE/"design.json"
    design = json.loads(design_path.read_text())
    data, _, _, _ = load_identified_selector_dataset(ROOT/design["dataset"])
    variants = design["variants"]
    primary = evaluate(data, variants, "leave_group_out")
    reference = json.loads((ROOT/"experiments/2026-10-20-local-replay-runtime-checks/historical_replay.json").read_text())
    previous = reference["variants"]["waves_pairwise"]["p4_leave_group_out"]["full_surface"]["decisions"]
    by_shape = {tuple(r["shape"]): r for r in previous}
    for row in primary["decisions"]:
        old = by_shape[tuple(row["shape"])]
        if row["choices"]["existing_waves_separate"] != old["choices"]["factorized_supported"]:
            raise AssertionError("existing model decision was not reproduced")
        for name, value in row["scores"]["existing_waves_separate"].items():
            if not math.isclose(value, old["predictions"][name]["log_ratio"], rel_tol=1e-10, abs_tol=1e-10):
                raise AssertionError("existing model score was not reproduced")
    stress = evaluate(data, variants, "leave_k_out")
    gates = {v["name"]: gate(primary, v["name"]) for v in variants}
    result = {
        "schema": "edmir.selector-decision-audit.v1", "status": "retrospective local development",
        "compiler_admission": False,
        "local_primary_gate_passed": any(g["passed"] for g in gates.values()),
        "server_ready": False,
        "server_decision": "Primary gate alone cannot authorize a server; strict review is required.",
        "design_sha256": sha256(design_path),
        "dataset_manifest_sha256": sha256(ROOT/design["dataset"]/"manifest.json"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), HERE/"candidate_model.py",
            ROOT/"EDM-IR/edmir/compiler/factorized_selection.py",
            ROOT/"EDM-IR/edmir/compiler/splitk_revision.py",
            ROOT/"EDM-IR/edmir/compiler/relative_selection.py")},
        "existing_scores_and_choices_reproduced": True,
        "feature_collisions": collisions(data),
        "component_replacement": component_replacement(previous),
        "primary": primary, "stress": stress, "local_gates": gates,
        "elapsed_seconds": time.monotonic()-start,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"local_primary_gate_passed": result["local_primary_gate_passed"],
                      "server_ready": result["server_ready"], "gates": gates,
                      "metrics": primary["metrics"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
