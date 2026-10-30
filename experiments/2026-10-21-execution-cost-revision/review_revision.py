#!/usr/bin/env python3
"""Post-fit diagnostics for the fixed execution_balanced proposal; no tuning."""
from dataclasses import replace
from itertools import combinations
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools"),
               str(ROOT/"experiments/2026-10-20-selector-decision-audit")]
from edmir.compiler.execution_cost_selection import ExecutionCostSelector, direction_diagnostics
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from evaluate_execution_cost_selector import PRIORITY
from robustness_review import force_choice


def response_map(data, model):
    """Reconstruct normal equations independently of training_pairs/fit()."""
    rows = list(data.rows)
    index = {(r.shape, r.family): i for i, r in enumerate(rows)}
    width = len(model.coefficients)
    normal, projection = np.zeros((width, width)), np.zeros((width, len(rows)))
    xs = []
    for surface in data.validate().values():
        fused_count = len(surface)-1
        for left, right in combinations(surface.values(), 2):
            if model.balanced_host:
                weight = (.5/fused_count if left.is_host or right.is_host
                          else .5/math.comb(fused_count, 2))
                if fused_count == 1:
                    weight = 1.
            else:
                weight = 1/math.comb(len(surface), 2)
            x = np.asarray(model.transformed(left))-model.transformed(right)
            normal += weight*np.outer(x, x)
            projection[:, index[(left.shape, left.family)]] += weight*x
            projection[:, index[(right.shape, right.family)]] -= weight*x
            xs.append(math.sqrt(weight)*x)
    penalty = np.ones(width)
    penalty[0] = 0
    normal += np.diag(penalty)
    response = np.linalg.solve(normal, projection)
    labels = np.log([r.measured_ms for r in rows])
    np.testing.assert_allclose(response@labels, model.coefficients, atol=1e-10, rtol=1e-10)
    # Check per-workload scaling invariance, not just a common global shift.
    for shape in {r.shape for r in rows}:
        np.testing.assert_allclose(response@np.array([r.shape == shape for r in rows]), 0, atol=1e-10)
    return rows, response, {
        "rank": int(np.linalg.matrix_rank(xs)), "columns": width,
        "design_condition_number": float(np.linalg.cond(xs)),
        "regularized_condition_number": float(np.linalg.cond(normal)),
        "coefficient_error": float(np.max(np.abs(response@labels-model.coefficients))),
    }


def sensitivity(data):
    result = []
    for surface in data.validate().values():
        first = next(iter(surface.values()))
        if first.shape not in PRIORITY:
            continue
        train = replace(data, rows=tuple(r for r in data.rows if r.group_id != first.group_id))
        model = ExecutionCostSelector.fit(train)
        candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in surface.items()}
        selected = model.select(candidates, baseline=data.baseline)
        scores = selected["scores"]
        rows, response, numerical = response_map(train, model)
        influence = {n: np.asarray(model.transformed(candidates[n]))@response for n in scores}
        labels = np.log([r.measured_ms for r in rows])
        for n, h in influence.items():
            assert abs(float(h@labels)-scores[n]) < 1e-10
        best = min(r.measured_ms for r in surface.values())
        targets = []
        for name in scores:
            regret = surface[name].measured_ms/best-1
            if regret <= .05:
                continue
            certificate, delta = force_choice(scores, influence, name)
            entry = {"target": name, "observed_regret": regret, **certificate}
            if delta is not None:
                altered = replace(train, rows=tuple(replace(r, measured_ms=r.measured_ms*math.exp(float(d)))
                                                    for r, d in zip(rows, delta)))
                actual = ExecutionCostSelector.fit(altered).select(candidates, baseline=data.baseline)
                assert actual["choice"] == name
                score_error = max(abs(actual["scores"][n]-(s+float(influence[n]@delta))) for n, s in scores.items())
                assert score_error < 1e-9
                entry.update(refit_confirmed_choice=actual["choice"], refit_score_max_error=score_error,
                             witness_log_changes=delta.tolist())
            targets.append(entry)
        successful = [r for r in targets if r["success"]]
        smallest = min(successful, key=lambda r: r["epsilon_log"]) if successful else None
        result.append({"shape": list(first.shape), "choice": selected["choice"], "numerical": numerical,
                       "training_catalog": [{"shape": list(r.shape), "family": r.family} for r in rows],
                       "targets": targets,
                       "minimum_harmful_perturbation": {k: v for k, v in smallest.items() if k != "witness_log_changes"} if smallest else None})
    return result


def component_diagnosis(primary):
    records = []
    for row in primary["decisions"]:
        scores, times = row["scores"]["execution_balanced"], row["times_ms"]
        methods = {n: {} for n in ("original", "replace_gate", "replace_split", "replace_group")}
        detail = {}
        for name, score in scores.items():
            if name == "host":
                for values in methods.values():
                    values[name] = 0.
                continue
            plain = name.removesuffix("_groupedn2")
            predicted = {"gate": scores["direct"], "split": scores[plain]-scores["direct"], "group": score-scores[plain]}
            observed = {"gate": math.log(times["direct"]/times["host"]), "split": math.log(times[plain]/times["direct"]),
                        "group": math.log(times[name]/times[plain])}
            methods["original"][name] = score
            for part in predicted:
                methods["replace_"+part][name] = score-predicted[part]+observed[part]
            detail[name] = {"predicted_log_components": predicted, "observed_log_components": observed}
        choices, diagnostics = {}, {}
        for method, values in methods.items():
            best = min((n for n in values if n != "host"), key=lambda n: (values[n], n), default=None)
            choices[method] = best if best is not None and values[best] < 0 else "host"
            diagnostics[method] = direction_diagnostics(times, choices[method], tuple(values))
        records.append({"shape": row["shape"], "choices": choices, "diagnostics": diagnostics,
                        "components": detail if tuple(row["shape"]) in PRIORITY else None})
    return {"note": "Observed labels replace one component only to attribute errors; these are invalid deployable predictors.",
            "metrics": {m: {"mean_regret": sum(r["diagnostics"][m]["regret_all_observed"] for r in records)/len(records),
                            "max_regret": max(r["diagnostics"][m]["regret_all_observed"] for r in records)} for m in methods},
            "records": records}


def deletion_stability(data, primary):
    by_shape = {tuple(r["shape"]): {"shape": r["shape"], "original_choice": r["choices"]["execution_balanced"], "deletions": []}
                for r in primary["decisions"]}
    groups = sorted({r.group_id for r in data.rows})
    for heldout in groups:
        test = replace(data, rows=tuple(r for r in data.rows if r.group_id == heldout))
        for deleted in groups:
            if heldout == deleted:
                continue
            train = replace(data, rows=tuple(r for r in data.rows if r.group_id not in (heldout, deleted)))
            model = ExecutionCostSelector.fit(train)
            for surface in test.validate().values():
                first = next(iter(surface.values()))
                candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in surface.items()}
                chosen = model.select(candidates, baseline=data.baseline)["choice"]
                by_shape[first.shape]["deletions"].append({"removed_group": deleted, "choice": chosen,
                    "regret": surface[chosen].measured_ms/min(r.measured_ms for r in surface.values())-1})
    for row in by_shape.values():
        row["choice_counts"] = dict(Counter(d["choice"] for d in row["deletions"]))
        row["all_deletions_keep_choice"] = all(d["choice"] == row["original_choice"] for d in row["deletions"])
    return list(by_shape.values())


def main(results, out):
    if out.exists():
        raise ValueError("choose a new output")
    start = time.monotonic()
    primary = json.loads(results.read_text())
    for path, digest in primary["source_sha256"].items():
        assert sha256(ROOT/path) == digest, path
    data, _, _, _ = load_identified_selector_dataset(ROOT/"experiments/2026-10-20-offline-predictor/data")
    result = {"schema": "edmir.execution-cost-revision-review.v1", "results_sha256": sha256(results),
              "sensitivity": sensitivity(data), "component_diagnosis": component_diagnosis(primary["primary"]),
              "deletion_stability": deletion_stability(data, primary["primary"]),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                  Path(__file__), ROOT/"EDM-IR/edmir/compiler/execution_cost_selection.py",
                  ROOT/"experiments/2026-10-20-selector-decision-audit/robustness_review.py")},
              "elapsed_seconds": time.monotonic()-start, "server_ready": False}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"sensitivity": [{"shape": r["shape"], "choice": r["choice"], "minimum_harmful_perturbation": r["minimum_harmful_perturbation"]} for r in result["sensitivity"]],
                      "component_diagnosis": result["component_diagnosis"]["metrics"],
                      "stable_after_deletion": sum(r["all_deletions_keep_choice"] for r in result["deletion_stability"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.out)
