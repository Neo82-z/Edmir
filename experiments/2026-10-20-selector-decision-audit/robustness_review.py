#!/usr/bin/env python3
"""Strict post-fit diagnostics. No hyperparameter changes or GPU operations."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
import scipy

import run_audit as audit
from candidate_model import DecisionModel, joint_vector
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256


def linear_response(train, model):
    """Independently reconstruct coefficients as R @ log(training latency).

    Feature scaling and eligibility depend on geometry only. Therefore this
    response is exact for the fixed ridge design, not a first-order derivative.
    """
    rows = list(train.rows)
    index = {(r.shape, r.family): i for i, r in enumerate(rows)}
    width = len(model.fit.coefficients)
    a, b = np.zeros((width, width)), np.zeros((width, len(rows)))
    xs = []

    def z(row):
        if row.is_host:
            return np.zeros(width)
        return (np.asarray(joint_vector(row, True))-model.fit.means)/model.fit.scales

    for surface in train.validate().values():
        pairs = list(combinations(surface.values(), 2))
        weight = 1/len(pairs)
        for left, right in pairs:
            x = z(left)-z(right)
            a += weight*np.outer(x, x)
            b[:, index[(left.shape, left.family)]] += weight*x
            b[:, index[(right.shape, right.family)]] -= weight*x
            xs.append(x*math.sqrt(weight))
    penalty = np.ones(width)
    penalty[0] = 0
    a += np.diag(penalty)
    response = np.linalg.solve(a, b)
    labels = np.log([r.measured_ms for r in rows])
    np.testing.assert_allclose(response@labels, model.fit.coefficients, atol=1e-10, rtol=1e-10)
    # A common multiplier of all measured times cannot affect any ratio.
    np.testing.assert_allclose(response@np.ones(len(rows)), 0, atol=1e-10)
    x = np.asarray(xs)
    singular = np.linalg.svd(x, compute_uv=False)
    return rows, response, z, {
        "design_rank": int(np.linalg.matrix_rank(x)), "columns": width,
        "singular_values": singular.tolist(),
        "design_condition_number": float(singular[0]/singular[-1]),
        "regularized_hessian_condition_number": float(np.linalg.cond(a)),
        "coefficient_error_against_svd": float(np.max(np.abs(response@labels-model.fit.coefficients))),
    }


def force_choice(scores, influences, target, *, margin=1e-7):
    """Minimize max per-label log perturbation making target the unique winner.

    Variables are delta[training observations], epsilon. Constraints enforce
    score_target(delta) + margin <= every other candidate score, and
    -epsilon <= delta_i <= epsilon. This is worst-case sensitivity, not an
    estimate of noise probability or a statistically calibrated confidence set.
    """
    count = len(influences[target])
    inequalities, rhs = [], []
    for name in scores:
        if name == target:
            continue
        inequalities.append(np.r_[influences[target]-influences[name], 0.])
        rhs.append(scores[name]-scores[target]-margin)
    inequalities.extend(np.c_[np.eye(count), -np.ones(count)])
    inequalities.extend(np.c_[-np.eye(count), -np.ones(count)])
    rhs.extend([0.]*(2*count))
    matrix, rhs = np.asarray(inequalities), np.asarray(rhs)
    objective = np.r_[np.zeros(count), 1.]
    result = linprog(objective, A_ub=matrix, b_ub=rhs,
                     bounds=[(None, None)]*count+[(0., None)], method="highs",
                     options={"time_limit": 10., "maxiter": 10000,
                              "primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    if not result.success:
        return {"success": False, "status": int(result.status), "message": result.message}, None
    primal_error = max(0., float(np.max(matrix@result.x-rhs)), -float(result.x[-1]))
    dual_error = float(np.max(np.abs(objective-matrix.T@result.ineqlin.marginals-result.lower.marginals)))
    # <= row multipliers must be nonpositive; only epsilon has a finite
    # lower bound. Free delta variables cannot have bound multipliers.
    dual_sign_error = max(
        0., float(np.max(result.ineqlin.marginals)),
        -float(result.lower.marginals[-1]),
        float(np.max(np.abs(result.lower.marginals[:-1]))),
        float(np.max(np.abs(result.upper.marginals))),
    )
    complementarity = max(
        float(np.max(np.abs(result.ineqlin.marginals*(rhs-matrix@result.x)))),
        abs(float(result.lower.marginals[-1]*result.x[-1])),
    )
    dual_value = float(rhs@result.ineqlin.marginals)
    gap = abs(float(result.fun)-dual_value)
    if max(primal_error, dual_error, dual_sign_error, complementarity, gap) > 1e-7:
        raise AssertionError("linear-programming certificate failed")
    return {
        "success": True, "epsilon_log": float(result.fun),
        "max_upward_relative_label_change": math.expm1(float(result.fun)),
        "min_multiplicative_factor": math.exp(-float(result.fun)),
        "primal_violation": primal_error, "dual_residual": dual_error, "duality_gap": gap,
        "dual_sign_or_free_bound_violation": dual_sign_error,
        "complementarity_max_abs": complementarity,
    }, result.x[:-1]


def deletion_stability(data, primary):
    expected = {tuple(r["shape"]): r["choices"]["interactions_joint"] for r in primary["decisions"]}
    records = {shape: {"shape": list(shape), "original_choice": choice, "deletions": []}
               for shape, choice in expected.items()}
    groups = sorted({r.group_id for r in data.rows})
    for heldout in groups:
        test = replace(data, rows=tuple(r for r in data.rows if r.group_id == heldout))
        for deleted in groups:
            if deleted == heldout:
                continue
            train = replace(data, rows=tuple(r for r in data.rows if r.group_id not in (heldout, deleted)))
            model = DecisionModel(train, joint=True, interactions=True)
            for surface in test.validate().values():
                row = next(iter(surface.values()))
                candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id)
                              for n, r in surface.items()}
                chosen, scores = model.choose(candidates)
                records[row.shape]["deletions"].append({
                    "removed_training_group": deleted, "choice": chosen,
                    "regret": surface[chosen].measured_ms/min(r.measured_ms for r in surface.values())-1,
                    "scores": scores,
                })
    for record in records.values():
        record["choice_counts"] = dict(Counter(r["choice"] for r in record["deletions"]))
        record["all_deletions_keep_choice"] = all(r["choice"] == record["original_choice"]
                                                  for r in record["deletions"])
        record["worst_deletion_regret"] = max(r["regret"] for r in record["deletions"])
    return list(records.values())


def label_robustness(data):
    records = []
    for surface in data.validate().values():
        row = next(iter(surface.values()))
        if row.shape not in audit.PRIORITY:
            continue
        train = replace(data, rows=tuple(r for r in data.rows if r.group_id != row.group_id))
        model = DecisionModel(train, joint=True, interactions=True)
        candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in surface.items()}
        chosen, scores = model.choose(candidates)
        training_rows, response, z, numerical = linear_response(train, model)
        influences = {n: z(candidates[n])@response for n in scores}
        labels = np.log([r.measured_ms for r in training_rows])
        for name in scores:
            if not math.isclose(float(influences[name]@labels), scores[name], abs_tol=1e-10, rel_tol=1e-10):
                raise AssertionError("influence map changed a held-out score")
        best_ms = min(r.measured_ms for r in surface.values())
        attacks = []
        for target in scores:
            regret = surface[target].measured_ms/best_ms-1
            if regret <= .05:
                continue
            certificate, delta = force_choice(scores, influences, target)
            entry = {"target": target, "observed_regret": regret, **certificate}
            if delta is not None:
                # Actual refit, not just re-evaluation of the linearized map.
                altered = replace(train, rows=tuple(
                    replace(r, measured_ms=r.measured_ms*math.exp(float(d)))
                    for r, d in zip(training_rows, delta)))
                refit = DecisionModel(altered, joint=True, interactions=True)
                actual, actual_scores = refit.choose(candidates)
                if actual != target:
                    raise AssertionError(f"LP witness did not reproduce: {target} != {actual}")
                expected = {n: scores[n]+float(influences[n]@delta) for n in scores}
                error = max(abs(actual_scores[n]-expected[n]) for n in scores)
                if error > 1e-9:
                    raise AssertionError("exact response disagrees with independent refit")
                entry["refit_confirmed_choice"] = actual
                entry["refit_score_max_error"] = error
                entry["witness_log_changes"] = delta.tolist()
            attacks.append(entry)
        successful = [a for a in attacks if a["success"]]
        most_sensitive = min(successful, key=lambda a: a["epsilon_log"]) if successful else None
        records.append({
            "shape": list(row.shape), "original_choice": chosen,
            "original_regret": surface[chosen].measured_ms/best_ms-1,
            "training_catalog": [{"shape": list(r.shape), "family": r.family} for r in training_rows],
            "numerical": numerical, "targets": attacks,
            "minimum_harmful_perturbation": ({k: v for k, v in most_sensitive.items()
                                             if k != "witness_log_changes"} if most_sensitive else None),
        })
    return records


def alternate_statistics(primary, dataset_path):
    source = {}
    for line in (dataset_path/"selector_measurements.jsonl").read_text().splitlines():
        record = json.loads(line)
        r = record["measurement"]
        key = (r["world_size"], r["rows_per_peer"], r["k"], r["n"])
        if r["world_size"] == 4:
            source[(key, r["family"])] = record["reported_timing"]
    results = {}
    for statistic in ("median_ms", "mean_ms", "p95_ms"):
        rows = [{**row, "times_ms": {n: source[(tuple(row["shape"]), n)][statistic]
                                    for n in row["times_ms"]}}
                for row in primary["decisions"]]
        results[statistic] = {n: audit.metrics(rows, n) for n in primary["metrics"]}
    return {"note": "Choices are frozen. Mean and p95 change the scoring objective; these are not confidence intervals.",
            "metrics": results}


def main(out, primary_path=None):
    if out.exists():
        raise ValueError("choose a fresh output")
    start = time.monotonic()
    data_path = audit.ROOT/"experiments/2026-10-20-offline-predictor/data"
    data, _, _, _ = load_identified_selector_dataset(data_path)
    primary_path = primary_path or audit.HERE/"results.json"
    original = json.loads(primary_path.read_text())
    result = {
        "schema": "edmir.selector-strict-robustness.v1",
        "status": "post-fit diagnosis; no new hyperparameters or model promotion",
        "primary_result_sha256": sha256(primary_path),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "priority_label_robustness": label_robustness(data),
        "training_group_deletion": deletion_stability(data, original["primary"]),
        "alternate_timing_objectives": alternate_statistics(original["primary"], data_path),
        "source_sha256": {p.name: sha256(p) for p in (
            Path(__file__), audit.HERE/"candidate_model.py", audit.HERE/"run_audit.py")},
        "elapsed_seconds": time.monotonic()-start,
        "server_ready": False,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({
        "elapsed_seconds": result["elapsed_seconds"],
        "stable_shapes_after_training_group_deletion": sum(
            r["all_deletions_keep_choice"] for r in result["training_group_deletion"]),
        "total_shapes": len(result["training_group_deletion"]),
        "priority": [{"shape": r["shape"], "original_choice": r["original_choice"],
                      "original_regret": r["original_regret"],
                      "minimum_harmful_perturbation": r["minimum_harmful_perturbation"]}
                     for r in result["priority_label_robustness"]],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--primary", type=Path)
    args = parser.parse_args()
    main(args.out, args.primary)
