#!/usr/bin/env python3
"""Training-only coupling audit and retrospective comparison; no GPU required."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.factorized_selection import FactorizedSelector, features
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import RelativeCostSelector, ScheduleCandidate, strict_surfaces
from edmir.compiler.two_stage_selection import _feature_names, _feature_values

REPO = Path(__file__).resolve().parents[2]
METHODS = ("host", "train_best_fixed", "relative", "relative_supported", "factorized_supported")


def rank(matrix, tolerance=1e-9):
    """Rank of standardized design, independent of the ridge regularizer."""
    a = [list(row) for row in matrix]
    pivot_row = 0
    for col in range(len(a[0])):
        if pivot_row == len(a):
            break
        pivot = max(range(pivot_row, len(a)), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= tolerance:
            continue
        a[pivot], a[pivot_row] = a[pivot_row], a[pivot]
        divisor = a[pivot_row][col]
        a[pivot_row] = [v/divisor for v in a[pivot_row]]
        for r in range(pivot_row+1, len(a)):
            multiplier = a[r][col]
            a[r] = [v-multiplier*p for v, p in zip(a[r], a[pivot_row])]
        pivot_row += 1
    return pivot_row


def design_audit(names, values, *, intercept=True):
    columns = list(zip(*values))
    means = [sum(c)/len(c) for c in columns]
    scales = [math.sqrt(sum((v-m)**2 for v in c)/len(c)) for c, m in zip(columns, means)]
    z = [tuple((v-m)/s if s > 1e-12 else 0.0 for v, m, s in zip(row, means, scales))
         for row in values]
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            corr = sum(row[i]*row[j] for row in z)/len(z)
            if abs(corr) >= 0.95:
                pairs.append({"left": names[i], "right": names[j], "correlation": corr})
    # An anchored, no-intercept model uses uncentered columns for rank.
    design = [(1.0, *row) for row in z] if intercept else [
        tuple(v/max(s, 1e-12) for v, s in zip(row, scales)) for row in values]
    return {"rows": len(values), "columns_including_intercept": len(design[0]),
            "rank": rank(design), "rank_tolerance": 1e-9,
            "constant_features": [n for n, s in zip(names, scales) if s <= 1e-12],
            "absolute_correlation_at_least_0_95": pairs}


def fit_models(dataset, calibration=None):
    factorized = FactorizedSelector.fit(dataset)
    relative = RelativeCostSelector.fit(dataset.rows, environment_id=dataset.environment_id)
    surfaces = list(strict_surfaces(dataset.rows).values())
    common = set.intersection(*(set(s) for s in surfaces))
    fixed = min(common, key=lambda n: (
        sum(math.log(s[n].measured_ms)-math.log(s["host"].measured_ms) for s in surfaces), n))
    if calibration is not None:
        calibration.validate()
        if calibration.baseline != dataset.baseline:
            raise ValueError("baseline differs across fit/calibration")
        factorized = factorized.calibrate_offset(calibration)
        relative = relative.calibrate_offset(calibration.rows, environment_id=calibration.environment_id)
    return factorized, relative, fixed


def predict_methods(models, candidates, baseline):
    """This boundary sees no timing labels. All methods use the same host."""
    factorized, relative, fixed = models
    f, details = factorized.select(candidates, baseline=baseline, allow_environment_transfer=True)
    allowed = {n: c for n, c in candidates.items() if details["predictions"][n]["eligible"]}
    r, _ = relative.select(candidates, allow_environment_transfer=True)
    supported, _ = relative.select(allowed, allow_environment_transfer=True)
    if fixed not in candidates:
        raise ValueError("frozen fixed family is missing")
    return {
        "host": "host", "train_best_fixed": fixed, "relative": r.family,
        "relative_supported": supported.family, "factorized_supported": f.family,
    }, details


def decisions(models, dataset):
    dataset.validate()
    factorized = models[0]
    for env, shapes in ((factorized.training_environment_id, factorized.training_shapes),
                        (factorized.calibration_environment_id, factorized.calibration_shapes)):
        if dataset.environment_id == env and any(r.shape in shapes for r in dataset.rows):
            raise ValueError("evaluation overlaps training/calibration workloads")
    result = []
    for key, surface in sorted(strict_surfaces(dataset.rows).items()):
        choices, details = predict_methods(models, {
            n: ScheduleCandidate.from_measurement(r, dataset.environment_id) for n, r in surface.items()
        }, dataset.baseline)
        all_oracle = min(surface, key=lambda n: (surface[n].measured_ms, n))
        eligible = [n for n in surface if details["predictions"][n]["eligible"]]
        eligible_oracle = min(eligible, key=lambda n: (surface[n].measured_ms, n))
        result.append({
            "group_id": key[0], "shape": list(key[1:]), "choices": choices,
            "times_ms": {n: r.measured_ms for n, r in surface.items()},
            "observed_oracle": all_oracle, "eligible_oracle": eligible_oracle,
            "eligibility_oracle_regret": surface[eligible_oracle].measured_ms / surface[all_oracle].measured_ms - 1,
            **details,
        })
    return result


def summarize(rows):
    result = {"workloads": len(rows), "grouped_available": sum(d["grouped_available"] for d in rows),
              "grouped_eligible": sum(d["grouped_eligible"] for d in rows),
              "mean_eligibility_oracle_regret": sum(d["eligibility_oracle_regret"] for d in rows)/len(rows),
              "methods": {}}
    for method in METHODS:
        regrets, within, regressions = [], [], []
        for d in rows:
            t = d["times_ms"]
            chosen = t[d["choices"][method]]
            regrets.append(chosen/t[d["observed_oracle"]]-1)
            within.append(chosen/t[d["eligible_oracle"]]-1)
            regressions.append(chosen/t["host"]-1)
        entry = {
            "mean_regret": sum(regrets)/len(regrets), "max_regret": max(regrets),
            "host_regressions": sum(r > 1e-12 for r in regressions),
            "max_host_regression": max(0.0, max(regressions)),
        }
        if method.endswith("_supported"):
            entry["mean_regret_within_eligible"] = sum(within)/len(within)
        result["methods"][method] = entry
    return result


def grouped_observations(datasets):
    """Paired historical evidence, separately by environment; not training."""
    pairs, controls = [], defaultdict(list)
    for dataset in datasets:
        for surface in strict_surfaces(dataset.rows).values():
            for row in surface.values():
                if row.n_tiles_per_cta != 2:
                    continue
                base = surface[row.family.removesuffix("_groupedn2")]
                item = {"environment_id": dataset.environment_id, "shape": list(row.shape),
                        "q": row.split_k_slices, "grouped_over_plain": row.measured_ms/base.measured_ms}
                pairs.append(item)
                controls[(dataset.environment_id, row.k, row.n, row.split_k_slices)].append(item)
    summaries = []
    for env, m in sorted({(r["environment_id"], r["shape"][1]) for r in pairs}):
        subset = [r for r in pairs if (r["environment_id"], r["shape"][1]) == (env, m)]
        ratios = [r["grouped_over_plain"] for r in subset]
        summaries.append({"environment_id": env, "M": m, "pairs": len(subset),
                          "workloads": len({tuple(r["shape"]) for r in subset}),
                          "grouped_faster_pairs": sum(r < 1 for r in ratios),
                          "geomean_grouped_over_plain": math.exp(sum(map(math.log, ratios))/len(ratios)),
                          "min_ratio": min(ratios), "max_ratio": max(ratios)})
    matched = [items for _, items in sorted(controls.items()) if len(items) > 1]
    return {"status": "historical paired point estimates; no timing uncertainty",
            "by_environment_and_M": summaries,
            "matched_P_K_N_q_across_M": matched,
            "matched_sets_with_sign_reversal": sum(
                min(r["grouped_over_plain"] for r in items) < 1 < max(r["grouped_over_plain"] for r in items)
                for items in matched)}


def evaluate(directory):
    p4, calibration, p8, manifest = load_identified_selector_dataset(directory)
    fitted = fit_models(p4)
    calibrated = fit_models(p4, calibration)
    folds = []
    for group_id in sorted({r.group_id for r in p4.rows}):
        train = replace(p4, rows=tuple(r for r in p4.rows if r.group_id != group_id))
        test = replace(p4, rows=tuple(r for r in p4.rows if r.group_id == group_id))
        folds.extend(decisions(fit_models(train), test))
    protocols = {"p4_leave_group_out": folds,
                 "p4_to_p8_no_calibration": decisions(fitted, p8),
                 "p8_calibrated": decisions(calibrated, p8)}
    factorized = fitted[0]
    names = _feature_names("fused", "interaction_depth_grid")
    raw = [r for r in p4.rows if not r.is_host]
    m64 = [r for r in raw if r.rows_per_peer == 64]
    coupling = {
        "data": "P4 training features only, no test labels",
        "old_all_M": design_audit(names, [_feature_values(r, names, sm_count=108) for r in raw]),
        "old_M64": design_audit(names, [_feature_values(r, names, sm_count=108) for r in m64]),
        "gamma_identity_max_abs_error": max(abs(
            math.log2(r.estimated_ctas_per_op) - math.log2(math.ceil(r.rows_per_peer/64))
            - math.log2(r.remote_a_replay) - math.log2(r.split_k_slices)) for r in raw),
        "split_depth_identity_max_abs_error": max(abs(
            math.log2(r.k/r.split_k_slices) - math.log2(r.k) + math.log2(r.split_k_slices)) for r in raw),
    }
    stage_rows = {
        "gate": [r for r in raw if r.family == "direct"],
        "split": [r for r in raw if r.n_tiles_per_cta == 1 and r.split_k_slices > 1],
        "group": [r for r in raw if r.n_tiles_per_cta == 2],
    }
    for stage, rows in stage_rows.items():
        fit = getattr(factorized, stage)
        coupling[f"factorized_{stage}"] = design_audit(
            fit.names, [features(r, stage) for r in rows], intercept=stage != "split")
    support = defaultdict(list)
    for d in protocols["p8_calibrated"]:
        support[d["shape"][1]].append(d)
    source_paths = [Path(__file__), REPO / "EDM-IR/edmir/compiler/factorized_selection.py",
                    REPO / "EDM-IR/edmir/compiler/host_baseline.py",
                    REPO / "EDM-IR/edmir/compiler/relative_selection.py",
                    REPO / "EDM-IR/edmir/compiler/measurement_dataset.py"]
    return {
        "schema": "edmir.factorized-development.v1",
        "status": "retrospective development, not independent validation",
        "eligible_for_compiler_admission": False, "hyperparameter_search": False,
        "calibration_difference": "relative: all-fused offset; factorized: direct-only offset; neither expands grouped support",
        "baseline": manifest["identified_host_baseline"],
        "dataset_manifest_sha256": sha256(directory / "manifest.json"),
        "sources": {str(p.relative_to(REPO)): sha256(p) for p in source_paths},
        "coupling": coupling, "model": factorized.to_dict(),
        "grouped_observations": grouped_observations((p4, calibration, p8)),
        "calibrated_model": calibrated[0].to_dict(),
        "protocols": {name: {"summary": summarize(rows), "decisions": rows}
                      for name, rows in protocols.items()},
        "p8_calibrated_by_M": {str(m): summarize(rows) for m, rows in sorted(support.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new file")
    report = evaluate(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    for name, result in report["protocols"].items():
        print(name, json.dumps(result["summary"]))
