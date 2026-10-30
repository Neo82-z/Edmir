#!/usr/bin/env python3
"""Fixed offline comparison on previously examined data; not new validation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import (  # noqa: E402
    P4_ENV, P8_ENV, load_selector_dataset, sha256,
)
from edmir.compiler.relative_selection import (  # noqa: E402
    RelativeCostSelector, ScheduleCandidate, strict_surfaces,
)
from edmir.compiler.two_stage_selection import (  # noqa: E402
    SelectionGates, TwoStageSelector, _summarize_decisions,
)

REPO = Path(__file__).resolve().parents[2]
METHODS = ("host", "train_best_fixed", "original", "interaction_depth_grid", "relative")


def fit_method(method, rows, env):
    if method == "host":
        return "host"
    if method == "train_best_fixed":
        surfaces = list(strict_surfaces(rows).values())
        common = set.intersection(*(set(surface) for surface in surfaces))
        # Choose once on training workloads only; no scoring labels are read.
        return min(common, key=lambda family: (
            sum(math.log(s[family].measured_ms / s["host"].measured_ms)
                for s in surfaces) / len(surfaces), family))
    if method == "relative":
        return RelativeCostSelector.fit(rows, environment_id=env, ridge=1.0)
    return TwoStageSelector.fit(rows, ablation=(
        "full" if method == "original" else method), ridge=1.0)


def score(model, rows, environment_id):
    decisions = []
    for key, candidates in sorted(strict_surfaces(rows).items()):
        if isinstance(model, RelativeCostSelector):
            if environment_id == model.training_environment_id and any(
                row.shape in model.training_shapes for row in candidates.values()
            ):
                raise ValueError("evaluation overlaps training workloads")
            if any(row.shape in model.calibration_shapes for row in candidates.values()
                   if environment_id == model.calibration_environment_id):
                raise ValueError("evaluation overlaps calibration workloads")
            # Remove timing labels before calling the new inference API.
            selected, prediction = model.select({
                name: ScheduleCandidate.from_measurement(row, environment_id)
                for name, row in candidates.items()
            }, allow_environment_transfer=(environment_id != model.training_environment_id))
            chosen = candidates[selected.family]
        elif isinstance(model, str):
            chosen, prediction = candidates[model], {}
        else:
            chosen, prediction = model.select(candidates)
        oracle = min(candidates.values(), key=lambda row: (row.measured_ms, row.family))
        decisions.append({
            "group_id": key[0], "world_size": key[1], "rows_per_peer": key[2],
            "k": key[3], "n": key[4], "candidate_count": len(candidates),
            "oracle_family": oracle.family, "selected_family": chosen.family,
            "oracle_direction": "host" if oracle.is_host else "fused",
            "selected_direction": "host" if chosen.is_host else "fused",
            "oracle_ms": oracle.measured_ms, "selected_ms": chosen.measured_ms,
            "host_ms": candidates["host"].measured_ms,
            "regret": chosen.measured_ms / oracle.measured_ms - 1.0,
            **prediction,
        })
    return decisions


def summarize(decisions):
    result = {"metrics": _summarize_decisions(decisions, SelectionGates()),
              "decisions": decisions}
    for count in sorted({d["candidate_count"] for d in decisions}):
        subset = [d for d in decisions if d["candidate_count"] == count]
        result[f"observed_{count}_candidate_subset"] = _summarize_decisions(subset, SelectionGates())
    return result


def evaluate(directory: Path):
    p4, calibration, evaluation, manifest = load_selector_dataset(directory)
    reports = {}
    for method in METHODS:
        decisions, folds = [], []
        for group in sorted({r.group_id for r in p4}):
            train = tuple(r for r in p4 if r.group_id != group)
            test = tuple(r for r in p4 if r.group_id == group)
            model = fit_method(method, train, P4_ENV)
            decisions.extend(score(model, test, P4_ENV))
            folds.append({"excluded_group": group, "model": (
                model if isinstance(model, str) else model.to_dict())})
        transfer = fit_method(method, p4, P4_ENV)
        reports[method] = {
            "p4_group_heldout": {**summarize(decisions), "fold_models": folds},
            "p4_to_p8_no_calibration": {
                **summarize(score(transfer, evaluation, P8_ENV)),
                "model": transfer if isinstance(transfer, str) else transfer.to_dict(),
            },
        }
        if method == "relative":
            offset = transfer.calibrate_offset(calibration, environment_id=P8_ENV)
            reports[method]["p8_four_shape_offset"] = {
                **summarize(score(offset, evaluation, P8_ENV)), "model": offset.to_dict()}
        else:
            refit = fit_method(method, calibration, P8_ENV)
            reports[method]["p8_four_shape_refit"] = {
                **summarize(score(refit, evaluation, P8_ENV)),
                "model": refit if isinstance(refit, str) else refit.to_dict(),
            }
    # Normalization and static-feature typing must not change either old model.
    old_path = REPO / "experiments/2026-10-19-selector-baseline-design/interaction_results.json"
    old = json.loads(old_path.read_text())
    for method, old_method in (("original", "full"), ("interaction_depth_grid", "interaction_depth_grid")):
        for protocol, current in reports[method].items():
            reference = old["reports"][old_method][protocol]["metrics"]
            for key, value in reference.items():
                actual = current["metrics"][key]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"historical metric changed: {method}/{protocol}/{key}")
                elif actual != value:
                    raise ValueError(f"historical metric changed: {key}")
    source_paths = [Path(__file__), Path(__file__).with_name("prepare_selector_dataset.py")]
    source_paths += [REPO / "EDM-IR/edmir/compiler" / name for name in (
        "relative_selection.py", "measurement_dataset.py", "two_stage_selection.py")]
    return {
        "schema": "edmir.relative-selector-development.v1",
        "status": "retrospective development; P8 outcomes previously examined",
        "eligible_for_compiler_admission": False,
        "configuration": {
            "ridge": 1.0, "sm_count": 108, "hyperparameter_search": False,
            "minimum_fused_advantage": 0.0,
            "relative_loss": "within-workload squared log-ratio error; total pair weight 1/workload",
            "calibration_comparison": "relative: one offset; other predictors: full four-workload refit",
        },
        "dataset_manifest_sha256": sha256(directory / "manifest.json"),
        "historical_report_sha256": sha256(old_path),
        "counts": manifest["counts"],
        "sources": [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in source_paths],
        "limitations": [
            "No new measurements, public full-operator labels, or independent test set.",
            "Candidate oracle only covers observed schedules; missing candidates are not imputed.",
            "P4 complete 11-candidate surfaces all have M=64.",
            "Legacy per-source CUTLASS host only; merged and pipelined baselines remain unmeasured.",
            "The numeric ridge value is fixed but loss scaling differs between predictors.",
            "Offset calibration cannot repair a wrong fused schedule ranking.",
            "Host regressions compare point estimates, without confidence intervals.",
            "The relative model predicts ratios, not absolute milliseconds.",
        ],
        "reports": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; use a new file")
    report = evaluate(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for method, protocols in report["reports"].items():
        print(method, " ".join(f"{name}={100 * p['metrics']['mean_selection_regret']:.3f}%"
                               for name, p in protocols.items()))
