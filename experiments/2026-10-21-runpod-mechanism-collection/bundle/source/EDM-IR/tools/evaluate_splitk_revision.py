#!/usr/bin/env python3
"""Four predeclared split-K controls. P8 is diagnostic, never variant selection."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate, strict_surfaces
from edmir.compiler.splitk_revision import SplitKRevision
from evaluate_factorized_selector import decisions, fit_models, summarize

REPO = Path(__file__).resolve().parents[2]
VARIANTS = (("original", "reference"), ("original", "pairwise"),
            ("waves", "reference"), ("waves", "pairwise"))


def q_decisions(model, train, test):
    """Only static candidate objects cross the scoring boundary."""
    for env, shapes in (
        (model.original.training_environment_id, model.original.training_shapes),
        (model.original.calibration_environment_id, model.original.calibration_shapes),
    ):
        if test.environment_id == env and any(r.shape in shapes for r in test.rows):
            raise ValueError("split-K evaluation overlaps fit/calibration")
    train_surfaces = list(train.validate().values())
    common = set.intersection(*({n for n, r in s.items()
                                if not r.is_host and r.n_tiles_per_cta == 1}
                               for s in train_surfaces))
    fixed = min(common, key=lambda n: (
        sum(math.log(s[n].measured_ms/s["direct"].measured_ms) for s in train_surfaces), n))
    result = []
    for key, surface in sorted(test.validate().items()):
        rows = {n: r for n, r in surface.items() if not r.is_host and r.n_tiles_per_cta == 1}
        candidates = {n: ScheduleCandidate.from_measurement(r, test.environment_id) for n, r in rows.items()}
        for c in candidates.values():
            model.effective()._check_candidate(c, test.baseline, True)
        scores = {n: model.split.predict(c) for n, c in candidates.items()}
        if fixed not in rows:
            raise ValueError("training fixed q is absent from evaluation")
        times = {n: r.measured_ms for n, r in rows.items()}
        oracle = min(times, key=lambda n: (times[n], n))
        result.append({
            "group_id": key[0], "shape": list(key[1:]), "times_ms": times,
            "scores_log_over_direct": scores, "oracle": oracle,
            "predicted_ranking": sorted(scores, key=lambda n: (scores[n], n)),
            "observed_ranking": sorted(times, key=lambda n: (times[n], n)),
            "choices": {"split_revision": min(scores, key=lambda n: (scores[n], n)),
                        "train_best_fixed_q": fixed},
        })
    return result


def q_summary(rows):
    methods = {}
    for method in ("split_revision", "train_best_fixed_q"):
        regrets = [r["times_ms"][r["choices"][method]]/r["times_ms"][r["oracle"]]-1 for r in rows]
        methods[method] = {"mean_regret": sum(regrets)/len(rows), "max_regret": max(regrets)}
    return {"workloads": len(rows), "methods": methods}


def gate(q, full):
    candidate, fixed = q["methods"]["split_revision"], q["methods"]["train_best_fixed_q"]
    revised, relative = full["methods"]["factorized_supported"], full["methods"]["relative_supported"]
    checks = {
        "q_mean_le_fixed": candidate["mean_regret"] <= fixed["mean_regret"]+1e-12,
        "q_worst_le_fixed": candidate["max_regret"] <= fixed["max_regret"]+1e-12,
        "full_mean_le_relative": revised["mean_regret"] <= relative["mean_regret"]+1e-12,
        "full_worst_le_relative": revised["max_regret"] <= relative["max_regret"]+1e-12,
        "host_regressions_le_relative": revised["host_regressions"] <= relative["host_regressions"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate(directory, design_path):
    design = json.loads(design_path.read_text())
    if tuple((v["basis"], v["loss"]) for v in design["variants"]) != VARIANTS:
        raise ValueError("design does not match the four implemented controls")
    if design["ridge"] != 1.0 or design["hyperparameter_search"] is not False:
        raise ValueError("this controlled evaluation fixes ridge=1")
    p4, calibration, p8, _ = load_identified_selector_dataset(directory)
    result = {"schema": "edmir.splitk-controlled-results.v1",
              "status": "retrospective development; no independent labels",
              "eligible_for_compiler_admission": False,
              "design_sha256": sha256(design_path),
              "dataset_manifest_sha256": sha256(directory/"manifest.json"),
              "variants": {}, "decision_basis": "P4 leave-group-out only"}
    selection = []
    for basis, loss in VARIANTS:
        q_rows, full_rows = [], []
        max_control_error = 0.0
        for group in sorted({r.group_id for r in p4.rows}):
            train = replace(p4, rows=tuple(r for r in p4.rows if r.group_id != group))
            test = replace(p4, rows=tuple(r for r in p4.rows if r.group_id == group))
            original, relative, fixed = fit_models(train)
            revised = SplitKRevision.fit(train, basis=basis, loss=loss)
            if basis == "original" and loss == "reference":
                max_control_error = max(max_control_error, *(abs(a-b) for a, b in zip(
                    original.split.coefficients, revised.split.coefficients)))
            q_rows.extend(q_decisions(revised, train, test))
            full_rows.extend(decisions((revised.effective(), relative, fixed), test))
        q, full = q_summary(q_rows), summarize(full_rows)
        entry = {
            "p4_leave_group_out": {"q_only": {"summary": q, "decisions": q_rows},
                                  "full_surface": {"summary": full, "decisions": full_rows}},
            "p4_gate": gate(q, full),
        }
        if basis == "original" and loss == "reference":
            entry["control_max_coefficient_error"] = max_control_error
            if max_control_error > 1e-10:
                raise ValueError("original/reference failed to reproduce the old split fit")
        name = f"{basis}_{loss}"
        result["variants"][name] = entry
        if entry["p4_gate"]["passed"]:
            selection.append((full["methods"]["factorized_supported"]["mean_regret"], name))
    # Freeze the P4 decision before computing any P8 evaluation.
    result["p4_selected_variant"] = min(selection)[1] if selection else None
    result["local_decision"] = ("retain for independent validation only" if selection else
                                "do not promote any split-K revision")
    for basis, loss in VARIANTS:
        entry = result["variants"][f"{basis}_{loss}"]
        fitted = SplitKRevision.fit(p4, basis=basis, loss=loss)
        calibrated = fitted.calibrate_offset(calibration)
        entry["model"] = fitted.to_dict()
        entry["calibrated_model"] = calibrated.to_dict()
        for protocol, model, cal in (
            ("p8_no_calibration_diagnostic", fitted, None),
            ("p8_calibrated_diagnostic", calibrated, calibration),
        ):
            _, relative, fixed = fit_models(p4, cal)
            qs = q_decisions(model, p4, p8)
            fs = decisions((model.effective(), relative, fixed), p8)
            entry[protocol] = {"q_only": {"summary": q_summary(qs), "decisions": qs},
                               "full_surface": {"summary": summarize(fs), "decisions": fs}}
    source_paths = [Path(__file__), REPO/"EDM-IR/edmir/compiler/splitk_revision.py",
                    REPO/"EDM-IR/tools/evaluate_factorized_selector.py"]
    source_paths += [REPO/f"EDM-IR/edmir/compiler/{name}.py" for name in
                     ("factorized_selection", "relative_selection", "two_stage_selection",
                      "host_baseline", "measurement_dataset")]
    result["sources"] = {str(p.relative_to(REPO)): sha256(p) for p in source_paths}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    output = evaluate(args.dataset, args.design)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({name: {"gate": item["p4_gate"], "p4": item["p4_leave_group_out"]["full_surface"]["summary"],
                            "p8": item["p8_calibrated_diagnostic"]["full_surface"]["summary"]}
                      for name, item in output["variants"].items()}, indent=2))
