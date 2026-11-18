#!/usr/bin/env python3
"""Explore fixed feature alternatives on archived A100 data; no GPU execution.

The P8 outcomes have already informed model design. All reports, including
the original four/19 split replay, are retrospective development evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler.two_stage_selection import (  # noqa: E402
    SelectionGates,
    TwoStageSelector,
    _summarize_decisions,
    evaluate_group_heldout,
    read_schedule_measurements,
)

REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_rho", "interaction_depth", "interaction_depth_grid")


def surfaces(rows):
    result = defaultdict(dict)
    owners = {}
    for row in rows:
        if not math.isfinite(row.measured_ms):
            raise ValueError("non-finite measured latency")
        owner = owners.setdefault(row.shape, row.group_id)
        if owner != row.group_id:
            raise ValueError(f"same shape appears in distinct groups: {row.shape}")
        if row.family in result[row.shape]:
            raise ValueError(f"duplicate candidate: {row.shape}, {row.family}")
        result[row.shape][row.family] = row
    return result


def score(selector, rows):
    decisions = []
    for shape, candidates in sorted(surfaces(rows).items()):
        selected, prediction = selector.select(candidates)
        oracle = min(candidates.values(), key=lambda row: (row.measured_ms, row.family))
        decisions.append({
            "world_size": shape[0], "rows_per_peer": shape[1],
            "k": shape[2], "n": shape[3],
            "oracle_family": oracle.family, "selected_family": selected.family,
            "oracle_direction": "host" if oracle.is_host else "fused",
            "selected_direction": "host" if selected.is_host else "fused",
            "oracle_ms": oracle.measured_ms, "selected_ms": selected.measured_ms,
            "host_ms": candidates["host"].measured_ms,
            "regret": selected.measured_ms / oracle.measured_ms - 1.0,
            **prediction,
        })
    return {"metrics": _summarize_decisions(decisions, SelectionGates()),
            "decisions": decisions, "model": selector.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-report", type=Path, default=REPO /
                        "experiments/2026-08-29-a100-grouped-n-plan/two_stage_selector_report.json")
    parser.add_argument("--p8-root", type=Path, default=REPO /
                        "experiments/2026-08-30-a100-8gpu/extracted")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; use a new path")
    inputs = []

    def record(path):
        path = path.resolve()
        inputs.append({"path": str(path.relative_to(REPO)) if path.is_relative_to(REPO)
                       else str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return path

    p4_archive = json.loads(record(args.p4_report).read_text())
    p4_paths = [record(REPO / name) for name in p4_archive["inputs"]["results"]]
    p4 = read_schedule_measurements(p4_paths, world_size=4)
    if len(p4) != p4_archive["inputs"]["measurement_rows"]:
        raise ValueError("P4 archive row count mismatch")
    p4_surfaces = surfaces(p4)
    grouping_counts = Counter()
    for candidates in p4_surfaces.values():
        for family, grouped in candidates.items():
            if not family.endswith("_groupedn2"):
                continue
            base = candidates[family.removesuffix("_groupedn2")]
            halved = (base.remote_a_replay == 2 * grouped.remote_a_replay and
                      base.estimated_ctas_per_op == 2 * grouped.estimated_ctas_per_op)
            cohort = "counts_halved" if halved else "counts_not_halved"
            grouping_counts[cohort + "_pairs"] += 1
            grouping_counts[cohort + "_grouped_wins"] += (
                grouped.measured_ms < base.measured_ms)

    manifest = json.loads(record(args.p8_root /
        "experiments/2026-08-29-a100-8gpu-transfer-plan/manifest.json").read_text())
    expected_families = set(manifest["candidate_space"]["base"]) | {
        family + "_groupedn2" for family in manifest["candidate_space"]["grouped_n_2"]
    }
    p8, roles, expected_shapes = [], {}, set()
    for sweep in manifest["sweeps"]:
        sweep_id = sweep["id"]
        path = args.p8_root / "results/p8-timing" / sweep_id / "repeat-01/results.csv"
        if not path.exists():
            path = args.p8_root / "results/p8-frozen" / sweep_id / "repeat-01/results.csv"
        rows = read_schedule_measurements([record(path)], world_size=8)
        p8.extend(replace(row, group_id=sweep_id) for row in rows)
        roles[sweep_id] = sweep["role"]
        expected_shapes.update((8, sweep["rows_per_peer"], k, sweep["n"])
                               for k in sweep["k_values"])
    p8_surfaces = surfaces(p8)
    if set(p8_surfaces) != expected_shapes or any(
        set(candidates) != expected_families for candidates in p8_surfaces.values()
    ):
        raise ValueError("P8 candidate coverage does not match frozen manifest")

    # The repeat medians only change scoring, never the four-shape training data.
    repeats = {}
    repeat_path = record(args.p8_root /
        "results/p8-analysis/boundary-repeats/replication_results.csv")
    with repeat_path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            shape = (8, int(raw["rows_per_peer"]), int(raw["k"]), int(raw["n"]))
            row = p8_surfaces[shape][raw["family"]]
            if row.group_id != raw["sweep_id"] or not math.isclose(
                row.measured_ms, float(raw["initial_ms"]), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise ValueError("repeat does not match initial candidate")
            key = (shape, row.family)
            if key in repeats:
                raise ValueError("duplicate repeat candidate")
            repeats[key] = float(raw["median_ms"])
    calibration = tuple(row for row in p8 if roles[row.group_id] == "calibration")
    evaluation = tuple(replace(row, measured_ms=repeats.get(
        (row.shape, row.family), row.measured_ms)) for row in p8
        if roles[row.group_id] == "heldout")

    reports = {}
    complete_shapes = {shape for shape, candidates in p4_surfaces.items()
                       if set(candidates) == expected_families}
    for variant in VARIANTS:
        p4_report = evaluate_group_heldout(p4, ablation=variant, ridge=1.0)
        p4_report.pop("fold_models")
        complete_decisions = [row for row in p4_report["decisions"] if (
            row["world_size"], row["rows_per_peer"], row["k"], row["n"]
        ) in complete_shapes]
        p4_report["complete_11_candidate_subset_metrics"] = _summarize_decisions(
            complete_decisions, SelectionGates())
        reports[variant] = {
            "p4_group_heldout": p4_report,
            "p4_to_p8_no_calibration": score(
                TwoStageSelector.fit(p4, ablation=variant, ridge=1.0), evaluation),
            "p8_four_shape_refit": score(
                TwoStageSelector.fit(calibration, ablation=variant, ridge=1.0), evaluation),
        }
        print(variant + " " + " ".join(
            f"{protocol}={100 * item['metrics']['mean_selection_regret']:.3f}%"
            for protocol, item in reports[variant].items()))

    # Confirm the unchanged model reproduces the historical P4 mean regret.
    if not math.isclose(reports["full"]["p4_group_heldout"]["metrics"]["mean_selection_regret"],
                        p4_archive["reports"]["full"]["metrics"]["mean_selection_regret"],
                        rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("original P4 model no longer reproduces archived regret")
    source_files = [Path(__file__), Path(__file__).resolve().parents[1] /
                    "edmir/compiler/two_stage_selection.py"]
    sources = [{"path": str(path.relative_to(REPO)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_files]
    report = {
        "schema": "edmir.selector-interaction-development.v1",
        "status": "retrospective development only; not a fresh held-out evaluation",
        "eligible_for_compiler_admission": False,
        "configuration": {"variants": VARIANTS, "ridge": 1.0, "sm_count": 108,
                          "hyperparameter_search": False},
        "coverage": {
            "p4_rows": len(p4), "p4_shapes": len(p4_surfaces),
            "p4_candidate_count_histogram": dict(Counter(map(len, p4_surfaces.values()))),
            "p4_complete_shapes": len(complete_shapes),
            "p4_complete_subset_m_values": sorted({shape[1] for shape in complete_shapes}),
            "p4_grouping_counts": dict(grouping_counts),
            "p8_rows": len(p8), "p8_shapes": len(p8_surfaces),
            "p8_calibration_shapes": len(surfaces(calibration)),
            "p8_scored_shapes": len(surfaces(evaluation)),
            "p8_repeat_candidates": len(repeats),
        },
        "limits": [
            "Old P8 outcomes informed these variants; no new generalization claim.",
            "P4 scores use observed candidate sets; missing grouped candidates are not imputed.",
            "All P4 shapes with 11 candidates have M=64; grouping coverage lacks M diversity.",
            "All comparisons retain the old per-source CUTLASS host; new baselines need new timings.",
            "host_regressions uses selected_ms > host_ms; admission errors are reported separately.",
            "Point estimates have no paired confidence intervals or tuning amortization.",
        ],
        "inputs": inputs, "sources": sources, "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
