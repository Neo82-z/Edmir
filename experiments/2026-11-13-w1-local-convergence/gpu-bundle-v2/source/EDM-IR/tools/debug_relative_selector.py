#!/usr/bin/env python3
"""Replay archived decisions and explain the largest P8 error, without GPUs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import P4_ENV, P8_ENV, load_selector_dataset, sha256
from edmir.compiler.relative_selection import RelativeCostSelector, ScheduleCandidate
from edmir.compiler.two_stage_selection import AnalyticalScheduleFeatures
from evaluate_relative_selector import REPO, evaluate


def debug(dataset: Path, reference_path: Path):
    current = evaluate(dataset)
    reference = json.loads(reference_path.read_text())
    decisions_checked = 0
    for method, protocols in current["reports"].items():
        for protocol, report in protocols.items():
            old = reference["reports"][method][protocol]
            for name, value in old["metrics"].items():
                actual = report["metrics"][name]
                equal = (math.isclose(value, actual, abs_tol=1e-10, rel_tol=1e-10)
                         if isinstance(value, (int, float)) and not isinstance(value, bool)
                         else value == actual)
                if not equal:
                    raise ValueError(f"metric changed: {method}/{protocol}/{name}")
            def identities(rows):
                return {(r["world_size"], r["rows_per_peer"], r["k"], r["n"]):
                        (r["selected_family"], r["oracle_family"]) for r in rows}
            if identities(report["decisions"]) != identities(old["decisions"]):
                raise ValueError(f"candidate decisions changed: {method}/{protocol}")
            decisions_checked += len(report["decisions"])

    p4, calibration, scoring, _ = load_selector_dataset(dataset)
    base = RelativeCostSelector.fit(p4, environment_id=P4_ENV)
    model = base.calibrate_offset(calibration, environment_id=P8_ENV)
    decisions = current["reports"]["relative"]["p8_four_shape_offset"]["decisions"]
    worst = max(decisions, key=lambda r: r["regret"])
    shape = tuple(worst[name] for name in ("world_size", "rows_per_peer", "k", "n"))
    rows = [r for r in scoring if r.shape == shape]
    host_ms = next(r.measured_ms for r in rows if r.is_host)
    trace = []
    for row in rows:
        candidate = ScheduleCandidate.from_measurement(row, P8_ENV)
        explanation = model.explain(candidate)
        support = [r for r in p4 if r.family == row.family]
        trace.append({
            "family": row.family,
            "measured_ms": row.measured_ms,
            "measured_host_ratio": row.measured_ms / host_ms,
            "predicted_host_ratio": math.exp(explanation["predicted_log_ratio"]),
            "uncalibrated_log_ratio": base.predict_log_ratio(candidate, allow_environment_transfer=True),
            "analytical_features": asdict(AnalyticalScheduleFeatures.from_measurement(candidate)),
            "training_support": {
                "family_m_values": sorted({r.rows_per_peer for r in support}),
                "same_m_family_workloads": len({r.shape for r in support
                                               if r.rows_per_peer == row.rows_per_peer}),
                "same_mkn_family_workloads": len({r.shape for r in support
                    if (r.rows_per_peer, r.k, r.n) == (row.rows_per_peer, row.k, row.n)}),
            },
            "explanation": explanation,
        })
    trace.sort(key=lambda r: (r["predicted_host_ratio"], r["family"]))
    chosen = next(r for r in trace if r["family"] == worst["selected_family"])
    oracle = next(r for r in trace if r["family"] == worst["oracle_family"])
    delta = [{"feature": a["feature"], "oracle_minus_selected": b["contribution"] - a["contribution"]}
             for a, b in zip(chosen["explanation"]["terms"], oracle["explanation"]["terms"])]
    delta.sort(key=lambda r: abs(r["oracle_minus_selected"]), reverse=True)
    difference = oracle["explanation"]["predicted_log_ratio"] - chosen["explanation"]["predicted_log_ratio"]
    if not math.isclose(sum(r["oracle_minus_selected"] for r in delta), difference, abs_tol=1e-12):
        raise ValueError("feature trace does not reconstruct the predicted ranking")
    sources = current["sources"] + [
        {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": sha256(Path(__file__))},
        *({"path": str(p.relative_to(REPO)), "sha256": sha256(p)}
          for p in (REPO / "EDM-IR/edmir/compiler" / name for name in (
              "fused_selection.py", "fused_evaluation.py", "roofline_accounting.py"))),
    ]
    summary = {
        "schema": "edmir.predictor-debug.v1",
        "eligible_for_compiler_admission": False,
        "evidence_status": "existing data, retrospective diagnosis only",
        "reference_report_sha256": sha256(reference_path),
        "replay_checks": {"all_metrics_unchanged": True, "all_decisions_unchanged": True,
                          "method_protocol_combinations": 15, "decision_instances": decisions_checked},
        "worst_point": worst, "candidate_trace": trace,
        "oracle_vs_selected": {
            "predicted_ratio": oracle["predicted_host_ratio"] / chosen["predicted_host_ratio"],
            "measured_ratio": oracle["measured_ms"] / chosen["measured_ms"],
            "feature_contribution_differences": delta,
            "interpretation": "linear-score decomposition, not a causal attribution",
        },
        "sources": sources,
        "limits": [
            "Input/model validation does not provide a latency-error bound.",
            "The relative model remains an offline experiment, not the compile-fused K-rule path.",
            "Environment strings are declared metadata, not a runtime hardware attestation.",
            "No new training labels or independent generalization evidence were added.",
        ],
    }
    return current, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory exists; use a new path")
    current, summary = debug(args.dataset, args.reference)
    args.out_dir.mkdir(parents=True)
    for name, result in (("replayed_results.json", current), ("debug_summary.json", summary)):
        (args.out_dir / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["replay_checks"]))
    print(json.dumps(summary["oracle_vs_selected"], indent=2))
