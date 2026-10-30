#!/usr/bin/env python3
"""Summarize the completed strict review; cannot authorize a server or compiler."""
import argparse
import json
from pathlib import Path
import re

import run_audit as audit
from edmir.compiler.measurement_dataset import sha256


def read(path):
    return json.loads(path.read_text())


def assert_preserved(old, new):
    """All previous observations must match exactly; new certificate fields may be added."""
    if isinstance(old, dict):
        for key, value in old.items():
            assert_preserved(value, new[key])
    elif isinstance(old, list):
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert_preserved(a, b)
    else:
        assert old == new, (old, new)


def main(directory, out):
    directory, out = directory.resolve(), out.resolve()
    if out.exists():
        raise ValueError("choose a fresh output")
    primary = read(directory/"results.json")
    robustness = read(directory/"robustness_results.json")
    structure = read(directory/"structural_results.json")
    evidence = read(directory/"evidence_verification.json")
    progress = read(directory/"progress.json")
    assert progress["state"] == "completed" and progress["gpu_operations"] == 0
    assert all(r["exit_code"] == 0 for r in progress["stages"])
    assert robustness["primary_result_sha256"] == sha256(directory/"results.json")
    for path, expected in primary["source_sha256"].items():
        assert sha256(audit.ROOT/path) == expected, path
    for name, expected in robustness["source_sha256"].items():
        assert sha256(audit.HERE/name) == expected, name
    # Reproduction compares numerical evidence, not clock time or schema notes.
    previous = read(audit.HERE/"results.json")
    keys = ("primary", "stress", "local_gates", "feature_collisions", "component_replacement")
    for key in keys:
        assert primary[key] == previous[key], key
    previous_robustness = read(audit.HERE/"robustness_results.json")
    for key in ("priority_label_robustness", "training_group_deletion", "alternate_timing_objectives"):
        assert_preserved(previous_robustness[key], robustness[key])
    tests = {}
    for stage in ("audit_tests", "selection_tests", "splitk_tests", "reliability_tests"):
        text = (directory/f"{stage}.stderr.txt").read_text()
        count = re.search(r"Ran (\d+) tests? in", text)
        assert count and text.rstrip().endswith("OK"), stage
        tests[stage] = int(count.group(1))
    candidate = "interactions_joint"
    bad_priority = [
        {"shape": row["shape"], "choice": row["choices"][candidate],
         "regret": row["times_ms"][row["choices"][candidate]]/min(row["times_ms"].values())-1}
        for row in primary["primary"]["decisions"]
        if tuple(row["shape"]) in audit.PRIORITY
        and row["times_ms"][row["choices"][candidate]]/min(row["times_ms"].values())-1 > .05
    ]
    assert bad_priority, "Expected a documented unresolved priority error; re-review any changed outcome."
    witnesses = [r["minimum_harmful_perturbation"] for r in robustness["priority_label_robustness"]
                 if r["original_regret"] < .05]
    sensitive = min(witnesses, key=lambda r: r["epsilon_log"])
    result = {
        "schema": "edmir.selector-strict-review-decision.v1",
        "date": "2026-10-21",
        "audit_round_complete": True,
        "model_judgment_resolved": False,
        "server_ready": False,
        "compiler_admission": False,
        "decision_scope": "Review conclusion only; no GPU launcher was invoked or modified.",
        "original_primary_gate_passed": primary["local_gates"][candidate]["passed"],
        "original_gate_limitation": "Priority check only rejected false host choices; it did not reject false fused choices. Original thresholds and metrics are preserved.",
        "candidate": candidate,
        "unresolved_evidence": {
            "priority_regressions_over_five_percent": bad_priority,
            "leave_k_out_metrics": primary["stress"]["metrics"][candidate],
            "most_sensitive_correct_priority_choice": sensitive,
            "group_feature_collision": structure["collisions"]["group_component"][0],
            "fresh_independent_performance_workloads": 0,
            "historical_target": "legacy host wall timer; per-source CUTLASS host",
            "targets_without_matched_labels_in_this_bundle": [
                "prepared execution", "merged host", "compiler per-source cuBLAS fallback"],
        },
        "verification": {
            "source_and_result_hashes_checked": True, "numerical_reproduction_exact": True,
            "tests_passed": tests, "total_tests_passed": sum(tests.values()),
            "historical_csv_rows_verified": evidence["verified_original_csv_rows"],
            "frozen_package_sha256": evidence["frozen_package"]["package_sha256"],
            "kernel_template_build_passed": False,
        },
        "next_revision_requirements": [
            "Separate K and q in grouped-N cost representation; preserve the paired same-q denominator.",
            "Represent the actual tile/thread geometry and serial split-K epilogue dependencies; distinguish source-derived counts from unknown service times.",
            "Require symmetric checks for false host and false fused decisions; retain mean, worst-case and fixed-policy controls.",
            "Define a bounded support/uncertainty policy before collecting new labels; refusing a decision does not prove host is fastest.",
            "Keep all changes local and experimental until a revised decision passes the agreed local checks.",
        ],
        "evidence_directory": str(directory),
        "evidence_sha256": {
            str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
        },
        "audit_source_sha256": {p.name: sha256(p) for p in sorted(audit.HERE.glob("*.py"))},
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("audit_round_complete", "model_judgment_resolved",
                                            "server_ready", "compiler_admission", "verification")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.out)
