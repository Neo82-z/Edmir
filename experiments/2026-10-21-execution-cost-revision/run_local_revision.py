#!/usr/bin/env python3
"""Bounded local replay of the declared revision. Never build or use a GPU."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
from run_selector_gpu import command, write_json
from prepare_selector_gpu import source_manifest
from edmir.compiler.measurement_dataset import sha256


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    jobs = [
        ("evaluate", [sys.executable, ROOT/"EDM-IR/tools/evaluate_execution_cost_selector.py",
                      "--design", HERE/"design.json", "--out", out/"results.json"], 120),
        ("review", [sys.executable, HERE/"review_revision.py", "--results", out/"results.json",
                    "--out", out/"review_results.json"], 180),
        ("evidence", [sys.executable, ROOT/"experiments/2026-10-20-selector-decision-audit/verify_evidence.py",
                      out/"evidence.json"], 60),
    ]
    for name, directory, pattern in (
        ("selection_tests", ROOT/"EDM-IR/tests", "*selection_test.py"),
        ("splitk_tests", ROOT/"EDM-IR/tests", "splitk_revision_test.py"),
        ("reliability_tests", ROOT/"EDM-IR/tests", "predictor_reliability_test.py"),
        ("review_tests", HERE, "test_*.py"),
    ):
        jobs.append((name, [sys.executable, "-m", "unittest", "discover", "-s", directory,
                           "-p", pattern, "-v"], 60))
    stages, start = [], time.monotonic()
    for name, argv, limit in jobs:
        write_json(out/"progress.json", {"state": "running", "stage": name, "timeout_seconds": limit})
        before = time.monotonic()
        try:
            with (out/f"{name}.stdout.txt").open("w") as stdout, (out/f"{name}.stderr.txt").open("w") as stderr:
                command(argv, timeout=limit, stdout=stdout, stderr=stderr)
        except BaseException as error:
            write_json(out/"progress.json", {"state": "failed", "stage": name, "reason": str(error)})
            raise
        stages.append({"stage": name, "exit_code": 0, "elapsed_seconds": time.monotonic()-before,
                       "timeout_seconds": limit})
    result = json.loads((out/"results.json").read_text())
    review = json.loads((out/"review_results.json").read_text())
    assert review["results_sha256"] == sha256(out/"results.json")
    for artifact in (result, review):
        for path, digest in artifact["source_sha256"].items():
            assert sha256(ROOT/path) == digest, path
    # Compare only numbers/decisions, excluding wall time and paths.
    initial = json.loads((HERE/"results.json").read_text())
    for key in ("primary", "stress", "gates"):
        assert initial[key] == result[key], key
    initial_review = json.loads((HERE/"review_results.json").read_text())
    for key in ("sensitivity", "component_diagnosis", "deletion_stability"):
        assert initial_review[key] == review[key], key
    tests = {}
    for name, _, _ in jobs:
        if name.endswith("_tests"):
            log = (out/f"{name}.stderr.txt").read_text()
            match = re.search(r"Ran (\d+) tests? in", log)
            assert match and log.rstrip().endswith("OK"), name
            tests[name] = int(match.group(1))
    frozen_path = ROOT/"experiments/2026-10-20-local-replay-runtime-checks/gpu_pilot/package.json"
    frozen = json.loads(frozen_path.read_text())
    payload = {
        "schema": "edmir.execution-cost-revision-decision.v1", "date": "2026-10-21",
        "implementation_complete": True, "model_judgment_resolved": False,
        "proposed_model": result["proposed_model"], "local_gate": result["gates"][result["proposed_model"]],
        "server_ready": False, "compiler_admission": False,
        "reason": "The proposed fixed revision failed bidirectional priority and leave-K checks. Do not promote it or retune after these results.",
        "tests": tests, "total_tests": sum(tests.values()), "numerical_replay_exact": True,
        "gpu_operations": 0, "frozen_package_sha256": frozen["package_sha256"],
        "new_source_files_not_in_old_package": sorted(set(source_manifest())-set(frozen["package"]["sources"])),
        "design_sha256": sha256(HERE/"design.json"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT/"EDM-IR/edmir/compiler/execution_cost_selection.py",
            ROOT/"EDM-IR/tools/evaluate_execution_cost_selector.py",
            ROOT/"EDM-IR/tests/execution_cost_selection_test.py", HERE/"review_revision.py",
            HERE/"test_revision.py", Path(__file__))},
        "evidence_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "progress.json"},
    }
    write_json(out/"decision.json", payload)
    write_json(out/"progress.json", {"state": "completed", "elapsed_seconds": time.monotonic()-start,
                                     "stages": stages, "gpu_operations": 0})
    print(json.dumps({"decision": payload["local_gate"], "total_tests": payload["total_tests"],
                      "numerical_replay_exact": True, "new_source_files_not_in_old_package": payload["new_source_files_not_in_old_package"],
                      "elapsed_seconds": time.monotonic()-start}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
