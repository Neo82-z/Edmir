#!/usr/bin/env python3
"""Reproduce the finite audit locally with process-group timeouts and fresh logs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import run_audit as audit
from run_selector_gpu import command, write_json


def main(out):
    # No run_selector_gpu build/collect entry point is called. Only its bounded
    # subprocess helper is reused for CPU-only scripts and tests.
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(audit.ROOT)
    jobs = [
        ("audit", [sys.executable, audit.HERE/"run_audit.py", "--out", out/"results.json"], 60),
        ("robustness", [sys.executable, audit.HERE/"robustness_review.py",
                        out/"robustness_results.json", "--primary", out/"results.json"], 180),
        ("structure", [sys.executable, audit.HERE/"structural_review.py",
                       out/"structural_results.json"], 60),
        ("evidence", [sys.executable, audit.HERE/"verify_evidence.py",
                      out/"evidence_verification.json"], 60),
        ("audit_tests", [sys.executable, "-m", "unittest", "discover",
                        "-s", audit.HERE, "-p", "test_*.py", "-v"], 60),
    ]
    for name, pattern in (
        ("selection_tests", "*selection_test.py"),
        ("splitk_tests", "splitk_revision_test.py"),
        ("reliability_tests", "predictor_reliability_test.py"),
    ):
        jobs.append((name, [sys.executable, "-m", "unittest", "discover",
                           "-s", audit.ROOT/"EDM-IR/tests", "-p", pattern, "-v"], 60))
    results = []
    start = time.monotonic()
    for name, argv, limit in jobs:
        write_json(out/"progress.json", {"state": "running", "stage": name, "timeout_seconds": limit})
        stage_start = time.monotonic()
        try:
            with (out/f"{name}.stdout.txt").open("w") as stdout, (out/f"{name}.stderr.txt").open("w") as stderr:
                result = command(argv, timeout=limit, stdout=stdout, stderr=stderr)
        except BaseException as error:
            write_json(out/"progress.json", {"state": "failed", "stage": name, "error": str(error)})
            raise
        results.append({"stage": name, "exit_code": result.returncode,
                        "elapsed_seconds": time.monotonic()-stage_start, "timeout_seconds": limit})
    write_json(out/"progress.json", {"state": "completed", "elapsed_seconds": time.monotonic()-start,
                                     "stages": results, "gpu_operations": 0, "server_ready": False})
    print(json.dumps(json.loads((out/"progress.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new directory; never overwrite audit evidence.")
    main(parser.parse_args().out)
