#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_tlc import resolve_java, run_case, sha256


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "spec" / "tla"
MODEL = SPEC_DIR / "RemotePublication.tla"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded EDMIR remote-publication TLC checks"
    )
    parser.add_argument(
        "--jar",
        type=Path,
        default=Path(os.environ.get("TLA2TOOLS_JAR", "")),
        help="path to the official tla2tools.jar (or set TLA2TOOLS_JAR)",
    )
    parser.add_argument("--java", help="Java executable (or set JAVA_BIN)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SPEC_DIR / "publication_results",
    )
    args = parser.parse_args()
    if not str(args.jar) or not args.jar.is_file():
        raise SystemExit("tla2tools.jar not found; pass --jar or set TLA2TOOLS_JAR")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    java = resolve_java(args.java)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("safe", "RemotePublication.cfg", None),
        ("expanded", "RemotePublicationExpanded.cfg", None),
        (
            "unsafe_acquire",
            "RemotePublicationUnsafeAcquire.cfg",
            "AcquireAfterPublish",
        ),
        (
            "unsafe_consume",
            "RemotePublicationUnsafeConsume.cfg",
            "ConsumeAfterAcquire",
        ),
        (
            "unsafe_retire",
            "RemotePublicationUnsafeRetire.cfg",
            "CompleteConsumption",
        ),
        (
            "unsafe_overwrite",
            "RemotePublicationUnsafeOverwrite.cfg",
            "NoOverwriteBeforeConsume",
        ),
    )
    reports = [
        run_case(
            java=java,
            jar=args.jar,
            model=MODEL,
            config=SPEC_DIR / config_name,
            expected_violation=violation,
            raw_output=args.out_dir / f"{name}.txt",
            workers=args.workers,
        )
        for name, config_name, violation in cases
    ]

    version_result = subprocess.run(
        [java, "-cp", str(args.jar), "tlc2.TLC", "-help"],
        capture_output=True,
        text=True,
    )
    version_match = re.search(r"Version ([^\n]+)", version_result.stdout)
    java_result = subprocess.run(
        [java, "-version"], capture_output=True, text=True
    )
    summary = {
        "schema": "edmir.publication_tlc_report.v1",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL.name,
        "model_sha256": sha256(MODEL),
        "tla2tools_sha256": sha256(args.jar),
        "tlc_version": version_match.group(1).strip() if version_match else "unknown",
        "java": (java_result.stderr or java_result.stdout).splitlines()[0],
        "workers": args.workers,
        "all_expectations_met": all(case["expectation_met"] for case in reports),
        "cases": reports,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
