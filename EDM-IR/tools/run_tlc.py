#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "spec" / "tla"
MODEL = SPEC_DIR / "SlotLifecycle.tla"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_java(explicit: str | None) -> str:
    candidates = (
        explicit,
        os.environ.get("JAVA_BIN"),
        shutil.which("java"),
        "/opt/homebrew/opt/openjdk@17/bin/java",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return candidate
    raise SystemExit("Java 11+ not found; set JAVA_BIN to a working Java executable")


def extract_counts(output: str) -> tuple[int | None, int | None]:
    match = re.search(
        r"([0-9,]+) states generated, ([0-9,]+) distinct states found",
        output,
    )
    if match is None:
        return None, None
    return tuple(int(value.replace(",", "")) for value in match.groups())


def run_case(
    *,
    java: str,
    jar: Path,
    model: Path = MODEL,
    config: Path,
    expected_violation: str | None,
    raw_output: Path,
    workers: int,
) -> dict[str, object]:
    command = [
        java,
        "-XX:+UseParallelGC",
        "-cp",
        str(jar),
        "tlc2.TLC",
        "-cleanup",
        "-workers",
        str(workers),
        "-config",
        str(config),
        str(model),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    raw_output.write_text(output)
    generated, distinct = extract_counts(output)

    if expected_violation is None:
        passed = (
            result.returncode == 0
            and "Model checking completed. No error has been found." in output
        )
    else:
        passed = (
            result.returncode != 0
            and expected_violation in output
            and "is violated" in output
        )
    return {
        "config": config.name,
        "config_sha256": sha256(config),
        "expected": "safe" if expected_violation is None else "counterexample",
        "expected_violation": expected_violation,
        "expectation_met": passed,
        "returncode": result.returncode,
        "states_generated": generated,
        "distinct_states": distinct,
        "raw_output": raw_output.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded EDMIR slot-lifecycle TLC checks"
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
        default=SPEC_DIR / "results",
    )
    args = parser.parse_args()
    if not str(args.jar) or not args.jar.is_file():
        raise SystemExit("tla2tools.jar not found; pass --jar or set TLA2TOOLS_JAR")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    java = resolve_java(args.java)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("safe", "SlotLifecycle.cfg", None),
        ("expanded", "SlotLifecycleExpanded.cfg", None),
        ("unsafe_reuse", "SlotLifecycleUnsafeReuse.cfg", "NoPrematureReuse"),
        ("unsafe_read", "SlotLifecycleUnsafeRead.cfg", "ReadAfterPublish"),
        ("unsafe_release", "SlotLifecycleUnsafeRelease.cfg", "CompleteReduction"),
    )
    reports = []
    for name, config_name, violation in cases:
        reports.append(
            run_case(
                java=java,
                jar=args.jar,
                config=SPEC_DIR / config_name,
                expected_violation=violation,
                raw_output=args.out_dir / f"{name}.txt",
                workers=args.workers,
            )
        )

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
        "schema": "edmir.tlc_report.v1",
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
