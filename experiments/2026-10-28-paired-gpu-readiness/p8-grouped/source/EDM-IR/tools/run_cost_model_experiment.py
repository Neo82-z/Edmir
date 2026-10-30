#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from edmir.compiler import (
    CalibrationProfile,
    evaluate_cost_model,
    fit_execution_overhead,
    fit_gemm_profile,
    fit_local_copy_profile,
    read_runtime_measurements,
)


BENCH_PATTERN = re.compile(r"([a-z0-9_]+)=([^ ]+)")
MEASUREMENT_FIELDS = (
    "candidate_id",
    "problem_id",
    "split",
    "world_size",
    "rows_per_peer",
    "k",
    "n",
    "dtype",
    "strategy",
    "chunk_rows",
    "buffer_count",
    "measured_us",
    "min_us",
    "p95_us",
    "iterations",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def run(command: list[str], *, log: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("$ " + " ".join(command) + "\n" + result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.stdout


def append_measurement(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=MEASUREMENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        destination.flush()
        os.fsync(destination.fileno())


def install_generated_header(source: Path, destination: Path) -> None:
    """Replace the header with a fresh mtime so incremental builds cannot reuse a prior Program."""

    destination.unlink(missing_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and checkpoint an EDM-IR GPU cost-model sweep")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prepared = args.prepared.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(prepared / "manifest.csv")
    shapes = json.loads((prepared / "gemm_shapes.json").read_text(encoding="utf-8"))
    for row in manifest:
        header = prepared / row["artifact_dir"] / "edmir_program.hpp"
        if not header.is_file():
            raise ValueError(f"missing generated header: {header}")
    print(f"preflight candidates={len(manifest)} gemm_shapes={len(shapes)}")
    if args.dry_run:
        print("dry_run=ok gpu_not_touched=true")
        return 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    edmir_root = args.runtime.resolve().parent
    env["PYTHONPATH"] = str(edmir_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    logs = output / "logs"
    run(["nvidia-smi", "-L"], log=logs / "00_nvidia_smi_l.txt", env=env)
    run(["nvidia-smi", "topo", "-m"], log=logs / "01_topology.txt", env=env)

    current = output / "work" / "generated"
    current.mkdir(parents=True, exist_ok=True)
    install_generated_header(
        prepared / manifest[0]["artifact_dir"] / "edmir_program.hpp",
        current / "edmir_program.hpp",
    )
    build = output / "work" / "runtime-build"
    run(
        [
            "cmake", "-S", str(args.runtime.resolve()), "-B", str(build),
            "-DEDMIR_ENABLE_CUDA=ON", f"-DEDMIR_GENERATED_DIR={current}",
        ],
        log=logs / "02_cmake_configure.txt",
        env=env,
    )
    run(["cmake", "--build", str(build), "-j"], log=logs / "03_cmake_build.txt", env=env)
    run(["ctest", "--test-dir", str(build), "--output-on-failure"], log=logs / "04_ctest.txt", env=env)
    run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(edmir_root / "tests"), "-p", "*_test.py"],
        log=logs / "05_python_tests.txt",
        env=env,
    )
    run([str(build / "edmir_run_generated")], log=logs / "06_gpu_zero_smoke.txt", env=env)

    gemm_log = output / "gemm_measurements.jsonl"
    gemm_log.write_text("", encoding="utf-8")
    groups: dict[tuple[int, int], list[int]] = {}
    for shape in shapes:
        groups.setdefault((int(shape["n"]), int(shape["k"])), []).append(int(shape["m"]))
    for index, ((n, k), m_values) in enumerate(sorted(groups.items())):
        text = run(
            [
                str(build / "edmir_benchmark_gemm"), "--m", ",".join(map(str, sorted(set(m_values)))),
                "--n", str(n), "--k", str(k), "--warmup", "10", "--trials", "7",
                "--iterations", "100",
            ],
            log=logs / f"10_gemm_{index:02d}_n{n}_k{k}.txt",
            env=env,
        )
        with gemm_log.open("a", encoding="utf-8") as destination:
            destination.write(text)

    copy_sizes = sorted(
        {
            int(row["chunk_rows"]) * int(row["k"]) * (2 if row["dtype"] != "float32" else 4)
            for row in manifest
        }
    )
    copy_log = output / "local_copy_measurements.jsonl"
    copy_text = run(
        [
            str(build / "edmir_benchmark_copy"),
            "--sizes", ",".join(map(str, copy_sizes)),
            "--warmup", "10", "--trials", "7", "--iterations", "100",
        ],
        log=logs / "11_local_copy.txt",
        env=env,
    )
    copy_log.write_text(copy_text, encoding="utf-8")

    base_profile = CalibrationProfile.from_dict(json.loads(args.base_profile.read_text(encoding="utf-8")))
    device_profile = fit_local_copy_profile(fit_gemm_profile(base_profile, gemm_log), copy_log)
    device_profile_path = output / "device_profile.json"
    device_profile_path.write_text(json.dumps(device_profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    measurements_path = output / "runtime_measurements.csv"
    completed = {
        row["candidate_id"] for row in read_csv(measurements_path)
    } if measurements_path.exists() else set()
    for index, row in enumerate(manifest):
        candidate = row["candidate_id"]
        if candidate in completed:
            print(f"resume skip={candidate}")
            continue
        install_generated_header(
            prepared / row["artifact_dir"] / "edmir_program.hpp",
            current / "edmir_program.hpp",
        )
        run(
            ["cmake", "--build", str(build), "--target", "edmir_benchmark_generated", "-j"],
            log=logs / f"20_build_{index:03d}_{candidate}.txt",
            env=env,
        )
        text = run(
            [
                str(build / "edmir_benchmark_generated"),
                "--warmup", str(args.warmup), "--iterations", str(args.iterations),
            ],
            log=logs / f"21_bench_{index:03d}_{candidate}.txt",
            env=env,
        )
        bench_line = next(line for line in text.splitlines() if line.startswith("bench "))
        values = dict(BENCH_PATTERN.findall(bench_line))
        expected = {
            "strategy": row["strategy"],
            "ranks": row["world_size"],
            "rows_per_peer": row["rows_per_peer"],
            "k": row["k"],
            "n": row["n"],
            "chunk_rows": row["chunk_rows"],
            "buffer_count": row["buffer_count"],
            "op_dispatch": row["op_dispatch_count"],
            "event_record": row["event_record_count"],
            "event_wait": row["event_wait_count"],
            "buffer_reuse": row["buffer_reuse_count"],
        }
        mismatches = {
            key: (wanted, values.get(key))
            for key, wanted in expected.items()
            if str(wanted) != values.get(key)
        }
        if mismatches:
            raise RuntimeError(f"built Program does not match manifest for {candidate}: {mismatches}")
        append_measurement(
            measurements_path,
            {
                **{field: row[field] for field in MEASUREMENT_FIELDS if field in row},
                "measured_us": 1000.0 * float(values["median_ms"]),
                "min_us": 1000.0 * float(values["min_ms"]),
                "p95_us": 1000.0 * float(values["p95_ms"]),
                "iterations": values["iterations"],
            },
        )
        print(f"measured={candidate} median_us={1000.0 * float(values['median_ms']):.3f}")

    measurements = read_runtime_measurements(measurements_path)
    calibrated = fit_execution_overhead(device_profile, measurements)
    report = evaluate_cost_model(calibrated, measurements)
    (output / "calibrated_profile.json").write_text(
        json.dumps(calibrated.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"train_mape={100.0 * report['train']['mape']:.2f}%")
    print(f"heldout_mape={100.0 * report['heldout']['mape']:.2f}%")
    print(f"heldout_direction_accuracy={100.0 * report['heldout']['direction_accuracy']:.2f}%")
    print(f"selection_min_gain_ratio={100.0 * report['selection_min_gain_ratio']:.2f}%")
    print(f"heldout_policy_direction_accuracy={100.0 * report['heldout']['policy_direction_accuracy']:.2f}%")
    print(f"heldout_selection_regret={100.0 * report['heldout']['mean_selection_regret']:.2f}%")
    print(f"complete={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
