from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

from edmir.compiler import CppManifestBackend, Program, Target, TopologySnapshot


BENCHMARK = re.compile(
    r"median_ms=(?P<median>[0-9.]+) p95_ms=(?P<p95>[0-9.]+).*?"
    r"max_ms=(?P<max>[0-9.]+)"
)
FIELDS = (
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
    "p95_us",
    "max_us",
    "iterations",
)


def run_logged(command: list[str], log: Path) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    log.write_text(output, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
    return output


def measurement(manifest: dict[str, str], output: str, iterations: int) -> dict[str, object]:
    match = BENCHMARK.search(output)
    if match is None:
        raise ValueError(f"benchmark output for {manifest['candidate_id']} is not parseable")
    return {
        **{field: manifest[field] for field in FIELDS[:11]},
        "measured_us": 1000.0 * float(match["median"]),
        "p95_us": 1000.0 * float(match["p95"]),
        "max_us": 1000.0 * float(match["max"]),
        "iterations": iterations,
    }


def write_measurements(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = args.out / "artifacts"
    logs = args.out / "logs"
    build = args.out / "work" / "build"
    artifacts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)

    topology = TopologySnapshot.from_dict(
        json.loads(args.topology.read_text(encoding="utf-8"))
    )
    target = Target(
        "a100-sxm4",
        topology,
        arch="sm_80",
        gemm_provider="cutlass",
        gemm_kernel="auto",
    )
    with (args.prepared / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    if args.limit is not None:
        manifest = manifest[: args.limit]

    rows: list[dict[str, object]] = []
    measurements_path = args.out / "runtime_measurements.csv"
    for index, item in enumerate(manifest):
        candidate = item["candidate_id"]
        artifact = artifacts / candidate
        artifact.mkdir(exist_ok=True)
        source = args.prepared / item["artifact_dir"] / "program.json"
        program = Program.from_dict(json.loads(source.read_text(encoding="utf-8")))
        CppManifestBackend().emit(program, target).write_to(artifact)

        configure = [
            "cmake",
            "-S",
            str(args.code / "runtime"),
            "-B",
            str(build),
            "-DEDMIR_ENABLE_CUDA=ON",
            "-DEDMIR_ENABLE_CUTLASS=ON",
            f"-DEDMIR_CUTLASS_DIR={args.cutlass}",
            "-DEDMIR_BUILD_TESTS=ON",
            f"-DEDMIR_GENERATED_DIR={artifact}",
        ]
        run_logged(configure, logs / f"20_configure_{index:03d}_{candidate}.txt")
        run_logged(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "edmir_benchmark_generated",
                "-j4",
            ],
            logs / f"21_build_{index:03d}_{candidate}.txt",
        )
        output = run_logged(
            [
                str(build / "edmir_benchmark_generated"),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ],
            logs / f"22_bench_{index:03d}_{candidate}.txt",
        )
        rows.append(measurement(item, output, args.iterations))
        write_measurements(measurements_path, rows)
        print(
            f"[{index + 1}/{len(manifest)}] {candidate} {rows[-1]['measured_us']:.3f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()
