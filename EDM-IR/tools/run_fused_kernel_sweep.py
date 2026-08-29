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
    A2AGemmProblem,
    CppManifestBackend,
    FusedScheduleContract,
    Target,
    TopologySnapshot,
    build_fused_program,
    build_sequential_program,
)


BENCH_PATTERN = re.compile(r"([a-z0-9_]+)=([^ ]+)")
FIELDS = (
    "candidate_id",
    "family",
    "rows_per_peer",
    "k",
    "n",
    "gemm_kernel",
    "remote_input_bytes",
    "n_tiles",
    "n_tiles_per_cta",
    "remote_a_replay",
    "estimated_remote_a_load_bytes",
    "split_k_slices",
    "estimated_ctas_per_op",
    "min_ms",
    "median_ms",
    "p95_ms",
    "max_ms",
    "mean_ms",
    "warmup",
    "iterations",
)


def run(command: list[str], *, log: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("$ " + " ".join(command) + "\n" + result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.stdout


def install_header(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copyfile(source, destination)


def candidate(
    problem: A2AGemmProblem,
    topology: TopologySnapshot,
    family: str,
) -> tuple[str, str]:
    grouped_n = 1
    base_family = family
    if "_groupedn" in family:
        base_family, grouped_value = family.rsplit("_groupedn", 1)
        grouped_n = int(grouped_value)
    if base_family == "host":
        program = build_sequential_program(problem)
        kernel = "sm80_64x128x32"
        communication_api = "host_nccl"
    elif base_family == "direct":
        program = build_fused_program(problem, chunk_rows=problem.rows_per_peer)
        kernel = "sm80_64x128x32"
        communication_api = "nccl_device"
    elif base_family.startswith("splitk"):
        split_k_slices = int(base_family.removeprefix("splitk"))
        if split_k_slices not in {2, 4, 8, 16}:
            raise ValueError(f"unsupported split-K slice count {split_k_slices}")
        program = build_fused_program(problem, chunk_rows=problem.rows_per_peer)
        kernel = f"sm80_64x128x32_splitk{split_k_slices}"
        communication_api = "nccl_device"
    else:
        raise ValueError(f"unknown family {family!r}")
    target = Target(
        name=f"a100-{family}",
        topology=topology,
        arch="sm_80",
        gemm_provider="cutlass",
        gemm_kernel=kernel,
        communication_api=communication_api,
        fused_schedule=FusedScheduleContract(n_tiles_per_cta=grouped_n),
    )
    artifact = CppManifestBackend().emit(program, target)
    return artifact.files["edmir_program.hpp"], kernel


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure host, direct/split-K LSA, and optional grouped-N CUTLASS boundaries"
        )
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--nccl-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-per-peer", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k-values", default="256,512,1024,2048,4096")
    parser.add_argument("--split-k-values", default="8")
    parser.add_argument(
        "--grouped-n-values",
        default="",
        help="optional executable grouped-N factors; current runtime supports only 2",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--validate-first", action="store_true")
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    edmir_root = runtime.parent
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology = TopologySnapshot.from_dict(
        json.loads(args.topology.read_text(encoding="utf-8"))
    )
    k_values = tuple(int(value) for value in args.k_values.split(","))
    if not k_values or any(k <= 0 or k % 256 for k in k_values):
        raise ValueError("all K values must be positive multiples of 256")
    split_k_values = tuple(int(value) for value in args.split_k_values.split(","))
    if not split_k_values or any(value not in {2, 4, 8, 16} for value in split_k_values):
        raise ValueError("split-K values must be selected from 2,4,8,16")
    grouped_n_values = tuple(
        int(value) for value in args.grouped_n_values.split(",") if value
    )
    if any(value != 2 for value in grouped_n_values):
        raise ValueError("the executable grouped-N sweep currently supports only 2")
    base_families = ("direct", *(f"splitk{value}" for value in split_k_values))
    families = (
        "host",
        *base_families,
        *(
            f"{family}_groupedn{grouped_n}"
            for grouped_n in grouped_n_values
            for family in base_families
        ),
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["PYTHONPATH"] = str(edmir_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cuda_bin = Path("/usr/local/cuda-12.8/bin")
    if (cuda_bin / "nvcc").is_file():
        env["CUDACXX"] = str(cuda_bin / "nvcc")
        env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    nccl_lib = args.nccl_home.resolve() / "lib"
    env["LD_LIBRARY_PATH"] = str(nccl_lib) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    artifacts = output / "artifacts"
    manifest: list[dict[str, str | int]] = []
    for k in k_values:
        problem = A2AGemmProblem(
            world_size=topology.world_size,
            rows_per_peer=args.rows_per_peer,
            k=k,
            n=args.n,
        )
        for family in families:
            header, kernel = candidate(problem, topology, family)
            candidate_id = f"m{args.rows_per_peer}_k{k}_n{args.n}_{family}"
            directory = artifacts / candidate_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "edmir_program.hpp").write_text(header, encoding="utf-8")
            manifest.append(
                {
                    "candidate_id": candidate_id,
                    "family": family,
                    "rows_per_peer": args.rows_per_peer,
                    "k": k,
                    "n": args.n,
                    "gemm_kernel": kernel,
                    "artifact": str(directory.relative_to(output)),
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current = output / "work" / "generated"
    current.mkdir(parents=True, exist_ok=True)
    install_header(
        output / str(manifest[0]["artifact"]) / "edmir_program.hpp",
        current / "edmir_program.hpp",
    )
    build = output / "work" / "build"
    logs = output / "logs"
    configure = [
        "cmake",
        "-S",
        str(runtime),
        "-B",
        str(build),
        "-DEDMIR_ENABLE_CUDA=ON",
        "-DEDMIR_ENABLE_CUTLASS=ON",
        "-DEDMIR_ENABLE_NCCL_DEVICE=ON",
        f"-DEDMIR_GENERATED_DIR={current}",
        f"-DEDMIR_CUTLASS_DIR={args.cutlass.resolve()}",
        f"-DNCCL_INCLUDE_DIR={args.nccl_home.resolve() / 'include'}",
        f"-DNCCL_LIBRARY={nccl_lib / 'libnccl.so'}",
        "-DCMAKE_CUDA_ARCHITECTURES=80",
    ]
    run(configure, log=logs / "00_configure.txt", env=env)
    run(["cmake", "--build", str(build), "-j"], log=logs / "01_build.txt", env=env)
    run(
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
        log=logs / "02_ctest.txt",
        env=env,
    )

    rows: list[dict[str, str]] = []
    for index, entry in enumerate(manifest):
        install_header(
            output / str(entry["artifact"]) / "edmir_program.hpp",
            current / "edmir_program.hpp",
        )
        run(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "edmir_benchmark_generated",
                *(
                    ["edmir_validate_generated"]
                    if args.validate_first and index < len(families)
                    else []
                ),
                "-j",
            ],
            log=logs / f"10_build_{index:02d}_{entry['candidate_id']}.txt",
            env=env,
        )
        if args.validate_first and index < len(families):
            run(
                [str(build / "edmir_validate_generated")],
                log=logs / f"11_validate_{index:02d}_{entry['candidate_id']}.txt",
                env=env,
            )
        text = run(
            [
                str(build / "edmir_benchmark_generated"),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
            ],
            log=logs / f"12_bench_{index:02d}_{entry['candidate_id']}.txt",
            env=env,
        )
        line = next(line for line in text.splitlines() if line.startswith("bench "))
        values = dict(BENCH_PATTERN.findall(line))
        if values["gemm_kernel"] != str(entry["gemm_kernel"]):
            raise RuntimeError(f"stale generated header for {entry['candidate_id']}")
        row = {
            field: str(entry[field]) if field in entry else values[field]
            for field in FIELDS
        }
        rows.append(row)
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        print(
            f"measured={entry['candidate_id']} median_ms={values['median_ms']} "
            f"ctas_per_op={values['estimated_ctas_per_op']}"
        )
    print(f"complete={output / 'results.csv'} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
