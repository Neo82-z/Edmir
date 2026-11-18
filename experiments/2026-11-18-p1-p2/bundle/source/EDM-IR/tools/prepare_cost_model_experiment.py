#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from edmir.compiler import (
    A2AGemmProblem,
    CalibrationProfile,
    CppManifestBackend,
    Target,
    TopologySnapshot,
    build_pipelined_program,
    build_sequential_program,
    simulate_program,
)


MANIFEST_FIELDS = (
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
    "artifact_dir",
    "device_prediction_us",
    "op_dispatch_count",
    "event_record_count",
    "event_wait_count",
    "buffer_reuse_count",
)


def read_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def candidate_name(problem_id: str, strategy: str, chunk: int, buffers: int) -> str:
    if strategy == "sequential":
        return f"{problem_id}__sequential"
    return f"{problem_id}__pipelined_c{chunk}_b{buffers}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a reproducible EDM-IR cost-model sweep")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="sm_80")
    args = parser.parse_args()

    design = read_object(args.design)
    if design.get("schema") != "edmir.cost-model-experiment.v1":
        raise ValueError("unsupported experiment design schema")
    profile = CalibrationProfile.from_dict(read_object(args.profile))
    topology = TopologySnapshot.from_dict(read_object(args.topology))
    if not topology.has_full_p2p():
        raise ValueError("cost-model sweep requires full P2P connectivity")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_root = output / "artifacts"
    target = Target(name="cuda", kind="cuda", arch=args.arch, topology=topology)
    backend = CppManifestBackend()
    rows: list[dict] = []
    gemm_shapes: set[tuple[int, int, int]] = set()

    for spec in design["problems"]:
        problem = A2AGemmProblem(
            world_size=int(spec["world_size"]),
            rows_per_peer=int(spec["rows_per_peer"]),
            k=int(spec["k"]),
            n=int(spec["n"]),
            dtype=str(spec.get("dtype", "float16")),
            name=str(spec["id"]),
        )
        problem.validate()
        if problem.world_size != topology.world_size:
            raise ValueError(f"problem {problem.name} world size does not match topology")
        chunks = tuple(int(value) for value in spec["chunk_rows"])
        buffers = tuple(int(value) for value in design["buffer_counts"])
        programs = [build_sequential_program(problem)]
        programs.extend(
            build_pipelined_program(problem, chunk_rows=chunk, buffer_count=count)
            for chunk in chunks
            for count in buffers
        )
        for program in programs:
            name = candidate_name(
                problem.name,
                program.strategy,
                program.chunk_rows,
                program.buffer_count,
            )
            artifact_dir = artifact_root / name
            backend.emit(program, target).write_to(artifact_dir)
            simulation = simulate_program(program, profile)
            rows.append(
                {
                    "candidate_id": name,
                    "problem_id": problem.name,
                    "split": str(spec["split"]),
                    "world_size": problem.world_size,
                    "rows_per_peer": problem.rows_per_peer,
                    "k": problem.k,
                    "n": problem.n,
                    "dtype": problem.dtype,
                    "strategy": program.strategy,
                    "chunk_rows": program.chunk_rows,
                    "buffer_count": program.buffer_count,
                    "artifact_dir": str(artifact_dir.relative_to(output)),
                    "device_prediction_us": f"{simulation.device_makespan_us:.9f}",
                    "op_dispatch_count": simulation.features.op_dispatch_count,
                    "event_record_count": simulation.features.event_record_count,
                    "event_wait_count": simulation.features.event_wait_count,
                    "buffer_reuse_count": simulation.features.buffer_reuse_count,
                }
            )
            gemm_shapes.add((program.chunk_rows, problem.n, problem.k))

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "gemm_shapes.json").open("w", encoding="utf-8") as destination:
        json.dump(
            [
                {"dtype": "float16", "m": m, "n": n, "k": k}
                for m, n, k in sorted(gemm_shapes, key=lambda item: (item[1], item[2], item[0]))
            ],
            destination,
            indent=2,
        )
        destination.write("\n")
    print(f"prepared={output}")
    print(f"candidates={len(rows)} train={sum(row['split'] == 'train' for row in rows)} heldout={sum(row['split'] == 'heldout' for row in rows)}")
    print(f"gemm_shapes={len(gemm_shapes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
