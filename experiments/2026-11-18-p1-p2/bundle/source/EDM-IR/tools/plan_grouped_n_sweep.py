#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (
    A2AGemmProblem,
    SCHEMA_FUSED_SCHEDULE_SPACE,
    Target,
    TopologySnapshot,
    build_fused_program,
    enumerate_fused_schedule_space,
)


def csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan the direct/split-K x grouped-N fused schedule matrix"
    )
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kernels",
        type=csv_values,
        default=("sm80_64x128x32", "sm80_64x128x32_splitk8"),
    )
    parser.add_argument("--n-tiles-per-cta", type=csv_values, default=("1", "2", "4"))
    args = parser.parse_args()

    problem = A2AGemmProblem.from_dict(
        json.loads(args.problem.read_text(encoding="utf-8"))
    )
    topology = TopologySnapshot.from_dict(
        json.loads(args.topology.read_text(encoding="utf-8"))
    )
    program = build_fused_program(problem, chunk_rows=problem.rows_per_peer)
    target = Target(
        name="a100-grouped-n-planning",
        topology=topology,
        arch="sm_80",
        gemm_provider="cutlass",
        communication_api="nccl_device",
    )
    points = enumerate_fused_schedule_space(
        program,
        target,
        gemm_kernels=tuple(args.kernels),
        n_tiles_per_cta=tuple(int(value) for value in args.n_tiles_per_cta),
    )
    report = {
        "schema": SCHEMA_FUSED_SCHEDULE_SPACE,
        "problem": problem.to_dict(),
        "topology": topology.to_dict(),
        "points": [point.to_dict() for point in points],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} points={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
