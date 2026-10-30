#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    HardwareCeilings,
    SCHEMA_ROOFLINE_ACCOUNTING,
    ScheduleSpec,
    account_schedule,
    read_schedule_measurements,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty ledger {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def projection_specs(manifest: dict) -> list[ScheduleSpec]:
    base = list(manifest["candidate_space"]["base"])
    grouped = [
        f"{family}_groupedn2"
        for family in manifest["candidate_space"]["grouped_n_2"]
    ]
    specs = []
    for sweep in manifest["sweeps"]:
        for k in sweep["k_values"]:
            for family in (*base, *grouped):
                specs.append(
                    ScheduleSpec(
                        family=family,
                        world_size=int(manifest["hardware"]["world_size"]),
                        rows_per_peer=int(sweep["rows_per_peer"]),
                        k=int(k),
                        n=int(sweep["n"]),
                        group_id=str(sweep["id"]),
                    )
                )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the fixed compute/memory numerical ledger"
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path)
    parser.add_argument("--tensor-tflops", type=float, default=312.0)
    parser.add_argument("--hbm-gbps", type=float, default=2039.0)
    parser.add_argument("--lsa-gbps", type=float, default=274.0)
    parser.add_argument("--alltoall-gbps", type=float, default=269.82)
    args = parser.parse_args()

    ceilings = HardwareCeilings(
        tensor_tflops=args.tensor_tflops,
        hbm_gbps=args.hbm_gbps,
        lsa_gbps=args.lsa_gbps,
        alltoall_gbps=args.alltoall_gbps,
    )
    measurements = read_schedule_measurements(
        tuple(args.results), world_size=args.world_size
    )
    accounts = [
        account_schedule(ScheduleSpec.from_measurement(row), ceilings=ceilings)
        for row in measurements
    ]
    candidate_rows = [account.to_dict() for account in accounts]
    write_csv(args.out_dir / "candidate_accounting.csv", candidate_rows)

    shapes: dict[tuple[str, int, int, int], list] = {}
    for account in accounts:
        key = (account.group_id, account.rows_per_peer, account.k, account.n)
        shapes.setdefault(key, []).append(account)
    oracle_rows = []
    for (group_id, m, k, n), candidates in sorted(shapes.items()):
        oracle = min(candidates, key=lambda row: (row.measured_ms, row.family))
        host = next(row for row in candidates if row.family == "host")
        oracle_rows.append(
            {
                "group_id": group_id,
                "world_size": oracle.world_size,
                "rows_per_peer": m,
                "batch_rows_per_rank": oracle.batch_rows_per_rank,
                "batch_rows_global": oracle.batch_rows_global,
                "k": k,
                "n": n,
                "oracle_family": oracle.family,
                "oracle_ms": oracle.measured_ms,
                "host_ms": host.measured_ms,
                "oracle_speedup_over_host": host.measured_ms / oracle.measured_ms,
                "split_k_slices": oracle.split_k_slices,
                "n_tiles_per_cta": oracle.n_tiles_per_cta,
                "rho_n": oracle.rho_n,
                "gamma": oracle.gamma,
                "flops_per_rank": oracle.flops_per_rank,
                "replayed_remote_bytes_per_rank": oracle.replayed_remote_bytes_per_rank,
                "local_bytes_per_rank": oracle.local_bytes_per_rank,
                "compute_floor_ms": oracle.compute_floor_ms,
                "remote_floor_ms": oracle.remote_floor_ms,
                "hbm_floor_ms": oracle.hbm_floor_ms,
                "roofline_floor_ms": oracle.roofline_floor_ms,
                "critical_path_floor_ms": oracle.critical_path_floor_ms,
                "lower_bound_bottleneck": oracle.lower_bound_bottleneck,
                "floor_fraction": oracle.floor_fraction,
                "unexplained_ms": oracle.unexplained_ms,
                "achieved_tflops": oracle.achieved_tflops,
                "effective_remote_gbps": oracle.effective_remote_gbps,
            }
        )
    write_csv(args.out_dir / "shape_oracles.csv", oracle_rows)

    fused = [row for row in accounts if row.family != "host"]
    oracle_accounts = [
        min(candidates, key=lambda row: (row.measured_ms, row.family))
        for candidates in shapes.values()
    ]
    oracle_measured = [
        row.measured_ms for row in oracle_accounts if row.measured_ms is not None
    ]
    oracle_floor_fraction = [
        row.floor_fraction for row in oracle_accounts if row.floor_fraction is not None
    ]
    oracle_unexplained = [
        row.unexplained_ms for row in oracle_accounts if row.unexplained_ms is not None
    ]
    summary = {
        "schema": SCHEMA_ROOFLINE_ACCOUNTING,
        "world_size": args.world_size,
        "ceilings": ceilings.to_dict(),
        "batch_semantics": {
            "M": "balanced rows sent by one source rank to one destination rank",
            "batch_rows_per_rank": "P*M rows consumed by one destination rank",
            "batch_rows_global": "P^2*M rows across all destination ranks",
            "warning": "These routed rows are not the request batch size.",
        },
        "inputs": [str(path) for path in args.results],
        "inputs_sha256": digest(list(args.results)),
        "candidate_rows": len(accounts),
        "shapes": len(shapes),
        "oracle_family_counts": dict(sorted(Counter(row.family for row in oracle_accounts).items())),
        "oracle_bottleneck_counts": dict(
            sorted(Counter(row.lower_bound_bottleneck for row in oracle_accounts).items())
        ),
        "fused_bottleneck_counts": dict(
            sorted(Counter(row.lower_bound_bottleneck for row in fused).items())
        ),
        "machine_balance": {
            "tensor_to_lsa_flop_per_byte": (
                ceilings.tensor_tflops * 1000.0 / ceilings.lsa_gbps
            ),
            "tensor_to_hbm_flop_per_byte": (
                ceilings.tensor_tflops * 1000.0 / ceilings.hbm_gbps
            ),
        },
        "oracle_medians": {
            "measured_ms": statistics.median(oracle_measured),
            "critical_path_floor_ms": statistics.median(
                row.critical_path_floor_ms for row in oracle_accounts
            ),
            "floor_fraction": statistics.median(oracle_floor_fraction),
            "unexplained_ms": statistics.median(oracle_unexplained),
            "achieved_tflops": statistics.median(
                row.achieved_tflops
                for row in oracle_accounts
                if row.achieved_tflops is not None
            ),
            "effective_remote_gbps": statistics.median(
                row.effective_remote_gbps
                for row in oracle_accounts
                if row.effective_remote_gbps is not None
            ),
        },
        "host_win_shapes": [
            {
                "rows_per_peer": row.rows_per_peer,
                "batch_rows_per_rank": row.batch_rows_per_rank,
                "k": row.k,
                "n": row.n,
                "measured_ms": row.measured_ms,
                "lower_bound_bottleneck": row.lower_bound_bottleneck,
            }
            for row in oracle_accounts
            if row.family == "host"
        ],
        "fused_medians": {
            "achieved_tflops": statistics.median(
                row.achieved_tflops for row in fused if row.achieved_tflops is not None
            ),
            "effective_remote_gbps": statistics.median(
                row.effective_remote_gbps
                for row in fused
                if row.effective_remote_gbps is not None
            ),
            "floor_fraction": statistics.median(
                row.floor_fraction for row in fused if row.floor_fraction is not None
            ),
            "unexplained_ms": statistics.median(
                row.unexplained_ms for row in fused if row.unexplained_ms is not None
            ),
        },
    }

    if args.projection_manifest is not None:
        manifest = json.loads(args.projection_manifest.read_text(encoding="utf-8"))
        projected = [
            account_schedule(spec, ceilings=ceilings).to_dict()
            for spec in projection_specs(manifest)
        ]
        write_csv(args.out_dir / "projection_accounting.csv", projected)
        summary["projection"] = {
            "manifest": str(args.projection_manifest),
            "manifest_sha256": digest([args.projection_manifest]),
            "world_size": manifest["hardware"]["world_size"],
            "candidate_rows": len(projected),
            "shapes": len(
                {
                    (row["group_id"], row["rows_per_peer"], row["k"], row["n"])
                    for row in projected
                }
            ),
            "bottleneck_counts": dict(
                sorted(Counter(row["lower_bound_bottleneck"] for row in projected).items())
            ),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"candidates={len(accounts)} shapes={len(shapes)} "
        f"oracles={summary['oracle_family_counts']}"
    )
    print(f"wrote={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
