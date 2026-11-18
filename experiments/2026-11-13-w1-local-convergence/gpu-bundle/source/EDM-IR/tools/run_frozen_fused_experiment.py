#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "edmir.frozen-fused-experiment.v1"


def csv_values(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def validate_manifest(value: dict) -> None:
    if value.get("schema") != SCHEMA:
        raise ValueError(f"expected manifest schema {SCHEMA!r}")
    if value.get("status") != "frozen":
        raise ValueError("experiment manifest must have status=frozen")
    hardware = value["hardware"]
    if int(hardware["world_size"]) != 8:
        raise ValueError("this transfer protocol requires exactly eight GPUs")
    if hardware["arch"] != "sm_80":
        raise ValueError("this transfer protocol is frozen for SM80")
    identifiers: set[str] = set()
    for sweep in value["sweeps"]:
        identifier = str(sweep["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate sweep id {identifier!r}")
        identifiers.add(identifier)
        if min(int(sweep["rows_per_peer"]), int(sweep["n"])) <= 0:
            raise ValueError(f"invalid dimensions in sweep {identifier}")
        if not sweep["k_values"] or any(int(k) <= 0 for k in sweep["k_values"]):
            raise ValueError(f"invalid K values in sweep {identifier}")
        if int(sweep["repeats"]) != len(sweep["order_seeds"]):
            raise ValueError(f"repeat/seed count mismatch in sweep {identifier}")


def validate_topology(value: dict) -> None:
    if int(value.get("world_size", 0)) != 8:
        raise ValueError("topology snapshot world_size does not match frozen manifest")
    devices = value.get("devices", [])
    if len(devices) != 8 or any("A100" not in str(device) for device in devices):
        raise ValueError("topology snapshot must identify eight A100 GPUs")
    p2p = value.get("p2p", [])
    if len(p2p) != 8 or any(len(row) != 8 for row in p2p):
        raise ValueError("topology snapshot must report full P2P connectivity")
    for source in range(8):
        for destination in range(8):
            if source != destination and not bool(p2p[source][destination]):
                raise ValueError("topology snapshot must report full P2P connectivity")
    labels = value.get("link_labels", [])
    if len(labels) != 8 or any(len(row) != 8 for row in labels):
        raise ValueError("topology snapshot must include the observed link-label matrix")
    for source in range(8):
        for destination in range(8):
            if source != destination and not str(labels[source][destination]).startswith("NV"):
                raise ValueError("every off-diagonal topology link must be NVLink")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_environment(output: Path, *, cutlass: Path, nccl_home: Path) -> None:
    directory = output / "environment"
    directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    commands = {
        "uname.txt": ["uname", "-a"],
        "nvidia-smi-list.txt": ["nvidia-smi", "-L"],
        "nvidia-smi-query.txt": ["nvidia-smi", "-q"],
        "nvidia-smi-topology.txt": ["nvidia-smi", "topo", "-m"],
        "nvcc-version.txt": ["nvcc", "--version"],
        "cutlass-commit.txt": ["git", "-C", str(cutlass), "rev-parse", "HEAD"],
        "edmir-commit.txt": ["git", "-C", str(repository), "rev-parse", "HEAD"],
        "nccl-library.txt": ["ls", "-l", str(nccl_home / "lib/libnccl.so")],
    }
    for filename, command in commands.items():
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (directory / filename).write_text(
            "$ " + " ".join(command) + "\n" + result.stdout,
            encoding="utf-8",
        )


def command_for_sweep(
    sweep: dict,
    *,
    tool: Path,
    runtime: Path,
    topology: Path,
    cutlass: Path,
    nccl_home: Path,
    output: Path,
    devices: str,
    repeat: int,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(tool),
        "--runtime",
        str(runtime),
        "--topology",
        str(topology),
        "--cutlass",
        str(cutlass),
        "--nccl-home",
        str(nccl_home),
        "--out",
        str(output / str(sweep["id"]) / f"repeat-{repeat:02d}"),
        "--rows-per-peer",
        str(sweep["rows_per_peer"]),
        "--n",
        str(sweep["n"]),
        "--k-values",
        csv_values(sweep["k_values"]),
        "--split-k-values",
        csv_values(sweep["split_k_values"]),
        "--warmup",
        str(sweep["warmup"]),
        "--iterations",
        str(sweep["iterations"]),
        "--devices",
        devices,
        "--order-seed",
        str(seed),
        "--validate-first",
    ]
    grouped = sweep.get("grouped_n_values", [])
    if grouped:
        command.extend(("--grouped-n-values", csv_values(grouped)))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a frozen zero-shot/few-shot fused schedule experiment"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--nccl-home", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--role",
        choices=("all", "calibration", "heldout"),
        default="all",
        help="run the whole frozen matrix or one preregistered role",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute commands; without this flag the tool only records and prints the plan",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if len(tuple(value for value in args.devices.split(",") if value)) != 8:
        raise ValueError("--devices must name exactly eight visible GPUs")
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    validate_topology(topology)

    tool = Path(__file__).with_name("run_fused_kernel_sweep.py")
    commands: list[dict] = []
    for sweep in manifest["sweeps"]:
        if args.role != "all" and sweep["role"] != args.role:
            continue
        for repeat, seed in enumerate(sweep["order_seeds"], start=1):
            command = command_for_sweep(
                sweep,
                tool=tool,
                runtime=args.runtime.resolve(),
                topology=args.topology.resolve(),
                cutlass=args.cutlass.resolve(),
                nccl_home=args.nccl_home.resolve(),
                output=args.out.resolve(),
                devices=args.devices,
                repeat=repeat,
                seed=int(seed),
            )
            commands.append(
                {
                    "sweep_id": sweep["id"],
                    "role": sweep["role"],
                    "repeat": repeat,
                    "order_seed": seed,
                    "command": command,
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "edmir.frozen-fused-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "manifest": str(args.manifest.resolve()),
        "manifest_status": manifest["status"],
        "manifest_sha256": file_sha256(args.manifest),
        "topology_sha256": file_sha256(args.topology),
        "execute": args.execute,
        "role": args.role,
        "commands": commands,
    }
    (args.out / "run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.execute:
        capture_environment(
            args.out,
            cutlass=args.cutlass.resolve(),
            nccl_home=args.nccl_home.resolve(),
        )
    for entry in commands:
        print(" ".join(entry["command"]))
        if args.execute:
            subprocess.run(entry["command"], check=True)
    print(
        f"planned={len(commands)} execute={str(args.execute).lower()} "
        f"record={args.out / 'run_record.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
