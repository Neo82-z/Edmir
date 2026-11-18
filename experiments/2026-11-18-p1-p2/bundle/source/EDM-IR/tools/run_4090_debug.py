#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.debug_protocol import (  # noqa: E402
    DEBUG_SCHEMA,
    DebugCommand,
    build_debug_commands,
    evaluate_gpu_admission,
    parse_gpu_inventory,
    parse_gpu_processes,
)


INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
)
PROCESS_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("run id must contain only letters, digits, dot, dash, underscore")
    return value


def _run(
    command: DebugCommand,
    *,
    root: Path,
    logs: Path,
    env: dict[str, str],
) -> tuple[int, str, Path]:
    try:
        result = subprocess.run(
            command.argv,
            cwd=root,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        returncode = result.returncode
        output = result.stdout
    except FileNotFoundError as error:
        returncode = 127
        output = f"{type(error).__name__}: {error}\n"
    log = logs / f"{command.name}.txt"
    log.write_text(
        "$ " + shlex.join(command.argv) + "\n" + output,
        encoding="utf-8",
    )
    return returncode, output, log


def _capture(
    name: str,
    argv: tuple[str, ...],
    *,
    root: Path,
    logs: Path,
    env: dict[str, str],
) -> tuple[int, str, Path]:
    command = DebugCommand(name=name, phase="environment", argv=argv)
    return _run(command, root=root, logs=logs, env=env)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record_outputs(output: Path, manifest: dict) -> None:
    for directory_name in ("artifacts", "environment"):
        directory = output / directory_name
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(output))] = {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                    "required": True,
                }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the non-destructive, diagnostic-only EDM-IR RTX 4090 protocol"
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--maximum-memory-used-mib", type=int, default=1024)
    parser.add_argument("--maximum-utilization-percent", type=int, default=10)
    parser.add_argument("--cuda-toolkit-root", type=Path)
    parser.add_argument("--nccl-include-dir", type=Path)
    parser.add_argument("--nccl-library", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the frozen plan; otherwise create a dry-run record only",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="allow GPU phases only after all four requested RTX 4090s pass the idle gate",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repository = root.parent
    requested_devices = tuple(int(value) for value in args.devices.split(",") if value)
    if len(requested_devices) != 4 or len(set(requested_devices)) != 4:
        raise ValueError("--devices must name four distinct GPU indices")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if min(args.maximum_memory_used_mib, args.maximum_utilization_percent) < 0:
        raise ValueError("idle thresholds must be non-negative")
    toolchain_paths = (
        args.cuda_toolkit_root,
        args.nccl_include_dir,
        args.nccl_library,
    )
    if any(path is not None for path in toolchain_paths) and not all(
        path is not None for path in toolchain_paths
    ):
        raise ValueError(
            "--cuda-toolkit-root, --nccl-include-dir, and --nccl-library "
            "must be supplied together"
        )
    resolved_toolchain: tuple[Path, Path, Path] | None = None
    if all(path is not None for path in toolchain_paths):
        resolved = tuple(path.expanduser().resolve() for path in toolchain_paths)
        for path in resolved:
            if not path.exists():
                raise ValueError(f"toolchain path does not exist: {path}")
        resolved_toolchain = resolved
    run_id = _safe_run_id(
        args.run_id
        or datetime.now(timezone.utc).strftime("edmir-4090-debug-%Y%m%dT%H%M%SZ")
    )
    output = args.out_root.expanduser().resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    (output / "artifacts").mkdir()
    (output / "environment").mkdir()

    commands = build_debug_commands(
        root,
        output,
        python=sys.executable,
        rounds=args.rounds,
        cuda_toolkit_root=None if resolved_toolchain is None else resolved_toolchain[0],
        nccl_include_dir=None if resolved_toolchain is None else resolved_toolchain[1],
        nccl_library=None if resolved_toolchain is None else resolved_toolchain[2],
    )
    manifest_path = output / "manifest.json"
    manifest: dict = {
        "schema": DEBUG_SCHEMA,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "planned" if not args.execute else "running",
        "run_id": run_id,
        "repository": str(repository),
        "root": str(root),
        "execute": args.execute,
        "allow_gpu": args.allow_gpu,
        "requested_devices": requested_devices,
        "rounds": args.rounds,
        "paper_evidence": False,
        "purpose": "correctness, portability, fail-closed, and runtime reuse debugging",
        "forbidden_actions": [
            "delete pre-existing files",
            "stop containers or services",
            "kill foreign processes",
            "admit SM80 fused evidence on SM89",
            "treat diagnostic latency as a paper result",
        ],
        "idle_gate": {
            "maximum_memory_used_mib": args.maximum_memory_used_mib,
            "maximum_utilization_percent": args.maximum_utilization_percent,
        },
        "cuda_toolchain": None
        if resolved_toolchain is None
        else {
            "root": str(resolved_toolchain[0]),
            "nccl_include_dir": str(resolved_toolchain[1]),
            "nccl_library": str(resolved_toolchain[2]),
        },
        "commands": [
            {
                "name": command.name,
                "phase": command.phase,
                "argv": command.argv,
                "gpu_required": command.gpu_required,
                "expected_returncodes": command.expected_returncodes,
                "expected_output": command.expected_output,
                "status": "planned",
            }
            for command in commands
        ],
        "files": {},
    }
    _write_manifest(manifest_path, manifest)
    if not args.execute:
        print(f"planned={len(commands)} output={output}")
        return 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    environment_commands = (
        ("date", ("date", "-Iseconds")),
        ("uname", ("uname", "-a")),
        ("python-version", (sys.executable, "--version")),
        ("cmake-version", ("cmake", "--version")),
        ("nvcc-version", ("nvcc", "--version")),
        ("nvidia-smi-list", ("nvidia-smi", "-L")),
        ("nvidia-smi-topology", ("nvidia-smi", "topo", "-m")),
        ("docker-ps-read-only", ("docker", "ps", "--no-trunc")),
        ("git-commit", ("git", "-C", str(repository), "rev-parse", "HEAD")),
    )
    for name, argv in environment_commands:
        returncode, _output, log = _capture(
            name, argv, root=root, logs=logs, env=env
        )
        manifest["files"][str(log.relative_to(output))] = {
            "sha256": _sha256(log),
            "returncode": returncode,
            "required": False,
        }
    _write_manifest(manifest_path, manifest)

    failure: str | None = None
    gpu_admitted = False
    gpu_skip_reason: str | None = None
    for index, command in enumerate(commands):
        entry = manifest["commands"][index]
        if command.gpu_required:
            if not args.allow_gpu:
                entry["status"] = "skipped"
                entry["reason"] = "--allow-gpu was not supplied"
                continue
            inventory_code, inventory_text, inventory_log = _capture(
                f"gpu-inventory-before-{command.name}",
                INVENTORY_COMMAND,
                root=root,
                logs=logs,
                env=env,
            )
            process_code, process_text, process_log = _capture(
                f"gpu-process-before-{command.name}",
                PROCESS_COMMAND,
                root=root,
                logs=logs,
                env=env,
            )
            for code, path in (
                (inventory_code, inventory_log),
                (process_code, process_log),
            ):
                manifest["files"][str(path.relative_to(output))] = {
                    "sha256": _sha256(path),
                    "returncode": code,
                    "required": True,
                }
            if inventory_code or process_code:
                gpu_skip_reason = "GPU idle gate could not query nvidia-smi"
                reasons = (gpu_skip_reason,)
            else:
                try:
                    admission = evaluate_gpu_admission(
                        parse_gpu_inventory(inventory_text),
                        parse_gpu_processes(process_text),
                        requested_devices,
                        maximum_memory_used_mib=args.maximum_memory_used_mib,
                        maximum_utilization_percent=args.maximum_utilization_percent,
                    )
                except ValueError as error:
                    gpu_skip_reason = "GPU idle gate could not parse nvidia-smi"
                    reasons = (f"{type(error).__name__}: {error}",)
                else:
                    gpu_admitted = admission.admitted
                    reasons = admission.reasons
                    if not admission.admitted:
                        gpu_skip_reason = "GPU idle gate rejected execution"
            manifest.setdefault("gpu_admission_checks", []).append(
                {
                    "before": command.name,
                    "admitted": not gpu_skip_reason,
                    "reasons": reasons,
                }
            )
            if gpu_skip_reason:
                entry["status"] = "skipped"
                entry["reason"] = gpu_skip_reason
                for later in manifest["commands"][index + 1 :]:
                    if later["gpu_required"]:
                        later["status"] = "skipped"
                        later["reason"] = gpu_skip_reason
                break
        entry["status"] = "running"
        entry["started_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        returncode, output_text, log = _run(command, root=root, logs=logs, env=env)
        entry["finished_at"] = _utc_now()
        entry["returncode"] = returncode
        entry["log"] = str(log.relative_to(output))
        entry["log_sha256"] = _sha256(log)
        expected = returncode in command.expected_returncodes
        if command.expected_output is not None:
            expected = expected and command.expected_output in output_text
        entry["status"] = "passed" if expected else "failed"
        if not expected:
            failure = f"command {command.name!r} violated its expected outcome"
            for later in manifest["commands"][index + 1 :]:
                if later["status"] == "planned":
                    later["status"] = "not-run"
                    later["reason"] = failure
            break

    manifest["updated_at"] = _utc_now()
    _record_outputs(output, manifest)
    if failure:
        manifest["status"] = "failed"
        manifest["failure"] = failure
    elif args.allow_gpu and gpu_admitted and not gpu_skip_reason:
        manifest["status"] = "passed"
    else:
        manifest["status"] = "cpu-passed-gpu-skipped"
    _write_manifest(manifest_path, manifest)
    print(f"status={manifest['status']} output={output}")
    if failure:
        print(f"failure={failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
