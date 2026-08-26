from __future__ import annotations

import csv
import datetime as dt
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from .topology import probe_topology


SCHEMA_RUNTIME_CAPABILITIES = "edmir.runtime_capabilities.v1"
NCCL_DEVICE_API_MIN_VERSION = 22800
NCCL_DEVICE_API_SYMBOLS = (
    "ncclCommWindowRegister",
    "ncclDevCommCreate",
    "ncclGetLsaPointer",
    "ncclLsaBarrierSession",
)


def _run(command: list[str], *, input_text: str | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=20.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return completed.returncode, output


def _parse_nccl_version(header: str) -> dict[str, int | str | None]:
    values: dict[str, int] = {}
    for name in ("MAJOR", "MINOR", "PATCH", "VERSION_CODE"):
        match = re.search(rf"^\s*#\s*define\s+NCCL_{name}\s+(\d+)", header, re.MULTILINE)
        if match:
            values[name.lower()] = int(match.group(1))
    version = None
    if all(name in values for name in ("major", "minor", "patch")):
        version = f"{values['major']}.{values['minor']}.{values['patch']}"
    return {
        "major": values.get("major"),
        "minor": values.get("minor"),
        "patch": values.get("patch"),
        "version_code": values.get("version_code"),
        "version": version,
    }


def _evaluate_device_api(
    *,
    version_code: int | None,
    device_header: bool,
    symbols: dict[str, bool],
    compile_ok: bool,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    if version_code is None:
        blockers.append("NCCL version code is unavailable")
    elif version_code < NCCL_DEVICE_API_MIN_VERSION:
        blockers.append(
            f"NCCL {version_code} is older than required {NCCL_DEVICE_API_MIN_VERSION}"
        )
    if not device_header:
        blockers.append("nccl_device.h is unavailable")
    missing = [name for name in NCCL_DEVICE_API_SYMBOLS if not symbols.get(name, False)]
    if missing:
        blockers.append("missing Device API symbols: " + ", ".join(missing))
    if device_header and not compile_ok:
        blockers.append("nvcc could not compile the NCCL Device API headers")
    return not blockers, tuple(blockers)


def _find_nvcc() -> str | None:
    if path := shutil.which("nvcc"):
        return path
    candidates = sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True)
    return str(candidates[0]) if candidates else None


def _find_nccl_header() -> Path | None:
    candidates = [
        Path("/usr/include/nccl.h"),
        Path("/usr/local/include/nccl.h"),
    ]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "include" / "nccl.h")
    candidates.extend(sorted(Path("/usr/local").glob("cuda*/include/nccl.h"), reverse=True))
    return next((path for path in candidates if path.is_file()), None)


def _read_nccl_headers(root_header: Path) -> tuple[tuple[str, ...], str]:
    include_root = root_header.parent
    paths = [root_header]
    device_header = include_root / "nccl_device.h"
    if device_header.is_file():
        paths.append(device_header)
    device_directory = include_root / "nccl_device"
    if device_directory.is_dir():
        paths.extend(sorted(device_directory.rglob("*.h")))
    readable: list[str] = []
    chunks: list[str] = []
    for path in dict.fromkeys(paths):
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            readable.append(str(path))
        except OSError:
            continue
    return tuple(readable), "\n".join(chunks)


def _probe_device_header_compile(nvcc: str | None, include_root: Path) -> tuple[bool, str]:
    if nvcc is None or not (include_root / "nccl_device.h").is_file():
        return False, "compile probe skipped: nvcc or nccl_device.h is unavailable"
    source = (
        "#include <nccl.h>\n"
        "#include <nccl_device.h>\n"
        "auto* edmir_probe_nccl_dev_comm = &ncclDevCommCreate;\n"
    )
    returncode, output = _run(
        [
            nvcc,
            "-std=c++17",
            f"-I{include_root}",
            "-x",
            "cu",
            "-c",
            "-o",
            "/dev/null",
            "-",
        ],
        input_text=source,
    )
    return returncode == 0, output


def _probe_gpus() -> tuple[list[dict[str, str]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    returncode, output = _run(command)
    if returncode:
        return [], output or "nvidia-smi query failed"
    fields = ("index", "name", "uuid", "driver_version", "pci_bus_id", "compute_capability")
    rows = []
    for values in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(values) == len(fields):
            rows.append(dict(zip(fields, (value.strip() for value in values))))
    return rows, None if rows else "nvidia-smi returned no parseable GPUs"


def probe_runtime_capabilities() -> dict[str, Any]:
    """Capture compiler/runtime facts and evaluate the fused experiment gate."""

    errors: list[str] = []
    gpus, gpu_error = _probe_gpus()
    if gpu_error:
        errors.append(gpu_error)

    nvcc = _find_nvcc()
    nvcc_returncode, nvcc_output = _run([nvcc, "--version"]) if nvcc else (127, "nvcc not found")
    if nvcc_returncode:
        errors.append(nvcc_output)

    root_header = _find_nccl_header()
    if root_header is None:
        nccl_version = _parse_nccl_version("")
        header_paths: tuple[str, ...] = ()
        header_text = ""
        compile_ok, compile_output = False, "compile probe skipped: nccl.h is unavailable"
    else:
        header_paths, header_text = _read_nccl_headers(root_header)
        nccl_version = _parse_nccl_version(root_header.read_text(encoding="utf-8", errors="replace"))
        compile_ok, compile_output = _probe_device_header_compile(nvcc, root_header.parent)

    symbols = {name: bool(re.search(rf"\b{re.escape(name)}\b", header_text)) for name in (
        "ncclMemAlloc",
        *NCCL_DEVICE_API_SYMBOLS,
    )}
    device_header = any(Path(path).name == "nccl_device.h" for path in header_paths)
    device_api_ready, blockers = _evaluate_device_api(
        version_code=nccl_version["version_code"],
        device_header=device_header,
        symbols=symbols,
        compile_ok=compile_ok,
    )

    topology = None
    try:
        topology = probe_topology().to_dict()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f"topology probe failed: {exc}")
    full_p2p = bool(topology and all(all(row) for row in topology["p2p"]))
    fused_ready = device_api_ready and full_p2p and len(gpus) >= 2
    fused_blockers = list(blockers)
    if len(gpus) < 2:
        fused_blockers.append("at least two GPUs are required")
    if not full_p2p:
        fused_blockers.append("full peer-to-peer connectivity is unavailable or unprobed")

    return {
        "schema": SCHEMA_RUNTIME_CAPABILITIES,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "gpus": gpus,
        "cuda": {
            "nvcc": nvcc,
            "nvcc_version_output": nvcc_output,
            "compile_probe_ok": compile_ok,
            "compile_probe_output": compile_output,
        },
        "nccl": {
            **nccl_version,
            "headers": list(header_paths),
            "symbols": symbols,
            "device_header": device_header,
            "device_api_ready": device_api_ready,
        },
        "topology": topology,
        "features": {
            "fused_a2a_gemm_runtime_ready": fused_ready,
            "fused_a2a_gemm_blockers": fused_blockers,
        },
        "probe_errors": errors,
    }


__all__ = [
    "NCCL_DEVICE_API_MIN_VERSION",
    "NCCL_DEVICE_API_SYMBOLS",
    "SCHEMA_RUNTIME_CAPABILITIES",
    "probe_runtime_capabilities",
]
