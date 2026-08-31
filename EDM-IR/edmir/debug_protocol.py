from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEBUG_SCHEMA = "edmir.sm89-debug-protocol.v1"


@dataclass(frozen=True, slots=True)
class GpuInventoryRow:
    index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int


@dataclass(frozen=True, slots=True)
class GpuProcessRow:
    uuid: str
    pid: int
    used_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class GpuAdmission:
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugCommand:
    name: str
    phase: str
    argv: tuple[str, ...]
    expected_returncodes: tuple[int, ...] = (0,)
    expected_output: str | None = None
    gpu_required: bool = False


def _integer(value: str) -> int:
    normalized = value.strip().strip('"')
    if normalized in {"", "N/A", "[N/A]", "Not Supported"}:
        return 0
    return int(normalized)


def parse_gpu_inventory(text: str) -> tuple[GpuInventoryRow, ...]:
    rows: list[GpuInventoryRow] = []
    for fields in csv.reader(text.splitlines(), skipinitialspace=True):
        if not fields or all(not value.strip() for value in fields):
            continue
        if len(fields) != 6:
            raise ValueError(f"expected six GPU inventory fields, got {len(fields)}")
        rows.append(
            GpuInventoryRow(
                index=_integer(fields[0]),
                name=fields[1].strip().strip('"'),
                uuid=fields[2].strip().strip('"'),
                memory_total_mib=_integer(fields[3]),
                memory_used_mib=_integer(fields[4]),
                utilization_percent=_integer(fields[5]),
            )
        )
    return tuple(rows)


def parse_gpu_processes(text: str) -> tuple[GpuProcessRow, ...]:
    rows: list[GpuProcessRow] = []
    for fields in csv.reader(text.splitlines(), skipinitialspace=True):
        if not fields or all(not value.strip() for value in fields):
            continue
        if len(fields) != 3:
            raise ValueError(f"expected three GPU process fields, got {len(fields)}")
        memory = fields[2].strip().strip('"')
        rows.append(
            GpuProcessRow(
                uuid=fields[0].strip().strip('"'),
                pid=_integer(fields[1]),
                used_memory_mib=None
                if memory in {"", "N/A", "[N/A]", "Not Supported"}
                else int(memory),
            )
        )
    return tuple(rows)


def evaluate_gpu_admission(
    inventory: tuple[GpuInventoryRow, ...],
    processes: tuple[GpuProcessRow, ...],
    requested_devices: tuple[int, ...],
    *,
    maximum_memory_used_mib: int,
    maximum_utilization_percent: int,
) -> GpuAdmission:
    reasons: list[str] = []
    by_index = {row.index: row for row in inventory}
    requested_rows: list[GpuInventoryRow] = []
    for index in requested_devices:
        row = by_index.get(index)
        if row is None:
            reasons.append(f"requested GPU {index} is absent")
            continue
        requested_rows.append(row)
        if "4090" not in row.name:
            reasons.append(f"GPU {index} is {row.name!r}, not an RTX 4090")
        if row.memory_used_mib > maximum_memory_used_mib:
            reasons.append(
                f"GPU {index} uses {row.memory_used_mib} MiB, above "
                f"{maximum_memory_used_mib} MiB"
            )
        if row.utilization_percent > maximum_utilization_percent:
            reasons.append(
                f"GPU {index} utilization is {row.utilization_percent}%, above "
                f"{maximum_utilization_percent}%"
            )
    requested_uuids = {row.uuid for row in requested_rows}
    for process in processes:
        if process.uuid in requested_uuids:
            index = next(
                row.index for row in requested_rows if row.uuid == process.uuid
            )
            reasons.append(f"GPU {index} has foreign compute PID {process.pid}")
    return GpuAdmission(admitted=not reasons, reasons=tuple(reasons))


def build_debug_commands(
    root: Path,
    output: Path,
    *,
    python: str,
    rounds: int,
) -> tuple[DebugCommand, ...]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    graph = output / "artifacts" / "runtime-validation.dataflow.json"
    fixture_artifact = output / "artifacts" / "fixture-host-sm89"
    observed_topology = output / "environment" / "observed-topology.json"
    observed_artifact = output / "artifacts" / "observed-host-sm89"
    host_build = output / "build" / "host"
    cuda_build = output / "build" / "cuda"
    commands: list[DebugCommand] = [
        DebugCommand(
            "python-tests",
            "cpu",
            (
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "*_test.py",
            ),
        ),
        DebugCommand(
            "host-configure",
            "cpu",
            (
                "cmake",
                "-S",
                str(root / "runtime"),
                "-B",
                str(host_build),
                "-DEDMIR_BUILD_TESTS=ON",
            ),
        ),
        DebugCommand(
            "host-build", "cpu", ("cmake", "--build", str(host_build), "-j")
        ),
        DebugCommand(
            "host-ctest",
            "cpu",
            ("ctest", "--test-dir", str(host_build), "--output-on-failure"),
        ),
        DebugCommand(
            "frontend",
            "cpu",
            (
                python,
                "-m",
                "edmir",
                "frontend",
                str(root / "examples" / "runtime_validation_problem_fp16.json"),
                "--out",
                str(graph),
            ),
        ),
        DebugCommand(
            "fixture-host-sm89-codegen",
            "cpu",
            (
                python,
                "-m",
                "edmir",
                "compile-graph",
                str(graph),
                "--profile",
                str(root / "examples" / "synthetic_profile.json"),
                "--topology",
                str(root / "examples" / "full_p2p_topology.json"),
                "--target",
                "rtx4090-debug-fixture",
                "--arch",
                "sm_89",
                "--gemm-provider",
                "cublas",
                "--communication-api",
                "host_nccl",
                "--backend",
                "cpp",
                "--chunk-rows",
                "8",
                "--buffers",
                "1,2",
                "--out-dir",
                str(fixture_artifact),
            ),
        ),
        DebugCommand(
            "sm89-fused-fail-closed",
            "cpu",
            (
                python,
                "-m",
                "edmir",
                "lower-fused",
                str(graph),
                "--topology",
                str(root / "examples" / "full_p2p_topology.json"),
                "--chunk-rows",
                "8",
                "--target",
                "rtx4090-negative-gate",
                "--arch",
                "sm_89",
                "--gemm-provider",
                "cutlass",
                "--backend",
                "cpp",
                "--out-dir",
                str(output / "artifacts" / "forbidden-fused-sm89"),
            ),
            expected_returncodes=(1,),
            expected_output="CUTLASS v0.1 supports only arch=sm_80",
        ),
        DebugCommand(
            "topology-probe",
            "gpu",
            (python, "-m", "edmir", "probe", "--out", str(observed_topology)),
            gpu_required=True,
        ),
        DebugCommand(
            "runtime-probe",
            "gpu",
            (
                python,
                "-m",
                "edmir",
                "probe-runtime",
                "--out",
                str(output / "environment" / "runtime-capabilities.json"),
            ),
            gpu_required=True,
        ),
        DebugCommand(
            "observed-host-sm89-codegen",
            "gpu",
            (
                python,
                "-m",
                "edmir",
                "compile-graph",
                str(graph),
                "--profile",
                str(root / "examples" / "synthetic_profile.json"),
                "--topology",
                str(observed_topology),
                "--target",
                "rtx4090-observed",
                "--arch",
                "sm_89",
                "--gemm-provider",
                "cublas",
                "--communication-api",
                "host_nccl",
                "--backend",
                "cpp",
                "--chunk-rows",
                "8",
                "--buffers",
                "1,2",
                "--out-dir",
                str(observed_artifact),
            ),
            gpu_required=True,
        ),
        DebugCommand(
            "cuda-configure",
            "gpu",
            (
                "cmake",
                "-S",
                str(root / "runtime"),
                "-B",
                str(cuda_build),
                "-DEDMIR_ENABLE_CUDA=ON",
                "-DEDMIR_BUILD_TESTS=ON",
                f"-DEDMIR_GENERATED_DIR={observed_artifact}",
            ),
            gpu_required=True,
        ),
        DebugCommand(
            "cuda-build",
            "gpu",
            ("cmake", "--build", str(cuda_build), "-j"),
            gpu_required=True,
        ),
        DebugCommand(
            "cuda-ctest",
            "gpu",
            ("ctest", "--test-dir", str(cuda_build), "--output-on-failure"),
            gpu_required=True,
        ),
    ]
    for round_index in range(1, rounds + 1):
        commands.extend(
            (
                DebugCommand(
                    f"numerical-round-{round_index:02d}",
                    "gpu",
                    (str(cuda_build / "edmir_validate_generated"),),
                    gpu_required=True,
                ),
                DebugCommand(
                    f"reuse-round-{round_index:02d}",
                    "gpu",
                    (
                        str(cuda_build / "edmir_benchmark_generated"),
                        "--warmup",
                        "3",
                        "--iterations",
                        "10",
                    ),
                    gpu_required=True,
                ),
            )
        )
    return tuple(commands)


__all__ = [
    "DEBUG_SCHEMA",
    "DebugCommand",
    "GpuAdmission",
    "GpuInventoryRow",
    "GpuProcessRow",
    "build_debug_commands",
    "evaluate_gpu_admission",
    "parse_gpu_inventory",
    "parse_gpu_processes",
]
