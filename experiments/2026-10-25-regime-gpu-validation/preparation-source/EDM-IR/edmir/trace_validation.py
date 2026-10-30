from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .compiler import Program, derive_dependencies


@dataclass(frozen=True, slots=True)
class ObservedGpuEvent:
    iteration: int
    op_id: str
    kind: str
    device_id: int
    start_us: float
    end_us: float
    copy_kind: str = ""


@dataclass(frozen=True, slots=True)
class TraceViolation:
    kind: str
    message: str
    src_op: str | None = None
    dst_op: str | None = None
    device_id: int | None = None


@dataclass(frozen=True, slots=True)
class TraceValidationReport:
    valid: bool
    iteration: int
    expected_ops: int
    covered_ops: int
    gpu_events: int
    gpu_span_us: float
    comm_compute_overlap_us: float
    overlap_by_device_us: dict[int, float]
    dependency_device_checks: int
    dependency_kind_checks: dict[str, int]
    min_dependency_slack_us: float | None
    violations: tuple[TraceViolation, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "violations": [asdict(item) for item in self.violations],
        }


def _op_id(value: str) -> str:
    return value[len("op.") :] if value.startswith("op.") else value


def read_observed_gpu_events(path: str | Path) -> tuple[ObservedGpuEvent, ...]:
    events = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not row.get("iteration") or not row.get("edm_op") or not row.get("device_id"):
                continue
            kind = str(row["kind"])
            copy_kind = str(row.get("copy_kind", ""))
            if kind == "memcpy" and copy_kind != "Device-to-Device":
                continue
            if kind not in ("kernel", "memcpy"):
                continue
            events.append(
                ObservedGpuEvent(
                    iteration=int(row["iteration"]),
                    op_id=_op_id(str(row["edm_op"])),
                    kind=kind,
                    device_id=int(row["device_id"]),
                    start_us=float(row["start_us"]),
                    end_us=float(row["end_us"]),
                    copy_kind=copy_kind,
                )
            )
    return tuple(events)


def _op_devices(op) -> set[int]:
    if op.kind == "transfer" and op.peer_rank != op.rank:
        return {int(op.peer_rank), op.rank}
    return {op.rank}


def _comm_compute_overlap(
    op_map: dict[str, object],
    by_op_device: dict[tuple[str, int], list[ObservedGpuEvent]],
) -> dict[int, float]:
    intervals: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for (op_id, device), events in by_op_device.items():
        op = op_map[op_id]
        family = "comm" if op.kind == "transfer" else "compute" if op.kind == "gemm" else ""
        if not family:
            continue
        intervals.setdefault(device, {}).setdefault(family, []).append(
            (min(event.start_us for event in events), max(event.end_us for event in events))
        )
    overlap = {}
    for device, families in sorted(intervals.items()):
        total = 0.0
        for comm_start, comm_end in families.get("comm", ()):
            for compute_start, compute_end in families.get("compute", ()):
                total += max(0.0, min(comm_end, compute_end) - max(comm_start, compute_start))
        overlap[device] = total
    return overlap


def validate_trace(
    program: Program,
    events: tuple[ObservedGpuEvent, ...],
    *,
    iteration: int | None = None,
    tolerance_us: float = 0.5,
) -> TraceValidationReport:
    if tolerance_us < 0:
        raise ValueError("trace tolerance must be non-negative")
    available = sorted({event.iteration for event in events})
    if not available:
        raise ValueError("trace contains no mapped EDM GPU events")
    selected = available[-1] if iteration is None else iteration
    if selected not in available:
        raise ValueError(f"trace iteration {selected} is unavailable; found {available}")

    selected_events = tuple(event for event in events if event.iteration == selected)
    op_map = {op.id: op for op in program.ops}
    expected = {op.id: op for op in program.ops if op.kind != "barrier"}
    by_op_device: dict[tuple[str, int], list[ObservedGpuEvent]] = {}
    violations: list[TraceViolation] = []
    for event in selected_events:
        op = op_map.get(event.op_id)
        if op is None:
            violations.append(
                TraceViolation("unknown_op", f"trace references unknown op {event.op_id}")
            )
            continue
        if event.device_id not in _op_devices(op):
            violations.append(
                TraceViolation(
                    "unexpected_device",
                    f"{event.op_id} ran on unexpected device {event.device_id}",
                    src_op=event.op_id,
                    device_id=event.device_id,
                )
            )
        if op.kind == "gemm" and event.kind != "kernel":
            violations.append(
                TraceViolation(
                    "unexpected_event_kind",
                    f"GEMM {event.op_id} mapped to {event.kind}",
                    src_op=event.op_id,
                    device_id=event.device_id,
                )
            )
        if op.kind == "transfer":
            expected_kind = "memcpy" if op.peer_rank == op.rank else "kernel"
            if event.kind != expected_kind:
                violations.append(
                    TraceViolation(
                        "unexpected_event_kind",
                        f"transfer {event.op_id} expected {expected_kind}, got {event.kind}",
                        src_op=event.op_id,
                        device_id=event.device_id,
                    )
                )
        by_op_device.setdefault((event.op_id, event.device_id), []).append(event)

    covered = 0
    for op_id, op in expected.items():
        missing = [device for device in _op_devices(op) if (op_id, device) not in by_op_device]
        if missing:
            violations.append(
                TraceViolation(
                    "missing_device_event",
                    f"{op_id} has no mapped GPU event on devices {missing}",
                    src_op=op_id,
                )
            )
        else:
            covered += 1

    dependency_checks = 0
    dependency_kinds: Counter[str] = Counter()
    dependency_slacks = []
    for dependency in derive_dependencies(program).dependencies:
        src = op_map[dependency.src]
        dst = op_map[dependency.dst]
        for device in sorted(_op_devices(src) & _op_devices(dst)):
            src_events = by_op_device.get((src.id, device), ())
            dst_events = by_op_device.get((dst.id, device), ())
            if not src_events or not dst_events:
                continue
            dependency_checks += 1
            dependency_kinds[dependency.kind] += 1
            src_end = max(event.end_us for event in src_events)
            dst_start = min(event.start_us for event in dst_events)
            dependency_slacks.append(dst_start - src_end)
            if dst_start + tolerance_us < src_end:
                violations.append(
                    TraceViolation(
                        "dependency_order",
                        f"{dependency.kind} violated by {src_end - dst_start:.3f} us",
                        src_op=src.id,
                        dst_op=dst.id,
                        device_id=device,
                    )
                )

    mapped_events = [
        event for event in selected_events if event.op_id in op_map and event.kind in ("kernel", "memcpy")
    ]
    gpu_span = (
        max(event.end_us for event in mapped_events) - min(event.start_us for event in mapped_events)
        if mapped_events
        else 0.0
    )
    overlap = _comm_compute_overlap(op_map, by_op_device)
    return TraceValidationReport(
        valid=not violations,
        iteration=selected,
        expected_ops=len(expected),
        covered_ops=covered,
        gpu_events=len(selected_events),
        gpu_span_us=gpu_span,
        comm_compute_overlap_us=sum(overlap.values()),
        overlap_by_device_us=overlap,
        dependency_device_checks=dependency_checks,
        dependency_kind_checks=dict(sorted(dependency_kinds.items())),
        min_dependency_slack_us=min(dependency_slacks) if dependency_slacks else None,
        violations=tuple(violations),
    )


__all__ = [
    "ObservedGpuEvent",
    "TraceValidationReport",
    "TraceViolation",
    "read_observed_gpu_events",
    "validate_trace",
]
