from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .nsys_sqlite import connect, table_columns, table_names
except ImportError:
    from nsys_sqlite import connect, table_columns, table_names


ITER_RE = re.compile(r"\.iter(?P<iteration>\d+)$")


@dataclass(frozen=True, slots=True)
class NvtxRange:
    start_ns: int
    end_ns: int
    text: str
    global_tid: int | None
    op: str
    iteration: int | None

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class RuntimeCall:
    start_ns: int
    end_ns: int
    correlation_id: int
    name: str
    global_tid: int | None


@dataclass(frozen=True, slots=True)
class GpuEvent:
    start_ns: int
    end_ns: int
    kind: str
    name: str
    stream_id: int | None
    correlation_id: int
    bytes: int | None = None
    copy_kind: str | None = None

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.start_ns


def load_string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in table_names(conn):
        return {}
    rows = conn.execute('select id, value from "StringIds"').fetchall()
    return {int(row["id"]): str(row["value"]) for row in rows}


def resolve_string(value: Any, strings: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and value in strings:
        return strings[value]
    return str(value)


def parse_edm_text(text: str) -> tuple[str, int | None]:
    iteration = None
    match = ITER_RE.search(text)
    if match is not None:
        iteration = int(match.group("iteration"))
        text = text[: match.start()]

    if not text.startswith("edm."):
        return text, iteration

    body = text[len("edm.") :]
    if ":" not in body:
        return body, iteration
    prefix, name = body.split(":", 1)
    return f"{prefix}.{name}", iteration


def read_edm_nvtx_ranges(conn: sqlite3.Connection) -> list[NvtxRange]:
    if "NVTX_EVENTS" not in table_names(conn):
        return []

    columns = table_columns(conn, "NVTX_EVENTS")
    if "text" not in columns:
        return []

    rows = conn.execute(
        'select start, end, text, globalTid from "NVTX_EVENTS" '
        "where text like ? and end is not null order by start",
        ("edm.%",),
    ).fetchall()

    ranges = []
    for row in rows:
        op, iteration = parse_edm_text(str(row["text"]))
        ranges.append(
            NvtxRange(
                start_ns=int(row["start"]),
                end_ns=int(row["end"]),
                text=str(row["text"]),
                global_tid=_optional_int(row["globalTid"]),
                op=op,
                iteration=iteration,
            )
        )
    return ranges


def read_runtime_calls(conn: sqlite3.Connection, strings: dict[int, str]) -> list[RuntimeCall]:
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in table_names(conn):
        return []

    rows = conn.execute(
        'select start, end, correlationId, nameId, globalTid '
        'from "CUPTI_ACTIVITY_KIND_RUNTIME" order by start'
    ).fetchall()

    calls = []
    for row in rows:
        calls.append(
            RuntimeCall(
                start_ns=int(row["start"]),
                end_ns=int(row["end"]),
                correlation_id=int(row["correlationId"]),
                name=resolve_string(row["nameId"], strings),
                global_tid=_optional_int(row["globalTid"]),
            )
        )
    return calls


def read_gpu_events(conn: sqlite3.Connection, strings: dict[int, str]) -> list[GpuEvent]:
    events: list[GpuEvent] = []

    if "CUPTI_ACTIVITY_KIND_KERNEL" in table_names(conn):
        columns = table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_column = _first_existing(columns, ("demangledName", "shortName", "mangledName"))
        rows = conn.execute('select * from "CUPTI_ACTIVITY_KIND_KERNEL" order by start').fetchall()
        for row in rows:
            name = resolve_string(row[name_column], strings) if name_column else "kernel"
            events.append(
                GpuEvent(
                    start_ns=int(row["start"]),
                    end_ns=int(row["end"]),
                    kind="kernel",
                    name=name,
                    stream_id=_optional_int(row["streamId"]),
                    correlation_id=int(row["correlationId"]),
                )
            )

    if "CUPTI_ACTIVITY_KIND_MEMCPY" in table_names(conn):
        copy_kind = enum_map(conn, "ENUM_CUDA_MEMCPY_OPER")
        rows = conn.execute('select * from "CUPTI_ACTIVITY_KIND_MEMCPY" order by start').fetchall()
        for row in rows:
            kind_id = _optional_int(row["copyKind"])
            kind_name = copy_kind.get(kind_id, str(kind_id)) if kind_id is not None else ""
            events.append(
                GpuEvent(
                    start_ns=int(row["start"]),
                    end_ns=int(row["end"]),
                    kind="memcpy",
                    name="cuda_memcpy",
                    stream_id=_optional_int(row["streamId"]),
                    correlation_id=int(row["correlationId"]),
                    bytes=_optional_int(row["bytes"]),
                    copy_kind=kind_name,
                )
            )

    return sorted(events, key=lambda event: (event.start_ns, event.end_ns, event.kind))


def enum_map(conn: sqlite3.Connection, table: str) -> dict[int | None, str]:
    if table not in table_names(conn):
        return {}
    rows = conn.execute(f'select id, label from "{table}"').fetchall()
    return {int(row["id"]): str(row["label"]) for row in rows}


def assign_runtime_to_nvtx(
    runtime_calls: Iterable[RuntimeCall],
    nvtx_ranges: Iterable[NvtxRange],
) -> dict[int, NvtxRange]:
    ranges = list(nvtx_ranges)
    assigned: dict[int, NvtxRange] = {}
    for call in runtime_calls:
        candidates = [
            nvtx
            for nvtx in ranges
            if nvtx.start_ns <= call.start_ns and call.end_ns <= nvtx.end_ns
        ]
        if not candidates:
            continue
        assigned[call.correlation_id] = min(candidates, key=lambda nvtx: nvtx.dur_ns)
    return assigned


def export_nsys(sqlite_path: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with connect(sqlite_path) as conn:
        strings = load_string_ids(conn)
        nvtx = read_edm_nvtx_ranges(conn)
        runtime = read_runtime_calls(conn, strings)
        gpu_events = read_gpu_events(conn, strings)
        runtime_to_nvtx = assign_runtime_to_nvtx(runtime, nvtx)

    origin = min(
        [item.start_ns for item in nvtx]
        + [item.start_ns for item in runtime]
        + [item.start_ns for item in gpu_events],
        default=0,
    )

    nvtx_rows = [
        {
            "iteration": item.iteration,
            "op": item.op,
            "text": item.text,
            "start_us": us(item.start_ns, origin),
            "end_us": us(item.end_ns, origin),
            "dur_us": dur_us(item.dur_ns),
            "global_tid": item.global_tid,
        }
        for item in nvtx
    ]
    nvtx_fields = ("iteration", "op", "text", "start_us", "end_us", "dur_us", "global_tid")
    write_csv(out / "edm_nvtx_ranges.csv", nvtx_rows, nvtx_fields)

    runtime_rows = [
        {
            "correlation_id": item.correlation_id,
            "name": item.name,
            "start_us": us(item.start_ns, origin),
            "end_us": us(item.end_ns, origin),
            "dur_us": dur_us(item.end_ns - item.start_ns),
            "global_tid": item.global_tid,
        }
        for item in runtime
    ]
    runtime_fields = ("correlation_id", "name", "start_us", "end_us", "dur_us", "global_tid")
    write_csv(out / "cuda_runtime.csv", runtime_rows, runtime_fields)

    enriched_gpu_rows = []
    for event in gpu_events:
        nvtx_range = runtime_to_nvtx.get(event.correlation_id)
        enriched_gpu_rows.append(
            {
                "iteration": nvtx_range.iteration if nvtx_range else None,
                "edm_op": nvtx_range.op if nvtx_range else "",
                "edm_text": nvtx_range.text if nvtx_range else "",
                "kind": event.kind,
                "name": event.name,
                "stream_id": event.stream_id,
                "correlation_id": event.correlation_id,
                "bytes": event.bytes,
                "copy_kind": event.copy_kind,
                "start_us": us(event.start_ns, origin),
                "end_us": us(event.end_ns, origin),
                "dur_us": dur_us(event.dur_ns),
            }
        )

    gpu_fields = (
        "iteration",
        "edm_op",
        "edm_text",
        "kind",
        "name",
        "stream_id",
        "correlation_id",
        "bytes",
        "copy_kind",
        "start_us",
        "end_us",
        "dur_us",
    )
    write_csv(out / "edm_cuda_events.csv", enriched_gpu_rows, gpu_fields)

    summary_rows = iteration_summary(enriched_gpu_rows)
    summary_fields = (
        "iteration",
        "gpu_span_us",
        "memcpy_us",
        "kernel_us",
        "bytes",
        "gpu_event_count",
        "movement_ratio",
    )
    write_csv(out / "edm_iteration_summary.csv", summary_rows, summary_fields)

    write_readable_sqlite(
        out / "edm_readable.sqlite",
        {
            "edm_nvtx_ranges": (nvtx_rows, nvtx_fields),
            "cuda_runtime": (runtime_rows, runtime_fields),
            "edm_cuda_events": (enriched_gpu_rows, gpu_fields),
            "edm_iteration_summary": (summary_rows, summary_fields),
        },
    )


def iteration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        iteration = row["iteration"]
        if iteration is None:
            continue
        grouped.setdefault(int(iteration), []).append(row)

    summaries = []
    for iteration, items in sorted(grouped.items()):
        start = min(float(item["start_us"]) for item in items)
        end = max(float(item["end_us"]) for item in items)
        memcpy_us = sum(float(item["dur_us"]) for item in items if item["kind"] == "memcpy")
        kernel_us = sum(float(item["dur_us"]) for item in items if item["kind"] == "kernel")
        bytes_moved = sum(int(item["bytes"] or 0) for item in items)
        span = end - start
        summaries.append(
            {
                "iteration": iteration,
                "gpu_span_us": f"{span:.3f}",
                "memcpy_us": f"{memcpy_us:.3f}",
                "kernel_us": f"{kernel_us:.3f}",
                "bytes": bytes_moved,
                "gpu_event_count": len(items),
                "movement_ratio": f"{(memcpy_us / span if span > 0 else 0.0):.6f}",
            }
        )
    return summaries


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readable_sqlite(
    path: Path,
    tables: dict[str, tuple[list[dict[str, Any]], tuple[str, ...]]],
) -> None:
    if path.exists():
        path.unlink()

    with sqlite3.connect(path) as conn:
        for table, (rows, fieldnames) in tables.items():
            column_defs = ", ".join(f'"{name}"' for name in fieldnames)
            conn.execute(f'create table "{table}" ({column_defs})')
            if not rows:
                continue
            placeholders = ", ".join("?" for _ in fieldnames)
            conn.executemany(
                f'insert into "{table}" values ({placeholders})',
                [[row.get(field) for field in fieldnames] for row in rows],
            )


def us(timestamp_ns: int, origin_ns: int) -> str:
    return f"{(timestamp_ns - origin_ns) / 1000.0:.3f}"


def dur_us(duration_ns: int) -> str:
    return f"{duration_ns / 1000.0:.3f}"


def _first_existing(columns: tuple[str, ...], names: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a readable EDM view from an Nsight Systems sqlite file.")
    parser.add_argument("sqlite", help="path to an nsys-exported sqlite database")
    parser.add_argument(
        "--out",
        default="edm_nsys_export",
        help="directory for exported CSV tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_nsys(args.sqlite, args.out)
    print(f"exported readable EDM tables to {args.out}")


if __name__ == "__main__":
    main()
