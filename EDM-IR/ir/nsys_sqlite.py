from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]
    rows: int


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def table_names(conn: sqlite3.Connection) -> tuple[str, ...]:
    rows = conn.execute(
        "select name from sqlite_master where type='table' order by name"
    ).fetchall()
    return tuple(str(row["name"]) for row in rows)


def table_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = conn.execute(f'pragma table_info("{table}")').fetchall()
    return tuple(str(row["name"]) for row in rows)


def table_row_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f'select count(*) as count from "{table}"').fetchone()
    return int(row["count"])


def inspect_tables(conn: sqlite3.Connection) -> tuple[TableInfo, ...]:
    infos = []
    for name in table_names(conn):
        infos.append(
            TableInfo(
                name=name,
                columns=table_columns(conn, name),
                rows=table_row_count(conn, name),
            )
        )
    return tuple(infos)


def string_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    candidates = []
    for info in inspect_tables(conn):
        lowered = {column.lower() for column in info.columns}
        if {"id", "value"}.issubset(lowered) or {"id", "text"}.issubset(lowered):
            candidates.append(info.name)
    return tuple(candidates)


def find_edm_strings(conn: sqlite3.Connection, limit: int = 200) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for table in string_tables(conn):
        columns = table_columns(conn, table)
        text_column = _first_existing(columns, ("value", "text", "name"))
        if text_column is None:
            continue
        rows = conn.execute(
            f'select "{text_column}" as text from "{table}" '
            f'where "{text_column}" like ? order by "{text_column}" limit ?',
            ("edm.%", limit),
        ).fetchall()
        hits.extend((table, str(row["text"])) for row in rows)
    return hits[:limit]


def likely_event_tables(conn: sqlite3.Connection) -> tuple[TableInfo, ...]:
    keywords = ("NVTX", "CUPTI", "CUDA", "RUNTIME", "KERNEL", "MEM", "EVENT")
    infos = []
    for info in inspect_tables(conn):
        upper_name = info.name.upper()
        upper_columns = " ".join(info.columns).upper()
        if any(keyword in upper_name or keyword in upper_columns for keyword in keywords):
            infos.append(info)
    return tuple(infos)


def diagnostic_events(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    if "DIAGNOSTIC_EVENT" not in table_names(conn):
        return []
    return conn.execute(
        "select timestamp, source, severity, text from DIAGNOSTIC_EVENT order by timestamp"
    ).fetchall()


def target_gpus(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    if "TARGET_INFO_GPU" not in table_names(conn):
        return []
    return conn.execute(
        "select name, totalMemory, memoryBandwidth, smCount, computeMajor, computeMinor, smMajor, smMinor "
        "from TARGET_INFO_GPU order by id"
    ).fetchall()


def capture_health(conn: sqlite3.Connection) -> list[str]:
    tables = set(table_names(conn))
    messages: list[str] = []

    has_cuda_kernel = any("KERNEL" in table.upper() and not table.upper().startswith("ENUM_") for table in tables)
    has_cuda_memcpy = any("MEMCPY" in table.upper() and not table.upper().startswith("ENUM_") for table in tables)
    has_nvtx = any("NVTX" in table.upper() and not table.upper().startswith("ENUM_") for table in tables)
    has_edm = bool(find_edm_strings(conn, limit=1))

    if not has_cuda_kernel:
        messages.append("missing CUDA kernel activity table")
    if not has_cuda_memcpy:
        messages.append("missing CUDA memcpy activity table")
    if not has_nvtx:
        messages.append("missing NVTX event table")
    if not has_edm:
        messages.append("missing edm.* NVTX strings")
    if messages:
        messages.append(
            "likely causes: nsys did not wrap the Python process, --trace did not include cuda/nvtx, "
            "the exported report came from the wrong run, or CUDA/NVTX capture failed inside the container"
        )
    return messages


def _first_existing(columns: tuple[str, ...], names: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def print_summary(path: str | Path, *, limit: int) -> None:
    with connect(path) as conn:
        print("== capture health ==")
        health = capture_health(conn)
        if not health:
            print("ok: CUDA/NVTX-looking activity is present")
        for message in health:
            print(f"! {message}")

        print("\n== target GPUs ==")
        gpus = target_gpus(conn)
        if not gpus:
            print("<none>")
        for gpu in gpus:
            cc = f"{gpu['computeMajor']}.{gpu['computeMinor']}"
            sm = f"{gpu['smMajor']}.{gpu['smMinor']}"
            print(
                f"name={gpu['name']} sm_count={gpu['smCount']} "
                f"compute_capability={cc} sm={sm} "
                f"total_memory={gpu['totalMemory']} memory_bandwidth={gpu['memoryBandwidth']}"
            )

        print("\n== diagnostic events ==")
        diagnostics = diagnostic_events(conn)
        if not diagnostics:
            print("<none>")
        for event in diagnostics:
            print(f"{event['timestamp']} source={event['source']} severity={event['severity']} text={event['text']}")

        print("\n== sqlite tables ==")
        for info in inspect_tables(conn):
            columns = ", ".join(info.columns)
            print(f"{info.name:<44} rows={info.rows:<8} columns={columns}")

        print("\n== edm NVTX strings ==")
        hits = find_edm_strings(conn, limit=limit)
        if not hits:
            print("<none>")
        for table, text in hits:
            print(f"{table}: {text}")

        print("\n== likely CUDA/NVTX event tables ==")
        for info in likely_event_tables(conn):
            columns = ", ".join(info.columns)
            print(f"{info.name:<44} rows={info.rows:<8} columns={columns}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Nsight Systems sqlite exports for EDM-IR ranges.")
    parser.add_argument("sqlite", help="path to an nsys-exported sqlite database")
    parser.add_argument("--limit", type=int, default=200, help="maximum EDM NVTX strings to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_summary(args.sqlite, limit=args.limit)


if __name__ == "__main__":
    main()
