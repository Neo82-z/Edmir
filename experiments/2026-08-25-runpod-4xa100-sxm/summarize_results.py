from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DERIVED = ROOT / "derived"

COLLECTIVE_PATTERN = re.compile(
    r"30_(?P<tag>.+)_(?P<op>all_reduce|all_gather|alltoall)\.txt$"
)
PAIR_PATTERN = re.compile(r"pair_(?P<src>\d+)_(?P<dst>\d+)_default_sendrecv\.txt$")
KEY_VALUE_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")


def parse_nccl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            rows.append(
                {
                    "size_bytes": int(fields[0]),
                    "time_us": float(fields[5]),
                    "algbw_gbps": float(fields[6]),
                    "busbw_gbps": float(fields[7]),
                    "wrong": fields[8],
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no NCCL rows in {path}")
    return rows


def write_nccl_curves() -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    for path in sorted(RESULTS.glob("30_*.txt")):
        match = COLLECTIVE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        for row in parse_nccl(path):
            combined.append(
                {
                    "suite": "collective",
                    "tag": match.group("tag"),
                    "operation": match.group("op"),
                    "src_rank": "",
                    "dst_rank": "",
                    **row,
                }
            )
    for path in sorted(RESULTS.glob("pair_*_default_sendrecv.txt")):
        match = PAIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        for row in parse_nccl(path):
            combined.append(
                {
                    "suite": "pairwise",
                    "tag": f"pair{match.group('src')}{match.group('dst')}",
                    "operation": "sendrecv",
                    "src_rank": int(match.group("src")),
                    "dst_rank": int(match.group("dst")),
                    **row,
                }
            )
    path = DERIVED / "nccl_curves.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return combined


def parse_benchmarks() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(RESULTS.glob("*_benchmark*.txt")):
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "bench target=" not in raw:
                continue
            values = dict(KEY_VALUE_PATTERN.findall(raw))
            record: dict[str, object] = {"source": path.name, **values}
            for key in ("ranks", "chunk_rows", "buffer_count", "ops", "warmup", "iterations"):
                if key in record:
                    record[key] = int(str(record[key]))
            for key in ("min_ms", "median_ms", "p95_ms", "max_ms", "mean_ms"):
                if key in record:
                    record[key] = float(str(record[key]))
            records.append(record)
    path = DERIVED / "benchmark_summary.csv"
    columns = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    DERIVED.mkdir(exist_ok=True)
    curves = write_nccl_curves()
    benchmarks = parse_benchmarks()
    topology = json.loads((RESULTS / "a100_sxm_topology_fixed.json").read_text())
    selected = json.loads((RESULTS / "71_selected_program.json").read_text())
    max_rows = [row for row in curves if row["size_bytes"] == max(
        item["size_bytes"] for item in curves
        if item["suite"] == row["suite"] and item["tag"] == row["tag"]
        and item["operation"] == row["operation"]
    )]
    summary = {
        "world_size": topology["world_size"],
        "devices": topology["devices"],
        "full_p2p": all(
            src == dst or topology["p2p"][src][dst]
            for src in range(topology["world_size"])
            for dst in range(topology["world_size"])
        ),
        "link_labels": topology["link_labels"],
        "selected": {
            "strategy": selected["strategy"],
            "chunk_rows": selected["chunk_rows"],
            "buffer_count": selected["buffer_count"],
            "ops": len(selected["ops"]),
        },
        "nccl_rows": len(curves),
        "benchmark_rows": len(benchmarks),
        "largest_message_rows": max_rows,
    }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
