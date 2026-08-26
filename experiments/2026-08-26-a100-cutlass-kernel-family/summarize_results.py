from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
DERIVED = ROOT / "derived"
BENCHMARK = re.compile(
    r"gemm_provider=(?P<provider>\w+) gemm_kernel=(?P<kernel>\w+) .*?"
    r"rows_per_peer=(?P<rows>\d+) k=(?P<k>\d+) n=(?P<n>\d+).*?"
    r"median_ms=(?P<median>[0-9.]+) p95_ms=(?P<p95>[0-9.]+)"
)
PROCESS = re.compile(r"process(?P<process>\d+)\.txt$")
VALIDATION = re.compile(
    r"gemm_kernel=(?P<kernel>\w+).*?checked_elements=(?P<elements>\d+) "
    r"max_abs_error=(?P<absolute>[0-9.eE+-]+) "
    r"max_rel_error=(?P<relative>[0-9.eE+-]+)"
)


def parse(pattern: str, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(LOGS.glob(pattern)):
        match = BENCHMARK.search(path.read_text(encoding="utf-8"))
        process = PROCESS.search(path.name)
        if match is None or process is None:
            continue
        rows.append(
            {
                "experiment": label,
                "shape": f"r{match['rows']}_k{match['k']}_n{match['n']}",
                "provider": match["provider"],
                "kernel": match["kernel"],
                "process": int(process["process"]),
                "median_ms": float(match["median"]),
                "p95_ms": float(match["p95"]),
            }
        )
    return rows


def validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(LOGS.glob("*validate*.txt")):
        match = VALIDATION.search(path.read_text(encoding="utf-8"))
        if match is None:
            continue
        rows.append(
            {
                "case": path.stem,
                "kernel": match["kernel"],
                "checked_elements": int(match["elements"]),
                "max_abs_error": float(match["absolute"]),
                "max_rel_error": float(match["relative"]),
            }
        )
    return rows


def main() -> None:
    rows = parse("40_*_process*.txt", "kernel_sweep")
    rows += parse("70_*_process*.txt", "matched_provider")
    DERIVED.mkdir(exist_ok=True)
    with (DERIVED / "process_medians.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["experiment"] != "matched_provider":
            continue
        label = "cublas" if row["provider"] == "cublas" else "cutlass_auto"
        final.setdefault(str(row["shape"]), {}).setdefault(label, []).append(
            float(row["median_ms"])
        )
    shapes: dict[str, object] = {}
    for shape, providers in sorted(final.items()):
        cublas = statistics.median(providers["cublas"])
        auto = statistics.median(providers["cutlass_auto"])
        shapes[shape] = {
            "cublas_process_median_ms": cublas,
            "cutlass_auto_process_median_ms": auto,
            "latency_reduction_percent": 100.0 * (cublas - auto) / cublas,
        }
    summary = {
        "protocol": {
            "kernel_sweep_processes": 3,
            "matched_provider_processes": 5,
            "warmup_iterations": 10,
            "measured_iterations": 100,
        },
        "auto_policy": {
            "chunk_rows_le_96": "sm80_64x128x32",
            "otherwise": "sm80_128x128x32",
        },
        "shapes": shapes,
        "validation": validation_rows(),
    }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
