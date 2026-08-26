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
    r"gemm_provider=(?P<provider>\w+).*?rows_per_peer=(?P<rows>\d+) "
    r"k=(?P<k>\d+) n=(?P<n>\d+).*?median_ms=(?P<median>[0-9.]+) "
    r"p95_ms=(?P<p95>[0-9.]+)"
)
PROCESS = re.compile(r"process(?P<process>\d+)\.txt$")
VALIDATION = re.compile(
    r"gemm_provider=(?P<provider>\w+).*?checked_elements=(?P<elements>\d+) "
    r"max_abs_error=(?P<absolute>[0-9.eE+-]+) "
    r"max_rel_error=(?P<relative>[0-9.eE+-]+)"
)


def benchmark_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(LOGS.glob("*_process*.txt")):
        match = BENCHMARK.search(path.read_text(encoding="utf-8"))
        process = PROCESS.search(path.name)
        if match is None or process is None:
            continue
        rows.append(
            {
                "shape": f"r{match['rows']}_k{match['k']}_n{match['n']}",
                "provider": match["provider"],
                "process": int(process["process"]),
                "median_ms": float(match["median"]),
                "p95_ms": float(match["p95"]),
            }
        )
    return rows


def validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(LOGS.glob("*numerical_validation*.txt")):
        match = VALIDATION.search(path.read_text(encoding="utf-8"))
        if match is None:
            continue
        rows.append(
            {
                "case": path.stem,
                "provider": match["provider"],
                "checked_elements": int(match["elements"]),
                "max_abs_error": float(match["absolute"]),
                "max_rel_error": float(match["relative"]),
            }
        )
    return rows


def main() -> None:
    DERIVED.mkdir(exist_ok=True)
    rows = benchmark_rows()
    with (DERIVED / "process_medians.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_shape: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        by_shape.setdefault(str(row["shape"]), {}).setdefault(
            str(row["provider"]), []
        ).append(float(row["median_ms"]))
    summary: dict[str, object] = {
        "protocol": {
            "processes_per_provider": 5,
            "warmup_iterations": 10,
            "measured_iterations": 50,
            "process_order": "alternating",
        },
        "shapes": {},
        "validation": validation_rows(),
    }
    for shape, providers in sorted(by_shape.items()):
        cublas = statistics.median(providers["cublas"])
        cutlass = statistics.median(providers["cutlass"])
        summary["shapes"][shape] = {
            "cublas_process_median_ms": cublas,
            "cutlass_process_median_ms": cutlass,
            "cutlass_latency_reduction_percent": 100.0 * (cublas - cutlass) / cublas,
        }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
