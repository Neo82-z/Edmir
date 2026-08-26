from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
DERIVED = ROOT / "derived"
MEDIAN = re.compile(r"median_ms=(?P<median>[0-9.]+)")


def benchmark_medians(spec: str, mode: str) -> list[float]:
    values = []
    for path in sorted(LOGS.glob(f"40_{spec}_{mode}_process*.txt")):
        match = MEDIAN.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"cannot parse {path}")
        values.append(float(match["median"]))
    return values


def mapped_counts(spec: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    path = RESULTS / spec / "edm_cuda_events.csv"
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not row["iteration"] or not row["edm_op"]:
                continue
            iteration = int(row["iteration"])
            counts[iteration] = counts.get(iteration, 0) + 1
    return counts


def main() -> None:
    rows = []
    traces = {}
    for spec in ("r64", "r96"):
        plain = benchmark_medians(spec, "plain")
        nvtx = benchmark_medians(spec, "nvtx")
        plain_median = statistics.median(plain)
        nvtx_median = statistics.median(nvtx)
        rows.append(
            {
                "case": spec,
                "plain_process_median_ms": plain_median,
                "nvtx_process_median_ms": nvtx_median,
                "nvtx_delta_percent": 100.0 * (nvtx_median - plain_median) / plain_median,
            }
        )
        validation = json.loads(
            (RESULTS / spec / f"{spec}-trace-validation.json").read_text(encoding="utf-8")
        )
        traces[spec] = {**validation, "mapped_events_by_iteration": mapped_counts(spec)}

    DERIVED.mkdir(exist_ok=True)
    with (DERIVED / "nvtx_overhead.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {
            "gpus": 4,
            "warmup_iterations": 3,
            "selected_trace_iteration": 3,
            "nvtx_overhead_processes": 3,
            "nvtx_overhead_warmups": 10,
            "nvtx_overhead_iterations": 100,
        },
        "nvtx_overhead": rows,
        "trace_validation": traces,
        "interpretation": {
            "dependency_timing": "valid ordering evidence",
            "trace_overlap_timing": "profiler-perturbed; not a performance baseline",
        },
    }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
