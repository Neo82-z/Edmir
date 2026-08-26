from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
DERIVED = ROOT / "derived"
FINAL_BENCHMARK = re.compile(
    r"rows_per_peer=(?P<rows>\d+) k=(?P<k>\d+) n=(?P<n>\d+).*?"
    r"chunk_rows=(?P<chunk>\d+) buffer_count=(?P<buffers>\d+).*?"
    r"median_ms=(?P<median>[0-9.]+) p95_ms=(?P<p95>[0-9.]+)"
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gemm_comparison() -> list[dict[str, object]]:
    providers = {
        "cublas": read_jsonl(RESULTS / "cublas_gemm_measurements.jsonl"),
        "cutlass_auto": read_jsonl(RESULTS / "isolated_gemm_measurements.jsonl"),
    }
    grouped: dict[tuple[int, int, int], dict[str, list[float]]] = {}
    for provider, rows in providers.items():
        for row in rows:
            shape = (int(row["m"]), int(row["k"]), int(row["n"]))
            grouped.setdefault(shape, {}).setdefault(provider, []).append(float(row["median_us"]))
    output: list[dict[str, object]] = []
    for (m, k, n), values in sorted(grouped.items()):
        cublas = statistics.median(values["cublas"])
        cutlass = statistics.median(values["cutlass_auto"])
        output.append(
            {
                "m": m,
                "k": k,
                "n": n,
                "cublas_median_us": cublas,
                "cutlass_auto_median_us": cutlass,
                "cutlass_over_cublas": cutlass / cublas,
            }
        )
    return output


def final_validation() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted(LOGS.glob("92_final_*.txt")):
        match = FINAL_BENCHMARK.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"cannot parse {path}")
        output.append(
            {
                "case": path.stem.removeprefix("92_final_"),
                "rows_per_peer": int(match["rows"]),
                "k": int(match["k"]),
                "n": int(match["n"]),
                "chunk_rows": int(match["chunk"]),
                "buffer_count": int(match["buffers"]),
                "median_us": 1000.0 * float(match["median"]),
                "p95_us": 1000.0 * float(match["p95"]),
            }
        )
    return output


def main() -> None:
    evaluation = json.loads((RESULTS / "evaluation.json").read_text(encoding="utf-8"))
    comparison = gemm_comparison()
    selections = []
    for group in evaluation["groups"]:
        problem = group["problem"]
        selected = group["selected"]
        selections.append(
            {
                **problem,
                **selected,
                "selection_regret": group["selection_regret"],
                "is_oracle": selected == group["oracle"],
            }
        )

    DERIVED.mkdir(exist_ok=True)
    for name, rows in (
        ("gemm_provider_comparison.csv", comparison),
        ("selection_summary.csv", selections),
    ):
        with (DERIVED / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "protocol": {
            "provider": "cutlass",
            "kernel_policy": "auto",
            "runtime_candidates": len(evaluation["predictions"]),
            "train_candidates": evaluation["train"]["rows"],
            "heldout_candidates": evaluation["heldout"]["rows"],
            "warmup_iterations": 5,
            "measured_iterations": 30,
            "final_validation_iterations": 100,
        },
        "metrics": {
            "train_mape": evaluation["train"]["mape"],
            "heldout_mape": evaluation["heldout"]["mape"],
            "overall_direction_accuracy": evaluation["direction_accuracy"],
            "heldout_direction_accuracy": evaluation["heldout"]["direction_accuracy"],
            "mean_selection_regret": evaluation["mean_selection_regret"],
            "oracle_selections": sum(bool(row["is_oracle"]) for row in selections),
            "selection_groups": len(selections),
        },
        "execution_overhead_us": evaluation["overhead"],
        "selections": selections,
        "isolated_gemm": comparison,
        "final_validation": final_validation(),
    }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
