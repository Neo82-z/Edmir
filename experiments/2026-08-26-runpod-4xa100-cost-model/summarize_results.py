#!/usr/bin/env python3
"""Materialize paper-ready summaries from the archived A100 experiment."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DERIVED = ROOT / "derived"
BENCH_PATTERN = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def read_repeat_medians(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("bench "):
            continue
        fields = dict(BENCH_PATTERN.findall(line))
        values.append(float(fields["median_ms"]) * 1000.0)
    if len(values) != 3:
        raise ValueError(f"expected three independent medians in {path}, got {len(values)}")
    return values


def problem_id(problem: dict) -> str:
    return f"r{problem['rows_per_peer']}_k{problem['k']}_n{problem['n']}"


def main() -> None:
    evaluation = json.loads((RESULTS / "evaluation_v2.json").read_text(encoding="utf-8"))
    profile = json.loads((RESULTS / "calibrated_profile_v2.json").read_text(encoding="utf-8"))
    DERIVED.mkdir(exist_ok=True)

    prediction_fields = (
        "split",
        "problem_id",
        "world_size",
        "rows_per_peer",
        "k",
        "n",
        "strategy",
        "chunk_rows",
        "buffer_count",
        "measured_us",
        "predicted_us",
        "absolute_percentage_error",
    )
    with (DERIVED / "candidate_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        for row in evaluation["predictions"]:
            value = {name: row[name] for name in prediction_fields if name in row}
            value["problem_id"] = f"r{row['rows_per_peer']}_k{row['k']}_n{row['n']}"
            writer.writerow(value)

    predictions_by_problem: dict[str, list[dict]] = {}
    for row in evaluation["predictions"]:
        key = f"r{row['rows_per_peer']}_k{row['k']}_n{row['n']}"
        predictions_by_problem.setdefault(key, []).append(row)

    summaries: list[dict] = []
    repeat_records: list[dict] = []
    for group in evaluation["groups"]:
        problem = group["problem"]
        key = problem_id(problem)
        rows = predictions_by_problem[key]
        sequential = next(row for row in rows if row["strategy"] == "sequential")
        selected = group["selected"]
        split = rows[0]["split"]
        value = {
            "split": split,
            "problem_id": key,
            "rows_per_peer": problem["rows_per_peer"],
            "k": problem["k"],
            "n": problem["n"],
            "candidate_count": len(rows),
            "sequential_us": sequential["measured_us"],
            "selected_strategy": selected["strategy"],
            "selected_chunk_rows": selected["chunk_rows"],
            "selected_buffer_count": selected["buffer_count"],
            "selected_us": selected["measured_us"],
            "sweep_speedup": sequential["measured_us"] / selected["measured_us"],
            "sweep_reduction_ratio": (
                sequential["measured_us"] - selected["measured_us"]
            )
            / sequential["measured_us"],
            "selection_regret": group["selection_regret"],
            "repeat_sequential_us": "",
            "repeat_selected_us": "",
            "repeat_speedup": "",
            "repeat_reduction_ratio": "",
        }
        if split == "heldout":
            seq_path = RESULTS / f"repeat_heldout_{key}__sequential.txt"
            selected_path = RESULTS / (
                f"repeat_heldout_{key}__pipelined_"
                f"c{selected['chunk_rows']}_b{selected['buffer_count']}.txt"
            )
            seq_trials = read_repeat_medians(seq_path)
            selected_trials = read_repeat_medians(selected_path)
            seq_us = median(seq_trials)
            selected_us = median(selected_trials)
            for trial, (seq_trial, selected_trial) in enumerate(
                zip(seq_trials, selected_trials), start=1
            ):
                repeat_records.append(
                    {
                        "problem_id": key,
                        "trial": trial,
                        "sequential_us": seq_trial,
                        "selected_us": selected_trial,
                        "reduction_ratio": (seq_trial - selected_trial) / seq_trial,
                    }
                )
            value.update(
                repeat_sequential_us=seq_us,
                repeat_selected_us=selected_us,
                repeat_speedup=seq_us / selected_us,
                repeat_reduction_ratio=(seq_us - selected_us) / seq_us,
            )
        summaries.append(value)

    summary_fields = tuple(summaries[0])
    with (DERIVED / "problem_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    with (DERIVED / "repeat_process_medians.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = ("problem_id", "trial", "sequential_us", "selected_us", "reduction_ratio")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repeat_records)

    summary = {
        "hardware": "4x NVIDIA A100-SXM4-80GB, uniform NV12 topology",
        "candidate_rows": len(evaluation["predictions"]),
        "train": evaluation["train"],
        "heldout": evaluation["heldout"],
        "selection_min_gain_ratio": evaluation["selection_min_gain_ratio"],
        "execution_overhead": profile["execution_overhead"],
        "problems": summaries,
        "scope": (
            "single node, single topology, NCCL+cuBLAS backend; held-out shapes, "
            "not held-out hardware or backend"
        ),
    }
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
