from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

from .model import A2AGemmProblem, CalibrationProfile, ExecutionOverhead, Program
from .schedule import build_pipelined_program, build_sequential_program
from .simulate import ExecutionFeatures, simulate_program


FEATURE_NAMES = (
    "iteration_us",
    "op_dispatch_us",
    "event_record_us",
    "event_wait_us",
    "buffer_reuse_us",
)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    split: str
    world_size: int
    rows_per_peer: int
    k: int
    n: int
    dtype: str
    strategy: str
    chunk_rows: int
    buffer_count: int
    measured_us: float

    @property
    def problem_key(self) -> tuple[int, int, int, int, str]:
        return (self.world_size, self.rows_per_peer, self.k, self.n, self.dtype)

    def program(self) -> Program:
        problem = A2AGemmProblem(
            world_size=self.world_size,
            rows_per_peer=self.rows_per_peer,
            k=self.k,
            n=self.n,
            dtype=self.dtype,
        )
        problem.validate()
        if self.strategy == "sequential":
            return build_sequential_program(problem)
        if self.strategy == "pipelined":
            return build_pipelined_program(
                problem,
                chunk_rows=self.chunk_rows,
                buffer_count=self.buffer_count,
            )
        raise ValueError(f"unknown measured strategy: {self.strategy}")


def read_runtime_measurements(path: str | Path) -> tuple[RuntimeMeasurement, ...]:
    rows: list[RuntimeMeasurement] = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        for line, value in enumerate(csv.DictReader(source), start=2):
            try:
                measurement = RuntimeMeasurement(
                    split=str(value["split"]),
                    world_size=int(value["world_size"]),
                    rows_per_peer=int(value["rows_per_peer"]),
                    k=int(value["k"]),
                    n=int(value["n"]),
                    dtype=str(value["dtype"]),
                    strategy=str(value["strategy"]),
                    chunk_rows=int(value["chunk_rows"]),
                    buffer_count=int(value["buffer_count"]),
                    measured_us=float(value["measured_us"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid runtime measurement at line {line}: {error}") from error
            if measurement.split not in {"train", "heldout"}:
                raise ValueError(f"measurement line {line} split must be train or heldout")
            if measurement.measured_us <= 0:
                raise ValueError(f"measurement line {line} latency must be positive")
            measurement.program()
            rows.append(measurement)
    if not rows:
        raise ValueError("runtime measurement file is empty")
    return tuple(rows)


def _feature_vector(features: ExecutionFeatures) -> tuple[float, ...]:
    return (
        1.0,
        float(features.op_dispatch_count),
        float(features.event_record_count),
        float(features.event_wait_count),
        float(features.buffer_reuse_count),
    )


def _nonnegative_coordinate_descent(
    matrix: list[tuple[float, ...]],
    targets: list[float],
    *,
    ridge: float,
    iterations: int = 20_000,
    tolerance: float = 1e-12,
) -> tuple[float, ...]:
    """Small dependency-free NNLS solver for the five runtime features."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    columns = len(matrix[0])
    coefficients = [0.0] * columns
    predictions = [0.0] * len(matrix)
    for _iteration in range(iterations):
        largest_change = 0.0
        for column in range(columns):
            old = coefficients[column]
            numerator = 0.0
            denominator = ridge
            for row, target, prediction in zip(matrix, targets, predictions):
                value = row[column]
                numerator += value * (target - prediction + value * old)
                denominator += value * value
            new = max(0.0, numerator / denominator) if denominator else 0.0
            delta = new - old
            if delta:
                coefficients[column] = new
                for index, row in enumerate(matrix):
                    predictions[index] += row[column] * delta
                largest_change = max(largest_change, abs(delta))
        if largest_change <= tolerance:
            break
    return tuple(coefficients)


def fit_execution_overhead(
    base_profile: CalibrationProfile,
    measurements: tuple[RuntimeMeasurement, ...],
    *,
    ridge: float = 1e-6,
) -> CalibrationProfile:
    """Fit non-negative runtime residuals using training rows only."""

    training = tuple(row for row in measurements if row.split == "train")
    if not training:
        raise ValueError("at least one train measurement is required")
    device_profile = replace(base_profile, execution_overhead=ExecutionOverhead())
    matrix: list[tuple[float, ...]] = []
    targets: list[float] = []
    for row in training:
        simulation = simulate_program(row.program(), device_profile)
        matrix.append(_feature_vector(simulation.features))
        targets.append(max(0.0, row.measured_us - simulation.device_makespan_us))
    coefficients = _nonnegative_coordinate_descent(matrix, targets, ridge=ridge)
    overhead = ExecutionOverhead(**dict(zip(FEATURE_NAMES, coefficients)))
    profile = replace(
        base_profile,
        execution_overhead=overhead,
        source=f"{base_profile.source}; runtime overhead fit from {len(training)} train rows",
    )
    margin = _fit_selection_margin(profile, training)
    profile = replace(profile, selection_min_gain_ratio=margin)
    profile.validate()
    return profile


def _fit_selection_margin(
    profile: CalibrationProfile,
    training: tuple[RuntimeMeasurement, ...],
) -> float:
    """Choose a conservative gain threshold using train labels only."""

    grouped: dict[tuple[int, int, int, int, str], list[RuntimeMeasurement]] = {}
    for row in training:
        grouped.setdefault(row.problem_key, []).append(row)
    labels: list[tuple[float, bool]] = []
    for key, rows in grouped.items():
        sequential = next((row for row in rows if row.strategy == "sequential"), None)
        if sequential is None:
            raise ValueError(f"training problem {key} has no sequential baseline")
        sequential_us = simulate_program(sequential.program(), profile).makespan_us
        for row in rows:
            if row is sequential:
                continue
            predicted_us = simulate_program(row.program(), profile).makespan_us
            gain_ratio = (sequential_us - predicted_us) / sequential_us
            labels.append((gain_ratio, row.measured_us < sequential.measured_us))
    if not labels:
        return 0.0
    thresholds = sorted({0.0, *(max(0.0, gain) for gain, _actual in labels)})
    return max(
        thresholds,
        key=lambda threshold: (
            sum((gain > threshold) == actual for gain, actual in labels),
            -threshold,
        ),
    )


def evaluate_cost_model(
    profile: CalibrationProfile,
    measurements: tuple[RuntimeMeasurement, ...],
) -> dict:
    predictions: list[dict] = []
    grouped: dict[tuple[int, int, int, int, str], list[dict]] = {}
    for row in measurements:
        simulation = simulate_program(row.program(), profile)
        result = {
            **asdict(row),
            "predicted_us": simulation.makespan_us,
            "device_makespan_us": simulation.device_makespan_us,
            "overhead_us": simulation.overhead_us,
            "absolute_error_us": abs(simulation.makespan_us - row.measured_us),
            "absolute_percentage_error": abs(simulation.makespan_us - row.measured_us)
            / row.measured_us,
            "features": asdict(simulation.features),
        }
        predictions.append(result)
        grouped.setdefault(row.problem_key, []).append(result)

    def split_metrics(split: str) -> dict:
        rows = [row for row in predictions if row["split"] == split]
        if not rows:
            return {
                "rows": 0,
                "mape": None,
                "mean_absolute_error_us": None,
                "direction_accuracy": None,
                "direction_comparisons": 0,
                "mean_selection_regret": None,
                "policy_direction_accuracy": None,
            }
        return {
            "rows": len(rows),
            "mape": mean(row["absolute_percentage_error"] for row in rows),
            "mean_absolute_error_us": mean(row["absolute_error_us"] for row in rows),
        }

    direction_correct = 0
    direction_total = 0
    regrets: list[float] = []
    split_directions = {
        "train": {"correct": 0, "total": 0},
        "heldout": {"correct": 0, "total": 0},
    }
    split_policy_directions = {
        "train": {"correct": 0, "total": 0},
        "heldout": {"correct": 0, "total": 0},
    }
    split_regrets: dict[str, list[float]] = {"train": [], "heldout": []}
    group_reports: list[dict] = []
    for key, rows in sorted(grouped.items()):
        sequential = next((row for row in rows if row["strategy"] == "sequential"), None)
        if sequential is None:
            raise ValueError(f"problem {key} has no sequential baseline")
        group_splits = {row["split"] for row in rows}
        if len(group_splits) != 1:
            raise ValueError(f"problem {key} appears in more than one split")
        split = next(iter(group_splits))
        for row in rows:
            if row is sequential:
                continue
            direction_total += 1
            predicted_gain = row["predicted_us"] < sequential["predicted_us"]
            measured_gain = row["measured_us"] < sequential["measured_us"]
            correct = predicted_gain == measured_gain
            direction_correct += correct
            split_directions[split]["correct"] += correct
            split_directions[split]["total"] += 1
            gain_ratio = (
                sequential["predicted_us"] - row["predicted_us"]
            ) / sequential["predicted_us"]
            policy_gain = gain_ratio > profile.selection_min_gain_ratio
            split_policy_directions[split]["correct"] += policy_gain == measured_gain
            split_policy_directions[split]["total"] += 1
        selected = min(rows, key=lambda row: row["predicted_us"])
        if selected is not sequential:
            selected_gain = (
                sequential["predicted_us"] - selected["predicted_us"]
            ) / sequential["predicted_us"]
            if selected_gain <= profile.selection_min_gain_ratio:
                selected = sequential
        oracle = min(rows, key=lambda row: row["measured_us"])
        regret = (selected["measured_us"] - oracle["measured_us"]) / oracle["measured_us"]
        regrets.append(regret)
        split_regrets[split].append(regret)
        group_reports.append(
            {
                "problem": {
                    "world_size": key[0],
                    "rows_per_peer": key[1],
                    "k": key[2],
                    "n": key[3],
                    "dtype": key[4],
                },
                "selected": {
                    field: selected[field]
                    for field in ("strategy", "chunk_rows", "buffer_count", "predicted_us", "measured_us")
                },
                "oracle": {
                    field: oracle[field]
                    for field in ("strategy", "chunk_rows", "buffer_count", "predicted_us", "measured_us")
                },
                "selection_regret": regret,
            }
        )

    metrics = {split: split_metrics(split) for split in ("train", "heldout")}
    for split in metrics:
        direction = split_directions[split]
        policy_direction = split_policy_directions[split]
        metrics[split]["direction_accuracy"] = (
            direction["correct"] / direction["total"] if direction["total"] else None
        )
        metrics[split]["direction_comparisons"] = direction["total"]
        metrics[split]["policy_direction_accuracy"] = (
            policy_direction["correct"] / policy_direction["total"]
            if policy_direction["total"]
            else None
        )
        metrics[split]["mean_selection_regret"] = (
            mean(split_regrets[split]) if split_regrets[split] else None
        )
    return {
        "profile_source": profile.source,
        "overhead": asdict(profile.execution_overhead),
        "selection_min_gain_ratio": profile.selection_min_gain_ratio,
        "train": metrics["train"],
        "heldout": metrics["heldout"],
        "direction_accuracy": direction_correct / direction_total if direction_total else None,
        "direction_comparisons": direction_total,
        "mean_selection_regret": mean(regrets) if regrets else None,
        "groups": group_reports,
        "predictions": predictions,
    }


__all__ = [
    "RuntimeMeasurement",
    "evaluate_cost_model",
    "fit_execution_overhead",
    "read_runtime_measurements",
]
