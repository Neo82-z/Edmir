from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FUSED_FAMILIES = ("host", "direct", "splitk8")


@dataclass(frozen=True, slots=True)
class FusedMeasurement:
    family: str
    rows_per_peer: int
    k: int
    n: int
    measured_ms: float
    remote_input_bytes: int
    remote_a_replay: int
    split_k_slices: int
    estimated_ctas_per_op: int

    def validate(self) -> None:
        if self.family not in FUSED_FAMILIES:
            raise ValueError(f"unsupported fused family {self.family!r}")
        for name in ("rows_per_peer", "k", "n", "remote_input_bytes", "remote_a_replay", "split_k_slices"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.measured_ms) or self.measured_ms <= 0:
            raise ValueError("measurement latency must be finite and positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.rows_per_peer, self.k, self.n)


def read_fused_measurements(paths: Sequence[str | Path]) -> tuple[FusedMeasurement, ...]:
    rows: list[FusedMeasurement] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                family = row["family"]
                if family not in FUSED_FAMILIES:
                    raise ValueError(f"unsupported fused family {family!r} in {path}")
                rows.append(
                    FusedMeasurement(
                        family=family,
                        rows_per_peer=int(row["rows_per_peer"]),
                        k=int(row["k"]),
                        n=int(row["n"]),
                        measured_ms=float(row["median_ms"]),
                        remote_input_bytes=int(row["remote_input_bytes"]),
                        remote_a_replay=int(row["remote_a_replay"]),
                        split_k_slices=int(row["split_k_slices"]),
                        estimated_ctas_per_op=int(row["estimated_ctas_per_op"]),
                    )
                )
    _group_by_shape(rows)
    return tuple(rows)


def _group_by_shape(
    measurements: Iterable[FusedMeasurement],
) -> dict[tuple[int, int, int], dict[str, FusedMeasurement]]:
    groups: dict[tuple[int, int, int], dict[str, FusedMeasurement]] = {}
    for row in measurements:
        row.validate()
        families = groups.setdefault(row.shape, {})
        if row.family in families:
            raise ValueError(f"duplicate {row.family} measurement for shape {row.shape}")
        families[row.family] = row
    for shape, families in groups.items():
        missing = set(FUSED_FAMILIES) - set(families)
        if missing:
            raise ValueError(f"shape {shape} is missing families {sorted(missing)}")
    if not groups:
        raise ValueError("fused measurement set must not be empty")
    return groups


def _selected_family(k: int, max_direct_k: int) -> str:
    return "direct" if k <= max_direct_k else "splitk8"


def evaluate_fused_k_boundary(
    measurements: Iterable[FusedMeasurement], max_direct_k: int
) -> dict:
    groups = _group_by_shape(measurements)
    correct = 0
    regrets: list[float] = []
    host_wins = 0
    decisions: list[dict] = []
    for shape, families in sorted(groups.items()):
        winner = min(families.values(), key=lambda row: row.measured_ms)
        selected = families[_selected_family(shape[1], max_direct_k)]
        best_ms = winner.measured_ms
        regret = (selected.measured_ms - best_ms) / best_ms
        correct += selected.family == winner.family
        host_wins += winner.family == "host"
        regrets.append(regret)
        decisions.append(
            {
                "rows_per_peer": shape[0],
                "k": shape[1],
                "n": shape[2],
                "winner": winner.family,
                "selected": selected.family,
                "best_ms": best_ms,
                "selected_ms": selected.measured_ms,
                "regret": regret,
            }
        )
    return {
        "problems": len(groups),
        "direction_accuracy": correct / len(groups),
        "mean_selection_regret": sum(regrets) / len(regrets),
        "max_selection_regret": max(regrets),
        "host_wins": host_wins,
        "decisions": decisions,
    }


def fit_fused_k_boundary(measurements: Iterable[FusedMeasurement]) -> int:
    rows = tuple(measurements)
    groups = _group_by_shape(rows)
    candidates = sorted({shape[1] for shape in groups})
    scored: list[tuple[int, float, int]] = []
    for threshold in candidates:
        report = evaluate_fused_k_boundary(rows, threshold)
        errors = report["problems"] - round(
            report["direction_accuracy"] * report["problems"]
        )
        scored.append((errors, report["mean_selection_regret"], threshold))
    return min(scored)[2]


def fit_and_evaluate_fused_boundary(
    train: Iterable[FusedMeasurement], heldout: Iterable[FusedMeasurement]
) -> dict:
    train_rows = tuple(train)
    heldout_rows = tuple(heldout)
    if {row.shape for row in train_rows} & {row.shape for row in heldout_rows}:
        raise ValueError("training and held-out workloads overlap")
    max_direct_k = fit_fused_k_boundary(train_rows)
    split_values = sorted({row.k for row in train_rows if row.k > max_direct_k})
    return {
        "schema": "edmir.fused-boundary-evaluation.v1",
        "rule": {
            "feature": "k",
            "max_direct_k": max_direct_k,
            "min_split_k": split_values[0] if split_values else None,
            "lower_family": "direct",
            "upper_family": "splitk8",
            "default_compiler_admission": False,
        },
        "train": evaluate_fused_k_boundary(train_rows, max_direct_k),
        "heldout": evaluate_fused_k_boundary(heldout_rows, max_direct_k),
    }


__all__ = [
    "FUSED_FAMILIES",
    "FusedMeasurement",
    "evaluate_fused_k_boundary",
    "fit_and_evaluate_fused_boundary",
    "fit_fused_k_boundary",
    "read_fused_measurements",
]
