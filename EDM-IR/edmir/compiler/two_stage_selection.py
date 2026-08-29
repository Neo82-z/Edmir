from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_TWO_STAGE_EVALUATION = "edmir.two-stage-evaluation.v1"
DEFAULT_SM_COUNT = 108
DEFAULT_THREADBLOCK_M = 64
DEFAULT_THREADBLOCK_N = 128


@dataclass(frozen=True, slots=True)
class ScheduleMeasurement:
    """One measured host or fused schedule candidate.

    ``group_id`` is the experimental unit used for held-out evaluation.  It is
    deliberately not inferred from M/N inside the evaluator: callers must
    state which sweep was collected together so that random row splitting
    cannot leak adjacent candidates into train and test sets.
    """

    group_id: str
    family: str
    world_size: int
    rows_per_peer: int
    k: int
    n: int
    measured_ms: float
    remote_input_bytes: int
    remote_a_replay: int
    split_k_slices: int
    n_tiles_per_cta: int
    estimated_ctas_per_op: int

    def validate(self) -> None:
        if not self.group_id:
            raise ValueError("measurement group_id must not be empty")
        if self.world_size < 2:
            raise ValueError("measurement world_size must be at least two")
        if min(self.rows_per_peer, self.k, self.n) <= 0:
            raise ValueError("measurement dimensions must be positive")
        if self.measured_ms <= 0.0:
            raise ValueError("measurement latency must be positive")
        if self.remote_input_bytes <= 0 or self.remote_a_replay <= 0:
            raise ValueError("measurement traffic terms must be positive")
        if self.split_k_slices <= 0 or self.n_tiles_per_cta <= 0:
            raise ValueError("measurement schedule axes must be positive")
        if self.estimated_ctas_per_op < 0:
            raise ValueError("estimated CTA count must be non-negative")
        expected_q, expected_g = parse_schedule_family(self.family)
        if self.family == "host":
            if self.estimated_ctas_per_op != 0:
                raise ValueError("host measurements must report zero fused CTAs")
        elif self.estimated_ctas_per_op == 0:
            raise ValueError("fused measurements must report a positive CTA count")
        if (expected_q, expected_g) != (
            self.split_k_slices,
            self.n_tiles_per_cta,
        ):
            raise ValueError(
                f"family {self.family!r} does not match q/g fields "
                f"({self.split_k_slices}, {self.n_tiles_per_cta})"
            )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.world_size, self.rows_per_peer, self.k, self.n)

    @property
    def grouped_shape(self) -> tuple[str, int, int, int, int]:
        return (self.group_id, *self.shape)

    @property
    def is_host(self) -> bool:
        return self.family == "host"


def parse_schedule_family(family: str) -> tuple[int, int]:
    """Return ``(split_k_slices, n_tiles_per_cta)`` for a runtime family."""

    grouped = 1
    base = family
    if "_groupedn" in base:
        base, raw_grouped = base.rsplit("_groupedn", 1)
        try:
            grouped = int(raw_grouped)
        except ValueError as error:
            raise ValueError(f"invalid grouped-N family {family!r}") from error
    if base in {"host", "direct"}:
        split_k = 1
    elif base.startswith("splitk"):
        try:
            split_k = int(base.removeprefix("splitk"))
        except ValueError as error:
            raise ValueError(f"invalid split-K family {family!r}") from error
    else:
        raise ValueError(f"unsupported schedule family {family!r}")
    if min(split_k, grouped) <= 0:
        raise ValueError(f"schedule axes in {family!r} must be positive")
    if split_k not in {1, 2, 4, 8, 16}:
        raise ValueError(f"unsupported split-K slice count in {family!r}")
    if grouped not in {1, 2}:
        raise ValueError(f"unsupported executable grouped-N factor in {family!r}")
    if base == "host" and grouped != 1:
        raise ValueError("host family cannot use grouped-N")
    return split_k, grouped


def read_schedule_measurements(
    paths: Sequence[str | Path], *, world_size: int
) -> tuple[ScheduleMeasurement, ...]:
    """Read complete or partial candidate surfaces from benchmark CSV files."""

    rows: list[ScheduleMeasurement] = []
    for raw_path in paths:
        path = Path(raw_path)
        group_id = path.parent.name
        with path.open(newline="", encoding="utf-8") as source:
            for raw in csv.DictReader(source):
                row = ScheduleMeasurement(
                    group_id=group_id,
                    family=raw["family"],
                    world_size=world_size,
                    rows_per_peer=int(raw["rows_per_peer"]),
                    k=int(raw["k"]),
                    n=int(raw["n"]),
                    measured_ms=float(raw["median_ms"]),
                    remote_input_bytes=int(raw["remote_input_bytes"]),
                    remote_a_replay=int(raw["remote_a_replay"]),
                    split_k_slices=int(raw["split_k_slices"]),
                    n_tiles_per_cta=int(raw.get("n_tiles_per_cta", "1")),
                    estimated_ctas_per_op=int(raw["estimated_ctas_per_op"]),
                )
                row.validate()
                rows.append(row)
    _group_measurements(rows)
    return tuple(rows)


def _group_measurements(
    measurements: Iterable[ScheduleMeasurement],
) -> dict[tuple[str, int, int, int, int], dict[str, ScheduleMeasurement]]:
    groups: dict[
        tuple[str, int, int, int, int], dict[str, ScheduleMeasurement]
    ] = {}
    for row in measurements:
        row.validate()
        candidates = groups.setdefault(row.grouped_shape, {})
        if row.family in candidates:
            raise ValueError(
                f"duplicate {row.family} measurement for {row.grouped_shape}"
            )
        candidates[row.family] = row
    if not groups:
        raise ValueError("schedule measurement set must not be empty")
    for shape, candidates in groups.items():
        if "host" not in candidates:
            raise ValueError(f"shape {shape} has no host baseline")
        if len(candidates) < 2:
            raise ValueError(f"shape {shape} has no fused candidate")
    return groups


@dataclass(frozen=True, slots=True)
class AnalyticalScheduleFeatures:
    flops: int
    unique_remote_bytes: int
    local_bytes: int
    rho_n: int
    gamma: int
    split_k_slices: int
    n_tiles_per_cta: int
    split_k_depth: float
    reduction_bytes: int
    underfill: float

    @classmethod
    def from_measurement(
        cls, row: ScheduleMeasurement, *, sm_count: int = DEFAULT_SM_COUNT
    ) -> "AnalyticalScheduleFeatures":
        if sm_count <= 0:
            raise ValueError("SM count must be positive")
        flops = 2 * row.world_size * row.rows_per_peer * row.k * row.n
        # A, B and C traffic. This is a compiler-visible local-memory proxy,
        # not a claim that every byte reaches HBM exactly once.
        local_bytes = 2 * row.world_size * (
            row.rows_per_peer * row.k
            + row.k * row.n
            + row.rows_per_peer * row.n
        )
        if row.is_host:
            gamma = (
                math.ceil(row.world_size * row.rows_per_peer / DEFAULT_THREADBLOCK_M)
                * math.ceil(row.n / DEFAULT_THREADBLOCK_N)
            )
            rho_n = 1
        else:
            gamma = row.estimated_ctas_per_op
            rho_n = row.remote_a_replay
        q = row.split_k_slices
        reduction_bytes = (
            4 * row.world_size * row.rows_per_peer * row.n * (q - 1)
            if q > 1
            else 0
        )
        return cls(
            flops=flops,
            unique_remote_bytes=row.remote_input_bytes,
            local_bytes=local_bytes,
            rho_n=rho_n,
            gamma=gamma,
            split_k_slices=q,
            n_tiles_per_cta=row.n_tiles_per_cta,
            split_k_depth=row.k / q,
            reduction_bytes=reduction_bytes,
            underfill=max(0.0, (sm_count - gamma) / sm_count),
        )


_HOST_FEATURES = (
    "log_flops",
    "log_unique_remote_bytes",
    "log_local_bytes",
)
_FULL_FUSED_FEATURES = (
    "log_flops",
    "log_unique_remote_bytes",
    "log_local_bytes",
    "log_split_k_depth",
    "log_q",
    "grouped",
    "log_rho_n",
    "log_gamma",
    "underfill",
    "log_reduction_bytes",
    "q_minus_one",
)


def _feature_names(stage: str, ablation: str) -> tuple[str, ...]:
    if ablation not in {"full", "no_gamma", "no_rho", "no_reduction", "k_only"}:
        raise ValueError(f"unknown feature ablation {ablation!r}")
    if stage == "host":
        return ("log_k",) if ablation == "k_only" else _HOST_FEATURES
    if stage != "fused":
        raise ValueError(f"unknown selector stage {stage!r}")
    if ablation == "k_only":
        return ("log_k", "log_q", "log_k_x_log_q", "grouped")
    names = list(_FULL_FUSED_FEATURES)
    if ablation == "no_gamma":
        names.remove("log_gamma")
        names.remove("underfill")
    elif ablation == "no_rho":
        names.remove("log_rho_n")
    elif ablation == "no_reduction":
        names.remove("log_reduction_bytes")
        names.remove("q_minus_one")
    return tuple(names)


def _feature_values(
    row: ScheduleMeasurement,
    names: Sequence[str],
    *,
    sm_count: int,
) -> tuple[float, ...]:
    features = AnalyticalScheduleFeatures.from_measurement(row, sm_count=sm_count)
    log_k = math.log2(row.k)
    log_q = math.log2(features.split_k_slices)
    values: Mapping[str, float] = {
        "log_k": log_k,
        "log_flops": math.log2(features.flops),
        "log_unique_remote_bytes": math.log2(features.unique_remote_bytes),
        "log_local_bytes": math.log2(features.local_bytes),
        "log_split_k_depth": math.log2(features.split_k_depth),
        "log_q": log_q,
        "log_k_x_log_q": log_k * log_q,
        "grouped": float(features.n_tiles_per_cta > 1),
        "log_rho_n": math.log2(features.rho_n),
        "log_gamma": math.log2(features.gamma),
        "underfill": features.underfill,
        "log_reduction_bytes": math.log2(features.reduction_bytes + 1),
        "q_minus_one": float(features.split_k_slices - 1),
    }
    return tuple(values[name] for name in names)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [matrix[index][:] + [rhs[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("latency model feature matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


@dataclass(frozen=True, slots=True)
class RidgeLatencyModel:
    stage: str
    ablation: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge: float
    sm_count: int

    @classmethod
    def fit(
        cls,
        measurements: Sequence[ScheduleMeasurement],
        *,
        stage: str,
        ablation: str,
        ridge: float,
        sm_count: int,
    ) -> "RidgeLatencyModel":
        if ridge <= 0.0:
            raise ValueError("ridge regularization must be positive")
        names = _feature_names(stage, ablation)
        rows = [
            row
            for row in measurements
            if (row.is_host if stage == "host" else not row.is_host)
        ]
        if len(rows) <= len(names):
            raise ValueError(
                f"{stage} model needs more than {len(names)} measurements; got {len(rows)}"
            )
        raw = [_feature_values(row, names, sm_count=sm_count) for row in rows]
        means = tuple(sum(row[index] for row in raw) / len(raw) for index in range(len(names)))
        scales = tuple(
            max(
                math.sqrt(
                    sum((row[index] - means[index]) ** 2 for row in raw) / len(raw)
                ),
                1e-9,
            )
            for index in range(len(names))
        )
        design = [
            (
                1.0,
                *((value - mean) / scale for value, mean, scale in zip(row, means, scales)),
            )
            for row in raw
        ]
        target = [math.log(row.measured_ms) for row in rows]
        width = len(names) + 1
        normal = [
            [
                sum(row[left] * row[right] for row in design)
                + (ridge if left == right and left > 0 else 0.0)
                for right in range(width)
            ]
            for left in range(width)
        ]
        projected = [
            sum(row[index] * value for row, value in zip(design, target))
            for index in range(width)
        ]
        coefficients = tuple(_solve_linear_system(normal, projected))
        return cls(
            stage=stage,
            ablation=ablation,
            feature_names=names,
            means=means,
            scales=scales,
            coefficients=coefficients,
            ridge=ridge,
            sm_count=sm_count,
        )

    def predict_ms(self, row: ScheduleMeasurement) -> float:
        values = _feature_values(row, self.feature_names, sm_count=self.sm_count)
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales)
        )
        prediction = self.coefficients[0] + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients[1:], normalized)
        )
        return math.exp(prediction)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ablation": self.ablation,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "ridge": self.ridge,
            "sm_count": self.sm_count,
        }


@dataclass(frozen=True, slots=True)
class TwoStageSelector:
    host_model: RidgeLatencyModel
    fused_model: RidgeLatencyModel
    minimum_fused_advantage: float = 0.0
    force_fused: bool = False

    @classmethod
    def fit(
        cls,
        measurements: Sequence[ScheduleMeasurement],
        *,
        ablation: str = "full",
        ridge: float = 1.0,
        sm_count: int = DEFAULT_SM_COUNT,
        minimum_fused_advantage: float = 0.0,
        force_fused: bool = False,
    ) -> "TwoStageSelector":
        if not 0.0 <= minimum_fused_advantage < 1.0:
            raise ValueError("minimum fused advantage must be in [0, 1)")
        rows = tuple(measurements)
        _group_measurements(rows)
        return cls(
            host_model=RidgeLatencyModel.fit(
                rows,
                stage="host",
                ablation=ablation,
                ridge=ridge,
                sm_count=sm_count,
            ),
            fused_model=RidgeLatencyModel.fit(
                rows,
                stage="fused",
                ablation=ablation,
                ridge=ridge,
                sm_count=sm_count,
            ),
            minimum_fused_advantage=minimum_fused_advantage,
            force_fused=force_fused,
        )

    def select(
        self, candidates: Mapping[str, ScheduleMeasurement]
    ) -> tuple[ScheduleMeasurement, dict[str, float | str | bool]]:
        host = candidates.get("host")
        fused = tuple(row for row in candidates.values() if not row.is_host)
        if host is None or not fused:
            raise ValueError("two-stage selection requires host and fused candidates")
        predicted_host = self.host_model.predict_ms(host)
        predicted_fused = min(
            ((self.fused_model.predict_ms(row), row) for row in fused),
            key=lambda item: (item[0], item[1].family),
        )
        fused_limit = predicted_host * (1.0 - self.minimum_fused_advantage)
        admit_fused = self.force_fused or predicted_fused[0] < fused_limit
        selected = predicted_fused[1] if admit_fused else host
        return selected, {
            "selected": selected.family,
            "admit_fused": admit_fused,
            "predicted_host_ms": predicted_host,
            "predicted_fused_ms": predicted_fused[0],
            "predicted_fused_family": predicted_fused[1].family,
        }

    def to_dict(self) -> dict:
        return {
            "host_model": self.host_model.to_dict(),
            "fused_model": self.fused_model.to_dict(),
            "minimum_fused_advantage": self.minimum_fused_advantage,
            "force_fused": self.force_fused,
        }


@dataclass(frozen=True, slots=True)
class SelectionGates:
    minimum_problems: int = 30
    minimum_direction_accuracy: float = 0.99
    maximum_mean_regret: float = 0.005
    maximum_worst_regret: float = 0.01
    maximum_host_regressions: int = 0

    def validate(self) -> None:
        if self.minimum_problems <= 0:
            raise ValueError("minimum problem count must be positive")
        if not 0.0 <= self.minimum_direction_accuracy <= 1.0:
            raise ValueError("minimum direction accuracy must be in [0, 1]")
        if min(self.maximum_mean_regret, self.maximum_worst_regret) < 0.0:
            raise ValueError("regret gates must be non-negative")
        if self.maximum_host_regressions < 0:
            raise ValueError("host regression gate must be non-negative")


def _summarize_decisions(decisions: Sequence[dict], gates: SelectionGates) -> dict:
    problems = len(decisions)
    regrets = [float(row["regret"]) for row in decisions]
    direction_correct = sum(row["oracle_direction"] == row["selected_direction"] for row in decisions)
    host_regressions = sum(
        row["oracle_direction"] == "host" and row["selected_direction"] == "fused"
        for row in decisions
    )
    oracle_fused = sum(row["oracle_direction"] == "fused" for row in decisions)
    selected_fused = sum(row["selected_direction"] == "fused" for row in decisions)
    admitted_oracle_fused = sum(
        row["oracle_direction"] == "fused" and row["selected_direction"] == "fused"
        for row in decisions
    )
    admitted_fused_rows = [
        row
        for row in decisions
        if row["oracle_direction"] == "fused" and row["selected_direction"] == "fused"
    ]
    fused_schedule_regrets = [float(row["regret"]) for row in admitted_fused_rows]
    fused_schedule_exact = sum(
        row["oracle_family"] == row["selected_family"] for row in admitted_fused_rows
    )
    exact = sum(row["oracle_family"] == row["selected_family"] for row in decisions)
    metrics = {
        "problems": problems,
        "direction_accuracy": direction_correct / problems,
        "exact_family_accuracy": exact / problems,
        "mean_selection_regret": sum(regrets) / problems,
        "max_selection_regret": max(regrets),
        "host_regressions": host_regressions,
        "oracle_fused_problems": oracle_fused,
        "selected_fused_problems": selected_fused,
        "fused_coverage": admitted_oracle_fused / oracle_fused if oracle_fused else 1.0,
        "fused_schedule_exact_accuracy": (
            fused_schedule_exact / len(admitted_fused_rows) if admitted_fused_rows else 1.0
        ),
        "fused_schedule_mean_regret": (
            sum(fused_schedule_regrets) / len(fused_schedule_regrets)
            if fused_schedule_regrets
            else 0.0
        ),
        "fused_schedule_max_regret": max(fused_schedule_regrets, default=0.0),
    }
    failures: list[str] = []
    if problems < gates.minimum_problems:
        failures.append("insufficient held-out problems")
    if metrics["direction_accuracy"] < gates.minimum_direction_accuracy:
        failures.append("direction accuracy below gate")
    if metrics["mean_selection_regret"] > gates.maximum_mean_regret:
        failures.append("mean regret above gate")
    if metrics["max_selection_regret"] > gates.maximum_worst_regret:
        failures.append("maximum regret above gate")
    if host_regressions > gates.maximum_host_regressions:
        failures.append("host regression count above gate")
    return {**metrics, "admitted": not failures, "gate_failures": failures}


def evaluate_group_heldout(
    measurements: Iterable[ScheduleMeasurement],
    *,
    ablation: str = "full",
    ridge: float = 1.0,
    sm_count: int = DEFAULT_SM_COUNT,
    minimum_fused_advantage: float = 0.0,
    force_fused: bool = False,
    gates: SelectionGates = SelectionGates(),
) -> dict:
    """Leave one declared sweep group out and evaluate the complete policy."""

    gates.validate()
    rows = tuple(measurements)
    grouped = _group_measurements(rows)
    group_ids = tuple(sorted({row.group_id for row in rows}))
    if len(group_ids) < 2:
        raise ValueError("group-held-out evaluation requires at least two groups")
    decisions: list[dict] = []
    fold_models: list[dict] = []
    for heldout_group in group_ids:
        train = tuple(row for row in rows if row.group_id != heldout_group)
        selector = TwoStageSelector.fit(
            train,
            ablation=ablation,
            ridge=ridge,
            sm_count=sm_count,
            minimum_fused_advantage=minimum_fused_advantage,
            force_fused=force_fused,
        )
        fold_models.append({"heldout_group": heldout_group, "model": selector.to_dict()})
        heldout_shapes = [
            (key, candidates)
            for key, candidates in grouped.items()
            if key[0] == heldout_group
        ]
        for key, candidates in sorted(heldout_shapes):
            selected, prediction = selector.select(candidates)
            oracle = min(
                candidates.values(), key=lambda row: (row.measured_ms, row.family)
            )
            regret = selected.measured_ms / oracle.measured_ms - 1.0
            decisions.append(
                {
                    "heldout_group": heldout_group,
                    "world_size": key[1],
                    "rows_per_peer": key[2],
                    "k": key[3],
                    "n": key[4],
                    "oracle_family": oracle.family,
                    "selected_family": selected.family,
                    "oracle_direction": "host" if oracle.is_host else "fused",
                    "selected_direction": "host" if selected.is_host else "fused",
                    "oracle_ms": oracle.measured_ms,
                    "selected_ms": selected.measured_ms,
                    "regret": regret,
                    **prediction,
                }
            )
    return {
        "schema": SCHEMA_TWO_STAGE_EVALUATION,
        "protocol": "leave-one-declared-sweep-group-out",
        "ablation": ablation,
        "ridge": ridge,
        "sm_count": sm_count,
        "minimum_fused_advantage": minimum_fused_advantage,
        "force_fused": force_fused,
        "groups": list(group_ids),
        "gates": asdict(gates),
        "metrics": _summarize_decisions(decisions, gates),
        "decisions": decisions,
        "fold_models": fold_models,
    }


def evaluate_two_stage_ablations(
    measurements: Iterable[ScheduleMeasurement],
    *,
    ridge: float = 1.0,
    sm_count: int = DEFAULT_SM_COUNT,
    minimum_fused_advantage: float = 0.0,
    gates: SelectionGates = SelectionGates(),
) -> dict:
    rows = tuple(measurements)
    configurations = (
        ("full", "full", False),
        ("no_admission", "full", True),
        ("no_gamma", "no_gamma", False),
        ("no_rho", "no_rho", False),
        ("no_reduction", "no_reduction", False),
        ("k_only", "k_only", False),
    )
    reports = {}
    for label, ablation, force_fused in configurations:
        report = evaluate_group_heldout(
            rows,
            ablation=ablation,
            ridge=ridge,
            sm_count=sm_count,
            minimum_fused_advantage=minimum_fused_advantage,
            force_fused=force_fused,
            gates=gates,
        )
        reports[label] = report
    return {
        "schema": "edmir.two-stage-ablation-suite.v1",
        "primary": "full",
        "reports": reports,
    }


__all__ = [
    "AnalyticalScheduleFeatures",
    "DEFAULT_SM_COUNT",
    "RidgeLatencyModel",
    "SCHEMA_TWO_STAGE_EVALUATION",
    "ScheduleMeasurement",
    "SelectionGates",
    "TwoStageSelector",
    "evaluate_group_heldout",
    "evaluate_two_stage_ablations",
    "parse_schedule_family",
    "read_schedule_measurements",
]
