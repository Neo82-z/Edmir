"""Experimental within-workload cost ratios. Not used by compiler admission.

One linear score per candidate avoids inconsistent pairwise preferences. The
host score is zero; fused scores estimate log(T_fused / T_host). Training
uses every observed pair within a workload, with total pair weight one per
workload. This is a squared log-ratio surrogate, not a regret guarantee.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from itertools import combinations
from typing import Mapping, Sequence

from .two_stage_selection import (
    DEFAULT_SM_COUNT,
    ScheduleMeasurement,
    _feature_names,
    _feature_values,
    _group_measurements,
    _require_finite,
    _require_int,
    _solve_linear_system,
    _validate_geometry,
    _validate_linear_state,
)


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    """Inference input, deliberately without a measured latency."""

    environment_id: str
    family: str
    world_size: int
    rows_per_peer: int
    k: int
    n: int
    remote_input_bytes: int
    remote_a_replay: int
    split_k_slices: int
    n_tiles_per_cta: int
    estimated_ctas_per_op: int

    @classmethod
    def from_measurement(cls, row: ScheduleMeasurement, environment_id: str):
        return cls(environment_id=environment_id, **{
            field.name: getattr(row, field.name)
            for field in fields(cls) if field.name != "environment_id"
        })

    @property
    def shape(self):
        return (self.world_size, self.rows_per_peer, self.k, self.n)

    @property
    def is_host(self):
        return self.family == "host"

    def validate(self):
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("candidate environment must be explicit")
        _validate_geometry(self)


def strict_surfaces(rows: Sequence[ScheduleMeasurement]):
    return _group_measurements(rows)


@dataclass(frozen=True, slots=True)
class RelativeCostSelector:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_environment_id: str
    training_shapes: tuple[tuple[int, int, int, int], ...]
    ridge: float
    sm_count: int
    minimum_fused_advantage: float = 0.0
    calibration_offset: float = 0.0
    calibration_environment_id: str | None = None
    calibration_shapes: tuple[tuple[int, int, int, int], ...] = ()

    def validate(self):
        if self.feature_names != _feature_names("fused", "interaction_depth_grid"):
            raise ValueError("relative model has an unknown feature schema")
        _validate_linear_state(self.feature_names, self.means, self.scales, self.coefficients,
                               ridge=self.ridge, sm_count=self.sm_count)
        _require_finite("calibration offset", self.calibration_offset)
        if not 0 <= self.minimum_fused_advantage < 1:
            raise ValueError("minimum fused advantage must be in [0, 1)")
        if not isinstance(self.training_environment_id, str) or not self.training_environment_id.strip():
            raise ValueError("training environment must be explicit")
        if not self.training_shapes:
            raise ValueError("training workload provenance is missing")
        for shapes in (self.training_shapes, self.calibration_shapes):
            for shape in shapes:
                if len(shape) != 4:
                    raise ValueError("workload shape must be (P, M, K, N)")
                for i, value in enumerate(shape):
                    _require_int("workload dimension", value, minimum=2 if i == 0 else 1)
            if len(set(shapes)) != len(shapes) or len({s[0] for s in shapes}) > 1:
                raise ValueError("workload provenance must be unique and use one world size")
        if self.calibration_environment_id is None:
            if self.calibration_shapes or self.calibration_offset != 0:
                raise ValueError("calibration state has no environment provenance")
        else:
            if not isinstance(self.calibration_environment_id, str) or not self.calibration_environment_id.strip():
                raise ValueError("calibration environment must be explicit")
            if not self.calibration_shapes:
                raise ValueError("calibration workload provenance is missing")
            if self.calibration_environment_id == self.training_environment_id:
                if self.calibration_shapes[0][0] != self.training_shapes[0][0]:
                    raise ValueError("one environment identity cannot describe different world sizes")
                if set(self.calibration_shapes) & set(self.training_shapes):
                    raise ValueError("calibration overlaps training workloads")

    @classmethod
    def fit(
        cls, rows: Sequence[ScheduleMeasurement], *, environment_id: str,
        ridge: float = 1.0, sm_count: int = DEFAULT_SM_COUNT,
        minimum_fused_advantage: float = 0.0,
    ):
        _require_finite("ridge", ridge)
        if not isinstance(environment_id, str) or not environment_id.strip() or ridge <= 0:
            raise ValueError("explicit environment and positive finite ridge required")
        if not 0 <= minimum_fused_advantage < 1:
            raise ValueError("minimum fused advantage must be in [0, 1)")
        groups = strict_surfaces(rows)
        if len(groups) < 2:
            raise ValueError("relative model requires at least two workloads")
        if len({row.world_size for row in rows}) != 1:
            raise ValueError("fit each environment/world size separately")
        names = _feature_names("fused", "interaction_depth_grid")
        # Fit the scaler on training data only. Each workload contributes one
        # unit of fused-feature weight, regardless of its candidate count.
        raw = []
        for candidates in groups.values():
            fused = [row for row in candidates.values() if not row.is_host]
            raw.extend((_feature_values(row, names, sm_count=sm_count), 1 / len(fused))
                       for row in fused)
        means = tuple(sum(x[i] * w for x, w in raw) / len(groups)
                      for i in range(len(names)))
        scales = tuple(max(math.sqrt(
            sum(w * (x[i] - means[i]) ** 2 for x, w in raw) / len(groups)
        ), 1e-9) for i in range(len(names)))
        width = len(names) + 1
        normal = [[0.0] * width for _ in range(width)]
        projected = [0.0] * width
        for candidates in groups.values():
            design = []
            for row in sorted(candidates.values(), key=lambda r: r.family):
                if row.is_host:
                    x = (0.0,) * width
                else:
                    values = _feature_values(row, names, sm_count=sm_count)
                    x = (1.0, *((v - m) / s for v, m, s in zip(values, means, scales)))
                design.append((x, math.log(row.measured_ms)))
            weight = 1 / math.comb(len(design), 2)
            for (a, ya), (b, yb) in combinations(design, 2):
                x = tuple(ai - bi for ai, bi in zip(a, b))
                for i in range(width):
                    projected[i] += weight * x[i] * (ya - yb)
                    for j in range(width):
                        normal[i][j] += weight * x[i] * x[j]
        for i in range(1, width):
            normal[i][i] += ridge
        return cls(names, means, scales, tuple(_solve_linear_system(normal, projected)),
                   environment_id, tuple(sorted(row[1:] for row in groups)),
                   ridge, sm_count, minimum_fused_advantage)

    def predict_log_ratio(
        self, candidate: ScheduleCandidate, *, allow_environment_transfer: bool = False,
    ) -> float:
        self.validate()
        if not isinstance(candidate, ScheduleCandidate):
            raise TypeError("inference requires ScheduleCandidate without timing labels")
        candidate.validate()
        for env, shapes in ((self.training_environment_id, self.training_shapes),
                            (self.calibration_environment_id, self.calibration_shapes)):
            if env == candidate.environment_id and candidate.world_size != shapes[0][0]:
                raise ValueError("candidate world size contradicts environment provenance")
        target_environment = self.calibration_environment_id or self.training_environment_id
        if candidate.environment_id != target_environment and not allow_environment_transfer:
            raise ValueError("unvalidated environment transfer must be explicit")
        if candidate.is_host:
            return 0.0
        values = _feature_values(candidate, self.feature_names, sm_count=self.sm_count)
        offset = (self.calibration_offset
                  if candidate.environment_id == self.calibration_environment_id else 0.0)
        prediction = self.coefficients[0] + offset + sum(
            c * (v - m) / s
            for c, v, m, s in zip(self.coefficients[1:], values, self.means, self.scales)
        )
        _require_finite("predicted log ratio", prediction)
        return prediction

    def explain(self, candidate: ScheduleCandidate, *, allow_environment_transfer: bool = False):
        """Feature contributions for debugging, not causal attributions."""
        prediction = self.predict_log_ratio(candidate, allow_environment_transfer=allow_environment_transfer)
        if candidate.is_host:
            return {"predicted_log_ratio": prediction, "intercept": 0.0,
                    "calibration_offset": 0.0, "terms": []}
        values = _feature_values(candidate, self.feature_names, sm_count=self.sm_count)
        return {
            "predicted_log_ratio": prediction, "intercept": self.coefficients[0],
            "calibration_offset": (self.calibration_offset
                                   if candidate.environment_id == self.calibration_environment_id else 0.0),
            "terms": [{"feature": name, "raw": value, "standardized": (value - mean) / scale,
                       "coefficient": coefficient, "contribution": coefficient * (value - mean) / scale}
                      for name, value, mean, scale, coefficient in zip(
                          self.feature_names, values, self.means, self.scales, self.coefficients[1:])],
        }

    def select(
        self, candidates: Mapping[str, ScheduleCandidate], *,
        allow_environment_transfer: bool = False,
    ):
        if not candidates or any(not isinstance(c, ScheduleCandidate)
                                 for c in candidates.values()):
            raise TypeError("inference requires static ScheduleCandidate inputs")
        if any(name != c.family for name, c in candidates.items()):
            raise ValueError("candidate mapping key must equal family")
        if len({(c.environment_id, c.shape) for c in candidates.values()}) != 1:
            raise ValueError("selection cannot mix workloads or environments")
        if "host" not in candidates or len(candidates) < 2:
            raise ValueError("host and fused candidates are required")
        predictions = {name: self.predict_log_ratio(
            candidate, allow_environment_transfer=allow_environment_transfer
        ) for name, candidate in candidates.items()}
        best = min((name for name in candidates if name != "host"),
                   key=lambda name: (predictions[name], name))
        selected = (best if predictions[best] < math.log1p(-self.minimum_fused_advantage)
                    else "host")
        return candidates[selected], {
            "predicted_fused_family": best,
            "predicted_fused_log_ratio": predictions[best],
            "admit_fused": selected != "host",
        }

    def calibrate_offset(self, rows: Sequence[ScheduleMeasurement], *, environment_id: str):
        """One target-environment intercept; never refit the schedule ranking."""
        self.validate()
        if not isinstance(environment_id, str) or not environment_id.strip():
            raise ValueError("explicit calibration environment required")
        groups = strict_surfaces(rows)
        if len({row.world_size for row in rows}) != 1:
            raise ValueError("calibration must use one environment/world size")
        if environment_id == self.training_environment_id and rows[0].world_size != self.training_shapes[0][0]:
            raise ValueError("calibration world size contradicts environment provenance")
        if environment_id == self.training_environment_id and any(
            row.shape in self.training_shapes for row in rows
        ):
            raise ValueError("calibration overlaps training workloads")
        base = replace(self, calibration_offset=0.0, calibration_environment_id=None,
                       calibration_shapes=())
        residuals = []
        for candidates in groups.values():
            host_ms = candidates["host"].measured_ms
            residuals.append(sum(
                (math.log(row.measured_ms) - math.log(host_ms)) - base.predict_log_ratio(
                    ScheduleCandidate.from_measurement(row, environment_id),
                    allow_environment_transfer=True,
                ) for row in candidates.values() if not row.is_host
            ) / (len(candidates) - 1))
        calibrated = replace(base, calibration_offset=sum(residuals) / len(residuals),
                             calibration_environment_id=environment_id,
                             calibration_shapes=tuple(sorted(row[1:] for row in groups)))
        calibrated.validate()
        return calibrated

    def to_dict(self):
        self.validate()
        return {"schema": "edmir.relative-cost-selector.v1",
                "eligible_for_compiler_admission": False, **asdict(self)}
