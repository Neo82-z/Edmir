"""Experimental factorization of three measured ratios; no compiler admission.

log(T(q,g)/T(host)) = log(T(1,1)/T(host))
                  + log(T(q,1)/T(1,1)) + log(T(q,g)/T(q,1)).

This is an identity for labels. Separate ridge fits approximate its terms;
neither the identity nor an eligibility guard guarantees prediction accuracy.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping

from .host_baseline import HostBaseline, PER_SOURCE_CUTLASS, host_baseline
from .relative_selection import ScheduleCandidate, strict_surfaces
from .two_stage_selection import (
    ScheduleMeasurement, _require_finite, _solve_linear_system, _validate_linear_state,
)

GATE_FEATURES = ("log_m", "log_k", "log_n")
SPLIT_FEATURES = (
    "log_q", "log_q_x_log_k_over_4096", "log_q_x_log_base_grid_over_64",
    "extra_partitions_over_k",
)
GROUP_FEATURES = ("log_split_depth_over_4096", "log_base_grid_over_64")


def features(row, stage):
    base_grid = math.ceil(row.rows_per_peer / 64) * math.ceil(row.n / 128)
    log_q = math.log2(row.split_k_slices)
    log_k = math.log2(row.k / 4096)
    log_grid = math.log2(base_grid / 64)
    if stage == "gate":
        return (math.log2(row.rows_per_peer), math.log2(row.k), math.log2(row.n))
    if stage == "split":
        # Zero for q=1: no intercept or centering in this component.
        return (log_q, log_q * log_k, log_q * log_grid,
                (row.split_k_slices - 1) * 4096 / row.k)
    if stage == "group":
        return (log_k - log_q, log_grid)
    raise ValueError("unknown ratio component")


@dataclass(frozen=True, slots=True)
class IdentifiedMeasurements:
    rows: tuple[ScheduleMeasurement, ...]
    environment_id: str
    baseline: HostBaseline

    def validate(self):
        self.baseline.validate()
        # The analytical geometry currently describes this partition/kernel.
        # A different baseline needs its own data and a separately fitted model.
        if self.baseline != PER_SOURCE_CUTLASS:
            raise ValueError("factorized v1 supports the measured per-source CUTLASS baseline only")
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("explicit environment identity required")
        groups = strict_surfaces(self.rows)
        if len({row.world_size for row in self.rows}) != 1:
            raise ValueError("one environment must have one world size")
        if any("direct" not in surface for surface in groups.values()):
            raise ValueError("each workload needs the direct reference")
        return groups


@dataclass(frozen=True, slots=True)
class RatioFit:
    stage: str
    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge: float
    samples: int

    def validate(self):
        if self.names != {"gate": GATE_FEATURES, "split": SPLIT_FEATURES,
                          "group": GROUP_FEATURES}.get(self.stage):
            raise ValueError("unknown component feature schema")
        _validate_linear_state(self.names, self.means, self.scales, self.coefficients,
                               ridge=self.ridge, sm_count=108)
        if type(self.samples) is not int or self.samples < 2:
            raise ValueError("component needs at least two measured pairs")
        if self.stage == "split" and (any(self.means) or self.coefficients[0] != 0):
            raise ValueError("split reference must be exactly zero at q=1")

    @classmethod
    def fit(cls, stage, samples, ridge):
        """samples = (geometry, log-ratio label, workload-normalized weight)."""
        _require_finite("ridge", ridge)
        if ridge <= 0 or len(samples) < 2:
            raise ValueError("positive ridge and at least two pairs required")
        names = {"gate": GATE_FEATURES, "split": SPLIT_FEATURES, "group": GROUP_FEATURES}[stage]
        raw = [(features(row, stage), y, w) for row, y, w in samples]
        for x, y, w in raw:
            for v in (*x, y, w):
                _require_finite("component sample", v)
            if w <= 0:
                raise ValueError("sample weight must be positive")
        total = sum(w for _, _, w in raw)
        intercept = stage != "split"
        means = tuple(sum(w*x[i] for x, _, w in raw)/total if intercept else 0.0
                      for i in range(len(names)))
        scales = tuple(max(math.sqrt(sum(w*(x[i]-means[i])**2 for x, _, w in raw)/total), 1e-9)
                       for i in range(len(names)))
        width = len(names) + int(intercept)
        normal, rhs = [[0.0]*width for _ in range(width)], [0.0]*width
        for raw_x, y, w in raw:
            x = tuple((v-m)/s for v, m, s in zip(raw_x, means, scales))
            if intercept:
                x = (1.0, *x)
            for i in range(width):
                rhs[i] += w*x[i]*y
                for j in range(width):
                    normal[i][j] += w*x[i]*x[j]
        for i in range(int(intercept), width):
            normal[i][i] += ridge
        coef = tuple(_solve_linear_system(normal, rhs))
        fitted = cls(stage, names, means, scales, coef if intercept else (0.0, *coef),
                     ridge, len(samples))
        fitted.validate()
        return fitted

    def predict(self, row):
        self.validate()
        result = self.coefficients[0] + sum(
            c*(v-m)/s for c, v, m, s in zip(
                self.coefficients[1:], features(row, self.stage), self.means, self.scales))
        _require_finite("component prediction", result)
        return result


@dataclass(frozen=True, slots=True)
class GroupSupport:
    m: int
    q: int
    k_min: int
    k_max: int
    n_min: int
    n_max: int
    pairs: int

    def covers(self, row):
        return (self.pairs >= 2 and self.m == row.rows_per_peer and self.q == row.split_k_slices
                and self.k_min <= row.k <= self.k_max and self.n_min <= row.n <= self.n_max)


@dataclass(frozen=True, slots=True)
class FactorizedSelector:
    gate: RatioFit
    split: RatioFit
    group: RatioFit | None
    baseline_id: str
    training_environment_id: str
    training_shapes: tuple[tuple[int, int, int, int], ...]
    trained_q: tuple[int, ...]
    grouped_support: tuple[GroupSupport, ...]
    minimum_fused_advantage: float = 0.0
    calibration_offset: float = 0.0
    calibration_environment_id: str | None = None
    calibration_shapes: tuple[tuple[int, int, int, int], ...] = ()

    def validate(self):
        if host_baseline(self.baseline_id) != PER_SOURCE_CUTLASS:
            raise ValueError("unsupported fitted host identity")
        for fit, stage in ((self.gate, "gate"), (self.split, "split"), (self.group, "group")):
            if fit is not None:
                fit.validate()
                if fit.stage != stage:
                    raise ValueError("ratio components are interchanged")
        if not isinstance(self.training_environment_id, str) or not self.training_environment_id.strip():
            raise ValueError("training environment is missing")
        if len(self.training_shapes) < 2:
            raise ValueError("training workloads are missing")
        for shapes in (self.training_shapes, self.calibration_shapes):
            if len(set(shapes)) != len(shapes) or len({s[0] for s in shapes}) > 1:
                raise ValueError("invalid workload provenance")
            if any(len(s) != 4 or any(type(v) is not int or v < (2 if i == 0 else 1)
                                     for i, v in enumerate(s)) for s in shapes):
                raise ValueError("invalid workload dimensions")
        if not self.trained_q or any(type(q) is not int or q not in (1, 2, 4, 8, 16)
                                     for q in self.trained_q):
            raise ValueError("invalid trained split factors")
        if len(set(self.trained_q)) != len(self.trained_q) or 1 not in self.trained_q:
            raise ValueError("split support needs a unique direct reference")
        if (self.group is None) != (not self.grouped_support):
            raise ValueError("group component and support disagree")
        for cell in self.grouped_support:
            if any(type(v) is not int or v < 1 for v in asdict(cell).values()):
                raise ValueError("invalid grouped support")
            if cell.q not in self.trained_q or cell.k_min > cell.k_max or cell.n_min > cell.n_max:
                raise ValueError("invalid grouped support bounds")
            if cell.m not in {s[1] for s in self.training_shapes}:
                raise ValueError("grouped M is absent from training")
        _require_finite("calibration offset", self.calibration_offset)
        _require_finite("minimum fused advantage", self.minimum_fused_advantage)
        if not 0 <= self.minimum_fused_advantage < 1:
            raise ValueError("minimum fused advantage must be in [0, 1)")
        if self.calibration_environment_id is None:
            if self.calibration_shapes or self.calibration_offset != 0:
                raise ValueError("calibration provenance is missing")
        else:
            if not isinstance(self.calibration_environment_id, str) or not self.calibration_environment_id.strip():
                raise ValueError("calibration environment is missing")
            if not self.calibration_shapes:
                raise ValueError("calibration shapes are missing")
            if self.calibration_environment_id == self.training_environment_id:
                if self.calibration_shapes[0][0] != self.training_shapes[0][0]:
                    raise ValueError("environment world size changed")
                if set(self.calibration_shapes) & set(self.training_shapes):
                    raise ValueError("calibration overlaps training")

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, ridge=1.0, minimum_fused_advantage=0.0):
        groups = dataset.validate()
        if len(groups) < 2:
            raise ValueError("at least two workloads required")
        gate, split, grouped, paired = [], [], [], []
        for surface in groups.values():
            direct = surface["direct"]
            gate.append((direct, math.log(direct.measured_ms)-math.log(surface["host"].measured_ms), 1.0))
            split_rows = [r for r in surface.values() if r.n_tiles_per_cta == 1 and r.split_k_slices > 1]
            split.extend((r, math.log(r.measured_ms)-math.log(direct.measured_ms), 1/len(split_rows))
                         for r in split_rows)
            pairs = []
            for r in surface.values():
                if r.n_tiles_per_cta == 2:
                    base = surface.get(r.family.removesuffix("_groupedn2"))
                    if base is None:
                        raise ValueError("grouped label has no same-q ungrouped measurement")
                    pairs.append((r, math.log(r.measured_ms)-math.log(base.measured_ms)))
                    paired.append(r)
            grouped.extend((r, y, 1/len(pairs)) for r, y in pairs)
        support = []
        for m, q in sorted({(r.rows_per_peer, r.split_k_slices) for r in paired}):
            subset = [r for r in paired if (r.rows_per_peer, r.split_k_slices) == (m, q)]
            support.append(GroupSupport(m, q, min(r.k for r in subset), max(r.k for r in subset),
                                        min(r.n for r in subset), max(r.n for r in subset), len(subset)))
        model = cls(
            RatioFit.fit("gate", gate, ridge), RatioFit.fit("split", split, ridge),
            RatioFit.fit("group", grouped, ridge) if grouped else None,
            dataset.baseline.baseline_id, dataset.environment_id,
            tuple(sorted({r.shape for r in dataset.rows})),
            tuple(sorted({r.split_k_slices for r in dataset.rows if r.n_tiles_per_cta == 1})),
            tuple(support), minimum_fused_advantage)
        model.validate()
        return model

    def _check_candidate(self, row, baseline, transfer):
        self.validate()
        baseline.validate()
        if baseline.baseline_id != self.baseline_id:
            raise ValueError("host implementation differs from the fitted ratio denominator")
        if not isinstance(row, ScheduleCandidate):
            raise TypeError("prediction needs static candidates without timing labels")
        row.validate()
        for env, shapes in ((self.training_environment_id, self.training_shapes),
                            (self.calibration_environment_id, self.calibration_shapes)):
            if row.environment_id == env and row.world_size != shapes[0][0]:
                raise ValueError("environment world size changed")
        if row.environment_id != (self.calibration_environment_id or self.training_environment_id) and not transfer:
            raise ValueError("environment transfer must be explicit")

    def exclusion_reason(self, row):
        if row.is_host:
            return None
        if row.split_k_slices not in self.trained_q:
            return "split factor absent from training"
        if row.n_tiles_per_cta == 2:
            if not any(cell.covers(row) for cell in self.grouped_support):
                return "no paired grouped-N support at this exact M/q and within observed K/N bounds"
        return None

    def explain(self, row, *, baseline: HostBaseline, allow_environment_transfer=False):
        self._check_candidate(row, baseline, allow_environment_transfer)
        reason = self.exclusion_reason(row)
        if reason:
            return {"eligible": False, "reason": reason, "log_ratio": None}
        if row.is_host:
            gate = split = grouped = offset = 0.0
        else:
            gate = self.gate.predict(row)
            split = self.split.predict(row)
            grouped = self.group.predict(row) if row.n_tiles_per_cta == 2 else 0.0
            offset = self.calibration_offset if row.environment_id == self.calibration_environment_id else 0.0
        result = gate + split + grouped + offset
        _require_finite("factorized prediction", result)
        return {"eligible": True, "gate": gate, "split": split, "group": grouped,
                "calibration_offset": offset, "log_ratio": result}

    def select(self, candidates: Mapping[str, ScheduleCandidate], *, baseline: HostBaseline,
               allow_environment_transfer=False):
        if not candidates or any(not isinstance(c, ScheduleCandidate) for c in candidates.values()):
            raise TypeError("selection needs static candidates")
        if any(name != c.family for name, c in candidates.items()):
            raise ValueError("candidate names disagree")
        if len({(c.environment_id, c.shape) for c in candidates.values()}) != 1 or "host" not in candidates:
            raise ValueError("one workload, one environment and the explicit host are required")
        predictions = {name: self.explain(c, baseline=baseline,
                        allow_environment_transfer=allow_environment_transfer) for name, c in candidates.items()}
        eligible = [name for name, p in predictions.items() if name != "host" and p["eligible"]]
        best = min(eligible, key=lambda name: (predictions[name]["log_ratio"], name)) if eligible else None
        chosen = best if best and predictions[best]["log_ratio"] < math.log1p(-self.minimum_fused_advantage) else "host"
        return candidates[chosen], {
            "admit_fused": chosen != "host", "predicted_fused_family": best,
            "predictions": predictions, "baseline_id": self.baseline_id,
            "eligible_candidate_count": sum(p["eligible"] for p in predictions.values()),
            "grouped_available": any(c.n_tiles_per_cta == 2 for c in candidates.values()),
            "grouped_eligible": any(c.n_tiles_per_cta == 2 and predictions[n]["eligible"]
                                    for n, c in candidates.items()),
        }

    def calibrate_offset(self, dataset: IdentifiedMeasurements):
        """Fit only direct/host residuals; never alter q/g rankings or support."""
        self.validate()
        groups = dataset.validate()
        if dataset.baseline.baseline_id != self.baseline_id:
            raise ValueError("cannot calibrate across host implementations")
        if dataset.environment_id == self.training_environment_id:
            if dataset.rows[0].world_size != self.training_shapes[0][0]:
                raise ValueError("environment world size changed")
            if any(r.shape in self.training_shapes for r in dataset.rows):
                raise ValueError("calibration overlaps training")
        residuals = [math.log(s["direct"].measured_ms)-math.log(s["host"].measured_ms)
                     - self.gate.predict(s["direct"]) for s in groups.values()]
        fitted = replace(self, calibration_offset=sum(residuals)/len(residuals),
                         calibration_environment_id=dataset.environment_id,
                         calibration_shapes=tuple(sorted({r.shape for r in dataset.rows})))
        fitted.validate()
        return fitted

    def to_dict(self):
        self.validate()
        return {"schema": "edmir.factorized-selector.v1", "eligible_for_compiler_admission": False,
                **asdict(self)}
