"""Controlled split-K-only revision of the experimental factorized predictor.

q=1 is the zero reference. Neither feature basis forces a monotone ranking.
The new wave proxy counts ideal CTA waves, not measured occupancy or latency.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from itertools import combinations

from .factorized_selection import FactorizedSelector, IdentifiedMeasurements, features
from .two_stage_selection import (
    _require_finite, _solve_linear_system, _validate_geometry, _validate_linear_state,
)

FEATURES = {
    "original": ("log_q", "log_q_x_log_k_over_4096",
                 "log_q_x_log_base_grid_over_64", "extra_partitions_over_k"),
    "waves": ("log_wave_work_ratio", "log_q", "extra_partitions_over_16",
              "inverse_k_tile_depth_delta"),
}
LOSSES = ("reference", "pairwise")


def split_features(row, basis: str):
    _validate_geometry(row)
    if basis == "original":
        return features(row, "split")
    if basis != "waves":
        raise ValueError("unknown split-K feature basis")
    q = row.split_k_slices
    grid = ((row.rows_per_peer + 63)//64) * ((row.n + 127)//128)
    base_waves = (grid + 107)//108
    split_waves = (grid*q + 107)//108
    return (math.log2(split_waves/(q*base_waves)), math.log2(q),
            (q-1)/16, 32*(q-1)/row.k)


@dataclass(frozen=True, slots=True)
class SplitKFit:
    basis: str
    loss: str
    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge: float
    workloads: int
    training_pairs: int
    stage: str = "split"

    def validate(self):
        if self.stage != "split" or self.basis not in FEATURES or self.loss not in LOSSES:
            raise ValueError("unknown split-K fit contract")
        _validate_linear_state(self.names, self.means, self.scales, self.coefficients,
                               ridge=self.ridge, sm_count=108)
        if self.names != FEATURES[self.basis] or any(self.means) or self.coefficients[0] != 0:
            raise ValueError("split-K reference/schema changed")
        if type(self.workloads) is not int or self.workloads < 2:
            raise ValueError("at least two split-K workloads required")
        if type(self.training_pairs) is not int or self.training_pairs < 2:
            raise ValueError("split-K training pairs are missing")

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, basis, loss, ridge=1.0):
        if basis not in FEATURES or loss not in LOSSES:
            raise ValueError("unknown split-K experiment configuration")
        _require_finite("ridge", ridge)
        if ridge <= 0:
            raise ValueError("positive ridge required")
        surfaces = dataset.validate()
        if len(surfaces) < 2:
            raise ValueError("at least two training workloads required")
        groups = []
        for surface in surfaces.values():
            rows = sorted((r for r in surface.values() if not r.is_host and r.n_tiles_per_cta == 1),
                          key=lambda r: r.split_k_slices)
            if len(rows) < 2 or rows[0].family != "direct":
                raise ValueError("each workload needs direct and a split-K candidate")
            groups.append(rows)
        width = len(FEATURES[basis])
        # Same scaler for both losses: one unit of q>1 feature weight/workload.
        scales = tuple(max(math.sqrt(sum(
            sum(split_features(r, basis)[i]**2 for r in group[1:])/(len(group)-1)
            for group in groups)/len(groups)), 1e-9) for i in range(width))
        normal, rhs = [[0.0]*width for _ in range(width)], [0.0]*width
        pairs = 0
        for group in groups:
            pair_rows = ([(r, group[0]) for r in group[1:]] if loss == "reference"
                         else list(combinations(group, 2)))
            pairs += len(pair_rows)
            weight = 1/len(pair_rows)
            for a, b in pair_rows:
                x = [(u-v)/s for u, v, s in zip(split_features(a, basis), split_features(b, basis), scales)]
                y = math.log(a.measured_ms)-math.log(b.measured_ms)
                for i in range(width):
                    rhs[i] += weight*x[i]*y
                    for j in range(width):
                        normal[i][j] += weight*x[i]*x[j]
        for i in range(width):
            normal[i][i] += ridge
        result = cls(basis, loss, FEATURES[basis], (0.0,)*width, scales,
                     (0.0, *_solve_linear_system(normal, rhs)), ridge, len(groups), pairs)
        result.validate()
        return result

    def predict(self, row):
        self.validate()
        prediction = sum(c*x/s for c, x, s in zip(
            self.coefficients[1:], split_features(row, self.basis), self.scales))
        _require_finite("split-K score", prediction)
        return prediction

    def to_dict(self):
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitKRevision:
    """Compose a new split component without changing gate/group/calibration."""
    original: FactorizedSelector
    split: SplitKFit

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, basis, loss, ridge=1.0):
        return cls(FactorizedSelector.fit(dataset, ridge=ridge),
                   SplitKFit.fit(dataset, basis=basis, loss=loss, ridge=ridge))

    def effective(self):
        self.original.validate()
        self.split.validate()
        return replace(self.original, split=self.split)

    def calibrate_offset(self, dataset):
        return replace(self, original=self.original.calibrate_offset(dataset))

    def select(self, candidates, *, baseline, allow_environment_transfer=False):
        return self.effective().select(candidates, baseline=baseline,
                                       allow_environment_transfer=allow_environment_transfer)

    def to_dict(self):
        self.effective().validate()
        return {"schema": "edmir.splitk-revision.v1", "eligible_for_compiler_admission": False,
                "unchanged_factorized_model": self.original.to_dict(), "split": self.split.to_dict()}
