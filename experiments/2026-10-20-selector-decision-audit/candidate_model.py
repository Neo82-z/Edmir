"""Local controlled alternatives; deliberately not imported by the compiler."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from edmir.compiler.factorized_selection import FactorizedSelector, features
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.splitk_revision import SplitKFit, split_features


def split_vector(row, interactions):
    base = split_features(row, "waves")
    if not interactions:
        return base
    q = math.log2(row.split_k_slices)
    grid = math.ceil(row.rows_per_peer/64)*math.ceil(row.n/128)
    return (*base, q*math.log2(row.k/4096), q*math.log2(grid/64))


def joint_vector(row, interactions):
    width = 4 + len(split_vector(row, interactions)) + 3
    if row.is_host:
        return (0.,)*width
    grouped = row.n_tiles_per_cta == 2
    return (1., *features(row, "gate"), *split_vector(row, interactions),
            float(grouped), *(features(row, "group") if grouped else (0., 0.)))


@dataclass
class Fit:
    interactions: bool
    joint: bool
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    gradient_max_abs: float
    design_rank: int

    @classmethod
    def train(cls, dataset, *, interactions, joint):
        surfaces = list(dataset.validate().values())
        vector = joint_vector if joint else split_vector
        groups = [
            sorted((r for r in surface.values() if joint or (
                not r.is_host and r.n_tiles_per_cta == 1)), key=lambda r: r.family)
            for surface in surfaces]
        # Training-only, one unit of scaling weight per workload. For split
        # fits use q>1, exactly as the existing wave model does.
        scaling = [[vector(r, interactions) for r in group
                    if not r.is_host and (joint or r.split_k_slices > 1)]
                   for group in groups]
        width = len(scaling[0][0])
        means = np.zeros(width)
        if joint:
            means[1:4] = np.mean([np.mean(g, axis=0) for g in scaling], axis=0)[1:4]
        scales = np.maximum(np.sqrt(np.mean([
            np.mean((np.asarray(g)-means)**2, axis=0) for g in scaling], axis=0)), 1e-9)
        if joint:
            scales[0] = 1.

        def transform(row):
            if joint and row.is_host:
                return np.zeros(width)
            return (np.asarray(vector(row, interactions))-means)/scales

        x, y = [], []
        for group in groups:
            pairs = list(combinations(group, 2))
            for a, b in pairs:
                weight = 1/math.sqrt(len(pairs))
                x.append((transform(a)-transform(b))*weight)
                y.append(math.log(a.measured_ms/b.measured_ms)*weight)
        x, y = np.asarray(x), np.asarray(y)
        penalty = np.ones(width)
        if joint:
            penalty[0] = 0.  # Global intercept, as in the existing relative model.
        coefficients = np.linalg.lstsq(
            np.vstack((x, np.diag(np.sqrt(penalty)))),
            np.r_[y, np.zeros(width)], rcond=None)[0]
        gradient = x.T@(x@coefficients-y)+penalty*coefficients
        return cls(interactions, joint, means, scales, coefficients,
                   float(np.max(np.abs(gradient))), int(np.linalg.matrix_rank(x)))

    def predict(self, candidate):
        if self.joint and candidate.is_host:
            return 0.
        vector = joint_vector if self.joint else split_vector
        return float(self.coefficients @ (
            (np.asarray(vector(candidate, self.interactions))-self.means)/self.scales))


class DecisionModel:
    def __init__(self, dataset, *, joint, interactions):
        self.reference = FactorizedSelector.fit(dataset)
        self.baseline = dataset.baseline
        self.joint = joint
        self.fit = Fit.train(dataset, joint=joint, interactions=interactions)
        # Independently reconstruct the old split fit before accepting any
        # interpretation of the new controls.
        if not joint and not interactions:
            old = SplitKFit.fit(dataset, basis="waves", loss="pairwise")
            np.testing.assert_allclose(self.fit.coefficients, old.coefficients[1:], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(self.fit.scales, old.scales, atol=1e-10, rtol=1e-10)

    def scores(self, candidates):
        if not candidates or any(not isinstance(c, ScheduleCandidate) for c in candidates.values()):
            raise TypeError("only static candidates without timing labels are accepted")
        if any(name != c.family for name, c in candidates.items()):
            raise ValueError("candidate identity mismatch")
        if len({(c.environment_id, c.shape) for c in candidates.values()}) != 1 or "host" not in candidates:
            raise ValueError("one workload and explicit host required")
        result = {}
        for name, candidate in candidates.items():
            detail = self.reference.explain(candidate, baseline=self.baseline)
            if not detail["eligible"]:
                continue
            if candidate.is_host:
                value = 0.
            elif self.joint:
                value = self.fit.predict(candidate)
            else:
                value = detail["gate"] + self.fit.predict(candidate) + detail["group"]
            if not math.isfinite(value):
                raise ValueError("nonfinite prediction")
            result[name] = value
        return result

    def choose(self, candidates):
        scores = self.scores(candidates)
        best = min((n for n in scores if n != "host"), key=lambda n: (scores[n], n), default=None)
        return (best if best and scores[best] < 0 else "host"), scores
