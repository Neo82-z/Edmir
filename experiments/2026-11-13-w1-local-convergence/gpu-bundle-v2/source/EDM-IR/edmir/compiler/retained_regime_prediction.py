"""Bounded test of a retained smaller cost proxy; no compiler admission.

Motivated by separate metric/performance validation, not a reproduction of
Warpspeed's cache estimator. All inputs and host expressions stay unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from .mechanism_prediction import HOSTS, MechanismModel
from .regime_prediction import RegimeModel, max_branch_fit, regime_basis

RETAINED_FRACTIONS = (0., .25, .5, .75, 1.)


@dataclass(frozen=True)
class RetainedRegimeModel(RegimeModel):
    retained_fraction: float = 0.

    @classmethod
    def fit(cls, configs, *, retained_fraction, include_flagged=False):
        if type(retained_fraction) not in (int, float) or retained_fraction not in RETAINED_FRACTIONS:
            raise ValueError("only the five preregistered retained fractions are supported")
        baseline = MechanismModel.fit(configs, variant="cutlass_depth", include_flagged=include_flagged)
        if "fused" not in baseline.models:
            raise ValueError("need at least two fused training shapes")
        rows = sorted((r for r in configs if r["family"] not in HOSTS), key=lambda r:r["config_id"])
        names = regime_basis(rows[0]["shape"], rows[0]["family"])[0]
        vectors = [regime_basis(r["shape"], r["family"])[1] for r in rows]
        counts = Counter(tuple(r["shape"]) for r in rows)
        result = max_branch_fit(vectors, [r["observed_us"] for r in rows],
                                [1/counts[tuple(r["shape"])] for r in rows],
                                retained_fraction=retained_fraction)
        fused = {**baseline.models["fused"], **result, "names":names,
                 "feature_bounds":[[min(v[j] for v in vectors), max(v[j] for v in vectors)]
                                   for j in range(len(names))]}
        fused.pop("enumerated_faces", None)
        return cls(f"retained_{retained_fraction:g}", baseline, fused, float(retained_fraction))

    def predict(self, shape, family, *, context_id, dependency_policy=None):
        result = self.baseline.predict(shape, family, context_id=context_id, dependency_policy=dependency_policy)
        if family in HOSTS:
            return result
        names, values = regime_basis(shape, family)
        potential = {n:v*c for n,v,c in zip(names, values, self.fused["coefficients_us"])}
        l, w = potential[names[-2]], potential[names[-1]]
        terms = dict(potential)
        # At a tie, retain the L term once and rho times W: (1+rho)*L.
        terms[names[-1] if l >= w else names[-2]] *= self.retained_fraction
        estimate = math.fsum(terms.values())
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError("invalid retained-proxy prediction")
        result.update(predicted_us=estimate, contributions_us=terms,
            feature_extrapolation=[n for n,v,(lo,hi) in zip(names,values,self.fused["feature_bounds"])
                                   if v < lo-1e-12 or v > hi+1e-12],
            coefficient_identification="rank_deficient" if self.fused["unregularized_rank"] < len(names)
                else "full_column_rank_not_causal_identification",
            regime={"composition":"common_plus_max_plus_retained_min", "retained_fraction":self.retained_fraction,
                    "latency_proxy_us":l, "throughput_proxy_us":w,
                    "retained_smaller_proxy_us":self.retained_fraction*min(l,w),
                    "dominant_proxy":"tie" if abs(l-w) <= 1e-7 else "latency" if l>w else "throughput",
                    "branch_margin_us":abs(l-w), "W_over_L":values[-1]/values[-2],
                    "physical_bottleneck_identified":False, "physical_overlap_fraction_identified":False})
        return result

    def to_dict(self):
        result = super().to_dict()
        result.update(schema="edmir.retained-regime.v1", retained_fraction=self.retained_fraction,
                      retained_fraction_selection="fixed declared grid; development evidence only")
        return result
