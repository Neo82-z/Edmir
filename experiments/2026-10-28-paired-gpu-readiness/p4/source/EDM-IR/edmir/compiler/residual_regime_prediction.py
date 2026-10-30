"""One declared finite-depth revision, preserving the earlier sealed models."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from .mechanism_prediction import HOSTS
from .regime_prediction import RegimeModel, max_branch_fit, regime_basis


def residual_basis(shape, family):
    names, values = regime_basis(shape, family)
    # Same dimensionality; a common depth contribution persists even when W
    # dominates the maximum. It is a hypothesis about incomplete hiding, not
    # a separately measured pipeline delay. Never a shape-specific correction.
    return names[:3]+("unhidden_partition_depth_64",)+names[4:], values[:3]+(values[-2],)+values[4:]


@dataclass(frozen=True)
class ResidualRegimeModel(RegimeModel):
    @classmethod
    def fit(cls, configs, *, variant="residual_depth", include_flagged=False):
        if variant != "residual_depth":
            raise ValueError("only the declared residual-depth revision is supported")
        parent = RegimeModel.fit(configs, variant="regime_max", include_flagged=include_flagged)
        rows = sorted((r for r in configs if r["family"] not in HOSTS), key=lambda r:r["config_id"])
        count = Counter(tuple(r["shape"]) for r in rows)
        vectors = [residual_basis(r["shape"],r["family"])[1] for r in rows]
        result = max_branch_fit(vectors,[r["observed_us"] for r in rows],
                                [1/count[tuple(r["shape"])] for r in rows])
        fused = {**parent.fused, **result, "names":residual_basis(rows[0]["shape"],rows[0]["family"])[0],
            "feature_bounds":[[min(r[j] for r in vectors),max(r[j] for r in vectors)] for j in range(8)]}
        return cls(variant,parent.baseline,fused)

    def predict(self,shape,family,*,context_id,dependency_policy=None):
        result = self.baseline.predict(shape,family,context_id=context_id,dependency_policy=dependency_policy)
        if family in HOSTS:
            if family in ("host_compiler_cublas","host_merged_cublaslt"):
                result["algorithm_support"] = {
                    "status":"library_policy_proxy_only", "target_algorithm_id":None,
                    "algorithm_specific_latency_identified":False,
                    "required_evidence":"Record pre-timing algorithm selection and matched-algorithm shape sweeps; never infer algorithm from the held-out latency."}
            return result
        names,values=residual_basis(shape,family)
        potential={n:v*c for n,v,c in zip(names,values,self.fused["coefficients_us"])}
        terms=dict(potential)
        l,w=potential[names[-2]],potential[names[-1]]
        terms[names[-1] if l>=w else names[-2]]=0.
        result.update(predicted_us=math.fsum(terms.values()),contributions_us=terms,
            feature_extrapolation=[n for n,v,(lo,hi) in zip(names,values,self.fused["feature_bounds"])
                                   if v<lo-1e-12 or v>hi+1e-12],
            coefficient_identification="rank_deficient" if self.fused["unregularized_rank"]<8 else "full_column_rank_not_causal_identification",
            regime={"composition":"common_plus_unhidden_depth_plus_max",
                "unhidden_depth_proxy_us":potential["unhidden_partition_depth_64"],
                "latency_proxy_us":l,"throughput_proxy_us":w,
                "dominant_proxy":"tie" if abs(l-w)<=1e-7 else "latency" if l>w else "throughput",
                "W_over_L":values[-1]/values[-2],"physical_bottleneck_identified":False})
        if not math.isfinite(result["predicted_us"]) or result["predicted_us"]<=0:
            raise ValueError("invalid residual-depth prediction")
        return result
