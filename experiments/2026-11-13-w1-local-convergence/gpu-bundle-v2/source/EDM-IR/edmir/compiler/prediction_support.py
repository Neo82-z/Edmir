"""Linear feature support diagnostics, never confidence or a target label filter."""
from __future__ import annotations
import math

from .mechanism_prediction import matrix_rank, MAX_ROWS, MAX_FEATURES


def linear_target_support(training_vectors, target_vector):
    if (not training_vectors or len(training_vectors)>MAX_ROWS or not 1<=len(target_vector)<=MAX_FEATURES
            or any(len(r)!=len(target_vector) for r in training_vectors)
            or any(not math.isfinite(v) for r in [*training_vectors,target_vector] for v in r)):
        raise ValueError("bounded finite feature matrix required")
    n=len(training_vectors);d=len(target_vector)
    scales=[max(math.sqrt(math.fsum(r[j]**2 for r in training_vectors)/n),1e-12) for j in range(d)]
    rows=[[v/s for v,s in zip(r,scales)] for r in training_vectors]
    target=[v/s for v,s in zip(target_vector,scales)]
    # Normalize row magnitudes too: span is unchanged and the rank tolerance
    # cannot be dominated by a very large extrapolating target.
    normalize=lambda r:[v/max(1.,max(abs(x) for x in r)) for v in r]
    rows=[normalize(r) for r in rows];target=normalize(target)
    rank=matrix_rank(rows);augmented=matrix_rank(rows+[target])
    outside=[j for j,v in enumerate(target_vector) if v<min(r[j] for r in training_vectors)-1e-12
             or v>max(r[j] for r in training_vectors)+1e-12]
    return {"training_rank":rank,"rank_with_target":augmented,"feature_count":d,
            "target_in_training_row_span":augmented==rank,"outside_training_feature_indices":outside,
            "interpretation":"Unconstrained linear identifiability only. Nonnegative constraints/ridge can pick coefficients but do not add observations. Full rank is not evidence of causal or extrapolation accuracy.",
            "confidence_probability":None}
