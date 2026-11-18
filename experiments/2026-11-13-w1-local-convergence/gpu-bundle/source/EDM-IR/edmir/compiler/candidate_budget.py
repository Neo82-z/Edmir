"""Retrospective shortlist evaluation; no fitting or compiler admission.

Model ranking reads predictions only. Observations enter the separate scorer.
Scoring the best observed member of a shortlist is an ideal replay, not a
measurement-noise-aware deployment simulation.
"""
from __future__ import annotations

import math
from statistics import fmean


def validate(rows):
    if not rows or len(rows) > 256:
        raise ValueError("candidate count must be 1..256")
    ids = [r["candidate_id"] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid or duplicate candidate identity")
    for row in rows:
        if row.get("quality_flags") or row.get("eligible_for_calibration") is False:
            raise ValueError("explicit stable subset required")
        if not math.isfinite(row["observed_us"]) or row["observed_us"] <= 0:
            raise ValueError("positive finite observations required")


def rank_predictions(rows):
    # Deliberately do not read observed_us: ranking must not use test labels.
    ids = [r["candidate_id"] for r in rows]
    if not ids or len(ids) > 256 or len(set(ids)) != len(ids):
        raise ValueError("invalid candidate identities")
    if any(not math.isfinite(r["predicted_us"]) or r["predicted_us"] <= 0 for r in rows):
        raise ValueError("positive finite predictions required")
    return [r["candidate_id"] for r in sorted(rows, key=lambda r: (r["predicted_us"], r["candidate_id"]))]


def shortlist(order, budget, *, protected=None):
    if type(budget) is not int or not 1 <= budget <= 256:
        raise ValueError("budget must be an integer in 1..256")
    if not order or len(set(order)) != len(order):
        raise ValueError("nonempty unique candidate order required")
    if protected is not None and protected not in order:
        raise ValueError("protected candidate unavailable; do not invent its label")
    if protected is None:
        return list(order[:budget])
    return [protected] + [c for c in order if c != protected][:budget-1]


def score(rows, selected, *, near=0.05):
    validate(rows)
    if not math.isfinite(near) or near < 0:
        raise ValueError("invalid near-optimal threshold")
    costs = {r["candidate_id"]: r["observed_us"] for r in rows}
    if not selected or len(set(selected)) != len(selected) or set(selected)-costs.keys():
        raise ValueError("shortlist must be a nonempty subset of observed candidates")
    best = min(costs.values())
    chosen = min(selected, key=lambda c: (costs[c], c))
    loss = costs[chosen]/best - 1
    return {"selected_candidates": list(selected), "ideal_measured_subset_choice": chosen,
            "observed_candidates": len(costs), "evaluated_candidates": len(selected),
            "choice_loss_fraction": loss, "near_optimal_retained": loss <= near,
            "candidate_count_reduction": 1-len(selected)/len(costs),
            "full_candidate_oracle": False, "deployment_policy_validated": False}


def uniform_random_reference(rows, budget, *, near=0.05):
    """Exact expected minimum of a uniform k-subset, without enumeration."""
    validate(rows)
    if type(budget) is not int or not 1 <= budget <= 256 or not math.isfinite(near) or near < 0:
        raise ValueError("invalid budget or threshold")
    times = sorted(r["observed_us"] for r in rows)
    n = len(times); k = min(n, budget); total = math.comb(n, k)
    # Candidate i is the earliest member of C(n-i-1,k-1) subsets.
    expected = math.fsum(t*math.comb(n-i-1, k-1)/total for i,t in enumerate(times[:n-k+1]))
    good = sum(t/times[0]-1 <= near for t in times)
    miss = math.comb(n-good, k)/total if n-good >= k else 0.
    return {"observed_candidates": n, "evaluated_candidates": k,
            "expected_choice_loss_fraction": max(0., expected/times[0]-1),
            "near_optimal_retention_probability": 1-miss,
            "candidate_count_reduction": 1-k/n,
            "expectation_is_over_random_subsets_not_model_confidence": True}


def training_only_order(rows, held_out_shape):
    """Fixed candidate ordering using only candidates present in EVERY train shape."""
    if any(tuple(r["shape"]) == tuple(held_out_shape) for r in rows):
        raise ValueError("held-out labels must not enter fixed baseline")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row["shape"]), []).append(row)
    if not groups:
        raise ValueError("no training shapes")
    for group in groups.values():
        validate(group)
    common = set.intersection(*({r["candidate_id"] for r in group} for group in groups.values()))
    scores = {}
    for candidate in common:
        ratios = []
        for group in groups.values():
            costs = {r["candidate_id"]: r["observed_us"] for r in group}
            ratios.append(math.log(costs[candidate]/min(costs.values())))
        scores[candidate] = fmean(ratios)
    return {"order": sorted(common, key=lambda c:(scores[c],c)),
            "mean_log_relative_cost": scores, "training_shapes": sorted(groups),
            "eligibility": "observed in every training shape; test availability is not imputed"}
