#!/usr/bin/env python3
"""Post-fit representation and loss audit; no additional model alternatives."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import run_audit as audit
from candidate_model import DecisionModel, split_vector
from edmir.compiler.factorized_selection import features
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.measurement_dataset import load_identified_selector_dataset


def collision_cells(data, component, interactions=False):
    cells = defaultdict(list)
    for surface in data.validate().values():
        for name, row in surface.items():
            if component == "split":
                if row.is_host or row.n_tiles_per_cta != 1 or row.split_k_slices == 1:
                    continue
                vector = split_vector(row, interactions)
                denominator = surface["direct"].measured_ms
            else:
                if row.n_tiles_per_cta != 2:
                    continue
                vector = features(row, "group")
                denominator = surface[name.removesuffix("_groupedn2")].measured_ms
            cells[tuple(vector)].append({
                "shape": list(row.shape), "family": name,
                "ratio": row.measured_ms/denominator,
            })
    collisions = []
    for vector, rows in cells.items():
        if len(rows) < 2:
            continue
        values = [r["ratio"] for r in rows]
        half_span = math.log(max(values)/min(values))/2
        collisions.append({
            "features": list(vector), "rows": rows,
            "unavoidable_max_abs_log_error_at_least": half_span,
            "unavoidable_multiplicative_error_factor_at_least": math.exp(half_span),
        })
    return sorted(collisions,
                  key=lambda c: c["unavoidable_max_abs_log_error_at_least"], reverse=True)


def main(out):
    if out.exists():
        raise ValueError("choose a fresh output")
    data, _, _, _ = load_identified_selector_dataset(
        audit.ROOT/"experiments/2026-10-20-offline-predictor/data")
    surfaces = data.validate()
    folds, priority = [], []
    for group in sorted({r.group_id for r in data.rows}):
        train = replace(data, rows=tuple(r for r in data.rows if r.group_id != group))
        test = replace(data, rows=tuple(r for r in data.rows if r.group_id == group))
        model = DecisionModel(train, joint=True, interactions=True)
        for surface in test.validate().values():
            first = next(iter(surface.values()))
            candidates = {n: ScheduleCandidate.from_measurement(r, data.environment_id)
                          for n, r in surface.items()}
            chosen, scores = model.choose(candidates)
            excluded = {n: model.reference.exclusion_reason(r) for n, r in candidates.items()
                        if n not in scores}
            best_all = min(surface, key=lambda n: surface[n].measured_ms)
            best_eligible = min(scores, key=lambda n: surface[n].measured_ms)
            row = {
                "shape": list(first.shape), "fold": group, "choice": chosen,
                "best_observed": best_all, "best_eligible": best_eligible,
                "observed_count": len(surface), "eligible_count": len(scores),
                "excluded": excluded,
                "regret_all_observed": surface[chosen].measured_ms/surface[best_all].measured_ms-1,
                "regret_eligible": surface[chosen].measured_ms/surface[best_eligible].measured_ms-1,
            }
            folds.append(row)
            if first.shape in audit.PRIORITY:
                priority.append(row)
    result = {
        "schema": "edmir.selector-structural-review.v1",
        "interpretation": "Diagnostic after seeing control results, not another validation set or tuned model.",
        "collisions": {
            "existing_split": collision_cells(data, "split"),
            "extended_split": collision_cells(data, "split", True),
            "group_component": collision_cells(data, "group"),
        },
        "coverage": {
            "workloads": len(surfaces), "measurement_rows": len(data.rows),
            "grouped_m_values": sorted({r.rows_per_peer for r in data.rows if r.n_tiles_per_cta == 2}),
            "candidate_count_workloads": dict(Counter(len(s) for s in surfaces.values())),
            "priority": priority, "all": folds,
        },
        "pairwise_loss": {
            "identity": "sum_(a<b)(e_a-e_b)^2 / C(n,2) = 2/(n-1) * sum_a(e_a-mean(e))^2",
            "interpretation": "Exact loss identity, not an accuracy guarantee. Pair constraints are correlated.",
            "by_candidate_count": [{
                "candidates": n, "pairs": math.comb(n, 2),
                "independent_within_workload_log_contrasts": n-1,
                "explicit_host_pair_weight_fraction": 2/n,
            } for n in sorted({len(s) for s in surfaces.values()})],
            "limitations": [
                "Equal total workload weight does not mean equal host-vs-fused pair weight.",
                "Average squared log-ratio error is not decision regret or worst-case loss.",
                "The same ridge number does not imply equal regularization across feature designs.",
            ],
        },
        "server_ready": False,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({
        "collision_summary": {
            name: {"cells": len(cells), "worst": cells[0] if cells else None}
            for name, cells in result["collisions"].items()
        },
        "coverage": {k: v for k, v in result["coverage"].items() if k != "all"},
        "pairwise_loss": result["pairwise_loss"],
    }, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
