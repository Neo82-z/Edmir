#!/usr/bin/env python3
"""Diagnose whether ANY single host threshold could fix priority directions.

This uses evaluation labels to derive necessary inequalities. It neither fits
a threshold nor changes inference, and is not an additional model variant.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
from evaluate_execution_cost_selector import PRIORITY
from edmir.compiler.measurement_dataset import sha256


def main(results, out):
    if out.exists():
        raise ValueError("choose a fresh output")
    data = json.loads(results.read_text())
    constraints = []
    for row in data["primary"]["decisions"]:
        if tuple(row["shape"]) not in PRIORITY:
            continue
        scores = row["scores"]["execution_balanced"]
        fused = [n for n in scores if n != "host"]
        best = min(fused, key=lambda n: (scores[n], n))
        times = row["times_ms"]
        best_actual = min(times[n] for n in fused)
        if times["host"] > 1.05*best_actual:
            relation, desired = "threshold > score", "fused"
        elif best_actual > 1.05*times["host"]:
            relation, desired = "threshold <= score", "host"
        else:
            continue
        constraints.append({"shape": row["shape"], "desired_direction": desired,
                            "best_fused_predicted": best, "score": scores[best],
                            "necessary_condition": relation,
                            "best_observed_fused_over_host": best_actual/times["host"]})
    lower = max((c for c in constraints if c["desired_direction"] == "fused"), key=lambda c: c["score"])
    upper = min((c for c in constraints if c["desired_direction"] == "host"), key=lambda c: c["score"])
    result = {"schema": "edmir.host-threshold-obstruction.v1", "rule": "choose fused iff best_fused_score < threshold",
              "constraints": constraints, "strict_lower_bound": lower, "inclusive_upper_bound": upper,
              "single_threshold_possible": lower["score"] < upper["score"],
              "use": "retrospective diagnosis only; no threshold tuning or inference change",
              "results_sha256": sha256(results), "source_sha256": sha256(Path(__file__))}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.results, args.out)
