#!/usr/bin/env python3
"""Explain candidate execution and uncertainty locally; no fitting or GPU access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.execution_ledger import FAMILIES, ALL_FAMILIES, execution_ledgers
from edmir.compiler.evidence_selection import assess_candidates
from edmir.compiler.execution_policy import POLICIES, LEGACY_POLICY


def explain(shape, *, environment_id, families=FAMILIES, metadata_mode="legacy",
            baseline_id=None, dependency_policy=LEGACY_POLICY, algorithm_ids=None):
    ledgers = execution_ledgers(shape, families, environment_id=environment_id,
                                metadata_mode=metadata_mode, baseline_id=baseline_id,
                                dependency_policy=dependency_policy, algorithm_ids=algorithm_ids)
    return {"schema": "edmir.execution-explanation.v1", "ledgers": ledgers,
            "decision": assess_candidates(ledgers)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=int, default=4)
    for name in ("m", "k", "n"):
        p.add_argument("--"+name, type=int, required=True)
    p.add_argument("--environment", required=True)
    p.add_argument("--families", nargs="+", choices=ALL_FAMILIES, default=FAMILIES)
    p.add_argument("--metadata-mode", choices=("legacy", "prepared"), default="legacy")
    p.add_argument("--baseline", default=None)
    p.add_argument("--dependency-policy", choices=POLICIES, default=LEGACY_POLICY,
                   help="applies to fused candidates only; host keeps its own schedule")
    p.add_argument("--cublaslt-algorithm-id", help="exact per-rank identity recorded by the benchmark, if available")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    result = explain((args.p, args.m, args.k, args.n), environment_id=args.environment,
                     families=args.families, metadata_mode=args.metadata_mode, baseline_id=args.baseline,
                     dependency_policy=args.dependency_policy,
                     algorithm_ids=({'host_merged_cublaslt': args.cublaslt_algorithm_id}
                                    if args.cublaslt_algorithm_id else None))
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if args.out:
        with args.out.open("x") as output:
            output.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
