#!/usr/bin/env python3
"""Explain candidate execution and uncertainty locally; no fitting or GPU access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.execution_ledger import FAMILIES, execution_ledgers
from edmir.compiler.evidence_selection import assess_candidates
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS


def explain(shape, *, environment_id, families=FAMILIES, metadata_mode="legacy",
            baseline_id=PER_SOURCE_CUTLASS.baseline_id):
    ledgers = execution_ledgers(shape, families, environment_id=environment_id,
                                metadata_mode=metadata_mode, baseline_id=baseline_id)
    return {"schema": "edmir.execution-explanation.v1", "ledgers": ledgers,
            "decision": assess_candidates(ledgers)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=int, default=4)
    for name in ("m", "k", "n"):
        p.add_argument("--"+name, type=int, required=True)
    p.add_argument("--environment", required=True)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    p.add_argument("--metadata-mode", choices=("legacy", "prepared"), default="legacy")
    p.add_argument("--baseline", default=PER_SOURCE_CUTLASS.baseline_id)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    result = explain((args.p, args.m, args.k, args.n), environment_id=args.environment,
                     families=args.families, metadata_mode=args.metadata_mode, baseline_id=args.baseline)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    if args.out:
        with args.out.open("x") as output:
            output.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
