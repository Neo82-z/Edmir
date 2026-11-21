#!/usr/bin/env python3
"""Whole-model upper bound from native attribution and measured first-projection speedups.

  python3 amdahl.py --attr native_attr.jsonl --units analysis/unit_effects.csv --out analysis/amdahl.csv

native_attr.jsonl, one row per measured step of one run:
  width_id, phase ("PREFILL" | "DECODE"), run (int), instrumented (bool), step_wall_ms,
  and on instrumented rows: replaceable_ms = sum over routed layers of the non-overlapped
  critical-path time of dispatch + gate/up (events on the actual streams).
f = median over instrumented steps of replaceable_ms / step_wall_ms.
perturbation = median instrumented step_wall_ms / median uninstrumented step_wall_ms - 1.
bound(s) = 1 / (1 - f + f / s), with s = the measured snapshot speedup of the same phase.
This is an upper bound under a separable model, not an end-to-end measurement; report it apart from TPOT.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_csv, write_csv  # noqa: E402

SPEEDUPS = ("R_Bb_q1", "R_Bb_q4", "R_Bb_j8", "R_native_vs_bnew", "R_flux_vs_native")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inst, plain = defaultdict(list), defaultdict(list)
    with args.attr.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["width_id"], r["phase"])
            if r["step_wall_ms"] <= 0:
                raise ValueError(f"{args.attr}:{n}: step_wall_ms must be positive")
            if r["instrumented"]:
                if not 0 <= r["replaceable_ms"] <= r["step_wall_ms"]:
                    raise ValueError(f"{args.attr}:{n}: replaceable_ms must lie in [0, step_wall_ms]")
                inst[key].append(r)
            else:
                plain[key].append(r["step_wall_ms"])

    speed = {}
    for u in read_csv(args.units):
        if u["comparison_id"] in SPEEDUPS and u["stage"] == "snapshot" and u["unit_class"] != "incomplete":
            width, phase = u["unit_id"].rsplit("_", 1)
            speed[(width, phase, u["comparison_id"], u["candidate_arm"])] = float(u["median_ratio"])

    rows = []
    for key in sorted(inst):
        f = statistics.median(r["replaceable_ms"] / r["step_wall_ms"] for r in inst[key])
        pert = (statistics.median(r["step_wall_ms"] for r in inst[key]) / statistics.median(plain[key]) - 1
                if plain[key] else None)
        for (width, phase, cid, arm), s in sorted(speed.items()):
            if (width, phase) != key:
                continue
            bound = 1.0 / (1.0 - f + f / s)
            rows.append(dict(width_id=width, phase=phase, f=f"{f:.6f}",
                             perturbation="" if pert is None else f"{pert:.6f}", comparison_id=cid,
                             candidate_arm=arm, segment_speedup=f"{s:.6f}", whole_model_bound=f"{bound:.6f}",
                             label="upper bound, separable model; not measured end-to-end"))
        if not any(r["width_id"] == key[0] and r["phase"] == key[1] for r in rows):
            rows.append(dict(width_id=key[0], phase=key[1], f=f"{f:.6f}",
                             perturbation="" if pert is None else f"{pert:.6f}", comparison_id="", candidate_arm="",
                             segment_speedup="", whole_model_bound="", label="no complete snapshot speedup"))
    write_csv(args.out, ("width_id", "phase", "f", "perturbation", "comparison_id", "candidate_arm",
                         "segment_speedup", "whole_model_bound", "label"), rows)
    for r in rows:
        print(f"{r['width_id']} {r['phase']}: f={r['f']} perturbation={r['perturbation'] or '-'} "
              f"{r['comparison_id'] or ''} s={r['segment_speedup'] or '-'} bound={r['whole_model_bound'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
