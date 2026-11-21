#!/usr/bin/env python3
"""Fit predict.py parameters from microbenchmark JSON Lines. Offline; no GPU commands.

  python3 calibrate.py --base ../params/scenario.json --microbench microbench.jsonl --out params.json

Each JSONL row has a "kind" and positive "seconds":
  F_direct, G_launch, F_staged, lt_launch, sem_per_slice   scalar; median of rows
  b_in       peers, ctas, bytes        -> table of median bytes/seconds per (peers, ctas)
  a2a        bytes_in_per_rank         -> least squares seconds = latency + bytes/bw
  allgather  bytes_in_per_rank         -> same fit
  hbm        bytes                     -> median of bytes/seconds over the largest half of transfers
  mma        flops, ctas, g            -> median flops/seconds over rows with ctas >= SMs*occupancy[g]
Every required kind must be present; otherwise the script fails and names what is missing.
Constants that are not measured (SM count, tile sizes, occupancy, L2 size, partial bytes)
are copied from --base, and the output says so.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCALARS = ("F_direct", "G_launch", "F_staged", "lt_launch", "sem_per_slice")
REQUIRED = ("F_direct", "G_launch", "F_staged", "b_in", "a2a", "hbm", "mma")


def rows(path):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                s = r.get("seconds")
                if type(s) not in (int, float) or not math.isfinite(s) or s <= 0:
                    raise ValueError(f"{path}:{n}: seconds must be positive and finite")
                yield r


def fit_line(points):
    """points: (bytes, seconds). Returns latency, bandwidth."""
    if len(points) < 2 or len({b for b, _ in points}) < 2:
        raise ValueError("a collective fit needs at least two distinct message sizes")
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    if slope <= 0:
        raise ValueError("collective time does not grow with message size; check the microbenchmark")
    return max(0.0, my - slope * mx), 1.0 / slope


def calibrate(base, records):
    by_kind = defaultdict(list)
    for r in records:
        by_kind[r["kind"]].append(r)
    missing = [k for k in REQUIRED if not by_kind[k]]
    if missing:
        raise ValueError("missing microbenchmark kinds: " + ", ".join(missing))
    p = json.loads(json.dumps(base))
    p["source"] = "microbench"
    p["copied_from_base"] = ["sm_count", "tile_m", "tile_n_base", "occupancy", "l2_bytes", "splitk.partial_bytes"]
    for k in SCALARS:
        if by_kind[k]:
            v = statistics.median(r["seconds"] for r in by_kind[k])
            if k == "sem_per_slice":
                p["splitk"]["sem_per_slice"] = v
            else:
                p[k] = v
        else:
            p.setdefault("copied_from_base", []).append(k)
    table = defaultdict(list)
    for r in by_kind["b_in"]:
        table[(int(r["peers"]), int(r["ctas"]))].append(r["bytes"] / r["seconds"])
    p["b_in"] = {"table": [[pe, c, statistics.median(v)] for (pe, c), v in sorted(table.items())]}
    for name in ("a2a", "allgather"):
        if by_kind[name]:
            lat, bw = fit_line([(r["bytes_in_per_rank"], r["seconds"]) for r in by_kind[name]])
            p[name] = {"latency": lat, "bw": bw}
        else:
            p["copied_from_base"].append(name)
    big = sorted(by_kind["hbm"], key=lambda r: r["bytes"])[len(by_kind["hbm"]) // 2:]
    p["b_hbm"] = statistics.median(r["bytes"] / r["seconds"] for r in big)
    full = [r for r in by_kind["mma"] if r["ctas"] >= p["sm_count"] * p["occupancy"][f"g{r['g']}"]]
    if not full:
        raise ValueError("mma rows need at least one launch that fills every SM slot")
    p["mma_peak"] = statistics.median(r["flops"] / r["seconds"] for r in full)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--microbench", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    params = calibrate(base, list(rows(args.microbench)))
    args.out.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}; copied from base: {', '.join(params['copied_from_base'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
