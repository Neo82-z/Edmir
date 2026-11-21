#!/usr/bin/env python3
"""Two-channel max model for direct (peer-read) vs staged first-projection time.

Predictions are written before the matching timing blocks run, then sealed:

  python3 predict.py --params params.json --design ../design --out pred_cells
  python3 predict.py --params params.json --design ../design --routes route_stats.json --out pred_snapshots

Model (per destination GPU; the slowest destination sets the 8-GPU completion time):

  kernel(problems) = max(t_remote, t_hbm, t_mma) + t_splitk
    t_remote = sum_remote(rows*K*B) * replay / b_in(peers, active CTAs)    replay = ceil(N / (128*g))
    t_hbm    = (weights read from HBM + local activations + outputs + split-K partials) / b_hbm
    t_mma    = FLOPs / (mma_peak * min(1, CTAs / (SMs * occupancy[g])))
  J1 direct    = F_direct + sum over source launches (kernel + G_launch)
  J8 direct    = F_direct + G_launch + kernel(all sources in one launch)
  staged, same consumer = F_staged + a2a(bytes in) + sum over source launches (kernel on local copies + G_launch)
  SLT          = F_staged + a2a(bytes in) + lt_launch + max(HBM bytes / b_hbm, FLOPs / mma rate)
  NATIVE       = F_staged + allgather(bytes in) + lt_launch + fused grouped GEMM over local experts

Byte counts are kept apart: direct reads per (source, expert) assignment and multiplies by replay;
the same-consumer staged arm moves each assignment once; NATIVE all-gathers every token.
Arms without a model (historical bridges, FLUX) are left unpredicted, never guessed.
"""

import argparse
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classify, intervals, read_csv, sha256, write_csv  # noqa: E402

ELEM = 2  # BF16/FP16 bytes


def ceil_div(a, b):
    return -(-a // b)


class Machine:
    def __init__(self, p):
        self.p = p
        self.sm = p["sm_count"]
        self.tile_m = p["tile_m"]
        self.tile_n_base = p["tile_n_base"]

    def occupancy(self, g):
        return self.p["occupancy"][f"g{g}"]

    def b_in(self, peers, ctas):
        """Effective per-destination ingress bandwidth for remote reads."""
        spec = self.p["b_in"]
        if peers <= 0 or ctas <= 0:
            return math.inf
        if "table" in spec:
            rows = spec["table"]
            peer_set = sorted({r[0] for r in rows})
            near = min(peer_set, key=lambda x: (abs(x - peers), x))
            curve = sorted((r[1], r[2]) for r in rows if r[0] == near)
            xs = [c for c, _ in curve]
            if ctas <= xs[0]:
                return curve[0][1] * ctas / xs[0]
            if ctas >= xs[-1]:
                return curve[-1][1]
            i = bisect_left(xs, ctas)
            (x0, y0), (x1, y1) = curve[i - 1], curve[i]
            w = (math.log(ctas) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + w * (y1 - y0)
        cap = spec["max_by_peers"]
        key = str(min((int(k) for k in cap), key=lambda x: (abs(x - peers), x)))
        return min(spec["per_cta"] * ctas, cap[key])

    def mma_rate(self, ctas, g):
        slots = self.sm * self.occupancy(g)
        return self.p["mma_peak"] * min(1.0, ctas / slots)

    def collective(self, name, bytes_in):
        c = self.p[name]
        return c["latency"] + bytes_in / c["bw"]


def kernel_time(mc, problems, k, n, g, q, weight_hbm_bytes):
    """problems: list of (rows, remote: bool). One launch."""
    p = mc.p
    tile_n = mc.tile_n_base * g
    replay = ceil_div(n, tile_n)
    tiles = sum(ceil_div(r, mc.tile_m) * replay for r, _ in problems if r > 0)
    if tiles == 0:
        return 0.0
    ctas = tiles * q
    active = min(ctas, mc.sm * mc.occupancy(g))
    remote = sum(r * k * ELEM for r, is_remote in problems if is_remote) * replay
    local = sum(r * k * ELEM for r, is_remote in problems if not is_remote)
    rows = sum(r for r, _ in problems)
    peers = sum(1 for r, is_remote in problems if is_remote and r > 0)
    out = rows * n * ELEM
    partial = (q - 1) * rows * n * p["splitk"]["partial_bytes"] * 2 if q > 1 else 0
    t_remote = remote / mc.b_in(peers, active) if remote else 0.0
    t_hbm = (weight_hbm_bytes + local + out + partial) / p["b_hbm"]
    t_mma = 2.0 * rows * k * n / mc.mma_rate(ctas, g)
    t_sem = p["splitk"]["sem_per_slice"] * (q - 1) if q > 1 else 0.0
    return max(t_remote, t_hbm, t_mma) + t_sem


def j1_sequence(mc, launches, k, n, g, q, weight_bytes, fixed):
    """launches: list over sources of list of (rows, remote, expert). Weights stay in L2 only if they fit."""
    p = mc.p
    active_experts = {e for launch in launches for r, _, e in launch if r > 0}
    fits = sum(weight_bytes[e] for e in active_experts) <= p["l2_bytes"]
    seen, total = set(), fixed
    for launch in launches:
        live = [(r, rem, e) for r, rem, e in launch if r > 0]
        if not live:
            continue
        experts = {e for _, _, e in live}
        fresh = experts - seen if fits else experts
        seen |= experts
        w = sum(weight_bytes[e] for e in fresh)
        total += kernel_time(mc, [(r, rem) for r, rem, _ in live], k, n, g, q, w) + p["G_launch"]
    return total


def j8_single(mc, launches, k, n, g, q, weight_bytes):
    p = mc.p
    live = [(r, rem, e) for launch in launches for r, rem, e in launch if r > 0]
    if not live:
        return p["F_direct"]
    w = sum(weight_bytes[e] for e in {e for _, _, e in live})
    return p["F_direct"] + p["G_launch"] + kernel_time(mc, [(r, rem) for r, rem, _ in live], k, n, g, q, w)


def merged_gemm(mc, rows, k, n, weight_hbm, experts_tiles=None):
    """One staged GEMM over local data (cuBLASLt or a fused grouped GEMM), full-rate when tiles fill the GPU."""
    p = mc.p
    tiles = experts_tiles if experts_tiles is not None else ceil_div(rows, 128) * ceil_div(n, 128)
    rate = p["mma_peak"] * min(1.0, tiles / mc.sm)
    t_hbm = (weight_hbm + rows * k * ELEM + rows * n * ELEM) / p["b_hbm"]
    return p["lt_launch"] + max(t_hbm, 2.0 * rows * k * n / rate if rows else 0.0)


ARM_SPECS = {
    # arm: (family, g, q)
    "D_Q1_G2": ("j1_direct", 2, 1), "D_Q4_G2": ("j1_direct", 2, 4), "D_Q1_G1": ("j1_direct", 1, 1),
    "BNEW_Q1_G2": ("j1_staged", 2, 1), "BNEW_Q4_G2": ("j1_staged", 2, 4),
    "J8_Q1_G2": ("j8_direct", 2, 1), "SLT": ("slt", None, None),
    "D_Q1_G2_R": ("j1_direct", 2, 1), "D_Q4_G2_R": ("j1_direct", 2, 4),
    "BNEW_Q1_G2_R": ("j1_staged", 2, 1), "BNEW_Q4_G2_R": ("j1_staged", 2, 4),
    "J8_Q1_G2_R": ("j8_direct", 2, 1), "NATIVE": ("native", None, None),
}


def predict_destination(mc, arm, rows_sde, tokens_per_source, d, k, n, ranks):
    """rows_sde[s][e] = rows routed from source s to local expert e of destination d."""
    p = mc.p
    family, g, q = ARM_SPECS[arm]
    experts = sorted({e for s in range(ranks) for e in rows_sde[s]})
    weight_bytes = {e: k * n * ELEM for e in experts}
    in_bytes = sum(r * k * ELEM for s in range(ranks) if s != d for r in rows_sde[s].values())
    if family in ("j1_direct", "j8_direct"):
        launches = [[(r, s != d, e) for e, r in sorted(rows_sde[s].items())] for s in range(ranks)]
        if family == "j1_direct":
            return j1_sequence(mc, launches, k, n, g, q, weight_bytes, p["F_direct"])
        return j8_single(mc, launches, k, n, g, q, weight_bytes)
    if family == "j1_staged":
        launches = [[(r, False, e) for e, r in sorted(rows_sde[s].items())] for s in range(ranks)]
        return mc.collective("a2a", in_bytes) + j1_sequence(mc, launches, k, n, g, q, weight_bytes, p["F_staged"])
    per_expert = {e: sum(rows_sde[s].get(e, 0) for s in range(ranks)) for e in experts}
    rows = sum(per_expert.values())
    w = sum(weight_bytes[e] for e, r in per_expert.items() if r > 0)
    if family == "slt":
        return p["F_staged"] + mc.collective("a2a", in_bytes) + merged_gemm(mc, rows, k, n, w)
    if family == "native":
        gather = sum(tokens_per_source[s] * k * ELEM for s in range(ranks) if s != d)
        tiles = sum(ceil_div(r, 64) * ceil_div(n, 128) for r in per_expert.values() if r > 0)
        return p["F_staged"] + mc.collective("allgather", gather) + merged_gemm(mc, rows, k, n, w, tiles)
    raise ValueError(arm)


def predict_unit(mc, arm, loads, tokens, k, n, ranks):
    """loads[d][s][e]: rows. Returns seconds for the slowest destination."""
    return max(predict_destination(mc, arm, loads[d], tokens, d, k, n, ranks) for d in range(ranks))


def balanced_loads(m, ranks):
    loads = [[{0: m} for _ in range(ranks)] for _ in range(ranks)]
    return loads, [m * ranks] * ranks  # each source sends m rows to each destination; one expert per destination


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--routes", type=Path, help="route_stats.json; predicts the snapshot units instead of the cells")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    params = json.loads(args.params.read_text(encoding="utf-8"))
    mc = Machine(params)
    policies = json.loads((args.design / "policies.json").read_text(encoding="utf-8"))
    comparisons = read_csv(args.design / "comparisons.csv")
    upper = policies["tie_band_upper"]
    stage = "snapshot" if args.routes else "balanced"

    units = []  # (unit_id, group, x, loads, tokens, k, n, ranks)
    if stage == "balanced":
        for c in read_csv(args.design / "cells.csv"):
            m, ranks = int(c["m_per_source_destination"]), int(c["ranks"])
            loads, tokens = balanced_loads(m, ranks)
            units.append((c["cell_id"], c["width_id"], m, loads, tokens, int(c["k"]), int(c["n"]), ranks))
    else:
        routes = json.loads(args.routes.read_text(encoding="utf-8"))
        snaps = {s["snapshot_id"]: s for s in read_csv(args.design / "snapshots.csv")}
        for sid, r in routes["snapshots"].items():
            s = snaps[sid]
            ranks = r["ranks"]
            loads = [[{int(e): v for e, v in r["rows"][d][src].items()} for src in range(ranks)] for d in range(ranks)]
            units.append((sid, s["width_id"], 0, loads, r["tokens_per_source"], int(s["k"]), int(s["n"]), ranks))

    arms = [a for a in ARM_SPECS if (a.endswith("_R") or a == "NATIVE") == (stage == "snapshot")]
    pred_rows, by_unit = [], {}
    for uid, group, x, loads, tokens, k, n, ranks in units:
        for arm in arms:
            t = predict_unit(mc, arm, loads, tokens, k, n, ranks)
            by_unit.setdefault(uid, {})[arm] = t
            pred_rows.append(dict(unit_id=uid, group=group, x=x, arm_id=arm, predicted_ms=f"{t * 1e3:.6f}"))

    comp_rows, seqs = [], {}
    for c in comparisons:
        if c["stage"] != stage or c["baseline_arm"] not in arms or c["candidate_arm"] not in arms:
            continue
        for uid, group, x, *_ in units:
            ratio = by_unit[uid][c["baseline_arm"]] / by_unit[uid][c["candidate_arm"]]
            cls = classify(ratio, upper)
            comp_rows.append(dict(comparison_id=c["comparison_id"], unit_id=uid, group=group, x=x,
                                  predicted_ratio=f"{ratio:.6f}", predicted_class=cls))
            if c["crossover"] == "1":
                seqs.setdefault((c["comparison_id"], group), []).append((x, cls))
    cross_rows = []
    for (cid, group), seq in sorted(seqs.items()):
        found = intervals(sorted(seq))
        if not found:
            cross_rows.append(dict(comparison_id=cid, group=group, kind="none", m_low="", m_high="", direction=""))
        for kind, lo, hi, direction in found:
            cross_rows.append(dict(comparison_id=cid, group=group, kind=kind, m_low=lo, m_high=hi, direction=direction))

    cfg = policies[stage]
    choice_rows = []
    for uid, *_ in units:
        options = {a: by_unit[uid][a] for a in cfg["selection_set"] if a in by_unit[uid]}
        if len(options) == len(cfg["selection_set"]):
            choice_rows.append(dict(unit_id=uid, policy="P0", arm_id=min(options, key=options.get)))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "predictions.csv", ("unit_id", "group", "x", "arm_id", "predicted_ms"), pred_rows)
    write_csv(args.out / "predicted_comparisons.csv",
              ("comparison_id", "unit_id", "group", "x", "predicted_ratio", "predicted_class"), comp_rows)
    write_csv(args.out / "predicted_crossover.csv",
              ("comparison_id", "group", "kind", "m_low", "m_high", "direction"), cross_rows)
    write_csv(args.out / "policy_choice.csv", ("unit_id", "policy", "arm_id"), choice_rows)
    inputs = [args.params, args.design / "policies.json", args.design / "comparisons.csv",
              args.design / ("cells.csv" if stage == "balanced" else "snapshots.csv"), Path(__file__),
              Path(__file__).with_name("common.py")]
    if args.routes:
        inputs.append(args.routes)
    seal = dict(stage=stage, params_source=params.get("source", "unspecified"),
                inputs={(p.name if p.is_absolute() else str(p)): sha256(p) for p in inputs},
                outputs={name: sha256(args.out / name) for name in
                         ("predictions.csv", "predicted_comparisons.csv", "predicted_crossover.csv", "policy_choice.csv")})
    (args.out / "PREDICTION_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{stage}: {len(units)} units x {len(arms)} arms predicted; params source = {seal['params_source']}; "
          f"sealed in {args.out / 'PREDICTION_SEAL.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
