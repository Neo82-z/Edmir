#!/usr/bin/env python3
"""Per-destination routing loads from captured vLLM router records. Offline only.

Input: one JSON Lines file per rank, plain or gzip, named <block>-rank<r>.jsonl[.gz], in the
format written by the 2026-11-17 collector:
  {"kind": "step",  "step": S, "rank": r, "category": "pure_prefill" | "pure_decode" | ...}
  {"kind": "route", "step": S, "rank": r, "layer": L, "topk_ids": [[e, ...], ...]}
Experts are placed statically: expert e lives on rank e // (experts / ranks).

  python3 route_stats.py --dir captures/GLM --block diag-S32 --experts 64 --ranks 8 \
      --snapshot GLM_PREFILL:pure_prefill:1:23 --snapshot GLM_DECODE:pure_decode:8:23 \
      --aggregate pure_prefill --out route_stats_GLM.json

A step counts for a category only if every rank reports that category for it.
--snapshot ID:category:ordinal:layer picks the ordinal-th such step (1-based, by step number).
--merge combines several per-model outputs into one file for predict.py --routes.
"""

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def open_any(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def rank_files(directory, block, ranks):
    files = []
    for r in range(ranks):
        hits = [p for p in (directory / f"{block}-rank{r}.jsonl.gz", directory / f"{block}-rank{r}.jsonl") if p.exists()]
        if len(hits) != 1:
            raise FileNotFoundError(f"need exactly one route file for rank {r} of block {block} in {directory}")
        files.append(hits[0])
    return files


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(files, experts, ranks, wanted_layers=None):
    per_rank = experts // ranks
    if per_rank * ranks != experts:
        raise ValueError("experts must divide evenly across ranks")
    categories = defaultdict(dict)                      # step -> rank -> category
    rows = defaultdict(lambda: [[defaultdict(int) for _ in range(ranks)] for _ in range(ranks)])  # (step,layer) -> d -> s -> e
    tokens = defaultdict(lambda: [0] * ranks)            # (step,layer) -> tokens per source
    distinct = defaultdict(list)                         # (step,layer) -> distinct remote ranks per token
    seen = set()
    for src, path in enumerate(files):
        with open_any(path) as f:
            for line in f:
                o = json.loads(line)
                if o.get("rank") != src:
                    raise ValueError(f"{path}: record for rank {o.get('rank')} in the rank-{src} file")
                if o["kind"] == "step":
                    categories[o["step"]][src] = o["category"]
                elif o["kind"] == "route":
                    key = (o["step"], o["layer"])
                    if wanted_layers is not None and o["layer"] not in wanted_layers:
                        continue
                    if (key, src) in seen:
                        raise ValueError(f"{path}: duplicate route record {key}")
                    seen.add((key, src))
                    tokens[key][src] += len(o["topk_ids"])
                    for ids in o["topk_ids"]:
                        dests = set()
                        for e in ids:
                            d = e // per_rank
                            rows[key][d][src][e] += 1
                            dests.add(d)
                        distinct[key].append(len(dests - {src}))
    return categories, rows, tokens, distinct


def steps_of(categories, category, ranks):
    return sorted(s for s, by_rank in categories.items()
                  if len(by_rank) == ranks and all(c == category for c in by_rank.values()))


def stats(rows_d, tokens_d, distinct_d, ranks):
    load = [sum(sum(rows_d[d][s].values()) for s in range(ranks)) for d in range(ranks)]
    mean = sum(load) / ranks
    remote = [sum(sum(rows_d[d][s].values()) for s in range(ranks) if s != d) for d in range(ranks)]
    return dict(dest_load=load, dest_remote_assignments=remote,
                max_over_mean=max(load) / mean if mean else None,
                mean_distinct_remote_ranks=statistics.fmean(distinct_d) if distinct_d else None,
                tokens_per_source=tokens_d)


def run(args):
    files = rank_files(args.dir, args.block, args.ranks)
    layers = {int(spec.split(":")[3]) for spec in args.snapshot} if args.snapshot and not args.aggregate else None
    categories, rows, tokens, distinct = load(files, args.experts, args.ranks, layers)
    out = dict(schema="route-stats.v1", block=args.block, experts=args.experts, ranks=args.ranks,
               source_files={p.name: sha256(p) for p in files}, snapshots={}, aggregate={})
    for spec in args.snapshot:
        sid, category, ordinal, layer = spec.split(":")
        steps = steps_of(categories, category, args.ranks)
        if int(ordinal) > len(steps):
            raise ValueError(f"{sid}: only {len(steps)} {category} steps captured, ordinal {ordinal} requested")
        step = steps[int(ordinal) - 1]
        key = (step, int(layer))
        if key not in rows:
            raise ValueError(f"{sid}: no route records for step {step}, layer {layer}")
        r = rows[key]
        snap = dict(step=step, layer=int(layer), category=category, ordinal=int(ordinal), ranks=args.ranks,
                    rows=[[{str(e): n for e, n in sorted(r[d][s].items())} for s in range(args.ranks)]
                          for d in range(args.ranks)])
        snap.update(stats(r, tokens[key], distinct[key], args.ranks))
        out["snapshots"][sid] = snap
    for category in args.aggregate:
        keys = [k for k in rows if k[0] in set(steps_of(categories, category, args.ranks))]
        ratios = [stats(rows[k], tokens[k], distinct[k], args.ranks)["max_over_mean"] for k in keys]
        all_distinct = [x for k in keys for x in distinct[k]]
        out["aggregate"][category] = dict(
            step_layers=len(keys), steps=len({k[0] for k in keys}),
            busiest_over_mean_median=statistics.median(ratios) if ratios else None,
            busiest_over_mean_min=min(ratios) if ratios else None,
            busiest_over_mean_max=max(ratios) if ratios else None,
            mean_distinct_remote_ranks=statistics.fmean(all_distinct) if all_distinct else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--block")
    ap.add_argument("--experts", type=int)
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--snapshot", action="append", default=[])
    ap.add_argument("--aggregate", action="append", default=[])
    ap.add_argument("--merge", type=Path, nargs="+", help="combine per-model outputs instead of reading captures")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.merge:
        merged = dict(schema="route-stats.v1", merged_from={}, snapshots={})
        for p in args.merge:
            part = json.loads(p.read_text(encoding="utf-8"))
            merged["merged_from"][p.name] = sha256(p)
            for sid, snap in part["snapshots"].items():
                if sid in merged["snapshots"]:
                    raise ValueError(f"snapshot {sid} appears twice")
                merged["snapshots"][sid] = snap
        result = merged
    else:
        if not (args.dir and args.block and args.experts):
            ap.error("--dir, --block and --experts are required unless --merge is given")
        result = run(args)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for sid, s in result["snapshots"].items():
        print(f"{sid}: step {s['step']} layer {s['layer']} busiest/mean {s['max_over_mean']:.3f} "
              f"distinct remote ranks {s['mean_distinct_remote_ranks']:.3f}")
    for cat, a in result.get("aggregate", {}).items():
        print(f"aggregate {cat}: {a['step_layers']} step-layers, busiest/mean median {a['busiest_over_mean_median']:.3f} "
              f"[{a['busiest_over_mean_min']:.3f}, {a['busiest_over_mean_max']:.3f}], "
              f"distinct remote ranks {a['mean_distinct_remote_ranks']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
