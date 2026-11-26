#!/usr/bin/env python3
"""Decide the exploratory repeats with the rule of the registered design.

Each process times its arms once in the listed order and once reversed; the geometric mean of
the two medians is one pair value. Three processes give three pair ratios of baseline time over
candidate time. A side wins when all three ratios are beyond 1.03 on its side, the cell is a tie
when all three lie within the band, and it is undecided otherwise.

  summarize_repeats.py <exploratory dir> [--csv out.csv]

Grid cells are read from variants/<cell>-seed<seed>.json and snapshots from
snapshots/<snapshot>-repeat<i>.json. Exploratory: not part of the registered design.
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

BAND = 1.03


def pair_values(path, key="wall_ms"):
    data = json.loads(path.read_text())
    by_arm = defaultdict(list)
    for row in data["rows"]:
        by_arm[row["arm"]].append(row[key])
    return ({arm:math.exp(statistics.fmean(map(math.log,v))) for arm,v in by_arm.items() if len(v)==2},
            {arm:v["status"] for arm,v in data["admission"].items()})


def decide(ratios):
    if len(ratios)!=3:
        return "incomplete"
    if all(r>BAND for r in ratios):
        return "candidate"
    if all(r<1/BAND for r in ratios):
        return "baseline"
    if all(1/BAND<=r<=BAND for r in ratios):
        return "tie"
    return "undecided"


def collect(folder, pattern, baseline):
    units = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        match = re.fullmatch(pattern,path.stem)
        if match:
            units[match[1]].append(path)
    rows = []
    for unit,paths in units.items():
        walls,release_free = [],[]
        for path in paths:
            walls.append(pair_values(path)[0])
            release_free.append(pair_values(path,"wall_without_release_ms")[0])
        arms = sorted(set().union(*walls)-{baseline})
        for arm in arms:
            usable = [w for w in walls if arm in w and baseline in w]
            ratios = [w[baseline]/w[arm] for w in usable]
            # Without the release barrier of the direct arm; the baseline keeps its full time.
            free = [w[baseline]/f[arm] for w,f in zip(walls,release_free) if arm in f and baseline in w] if arm.startswith("D_") else []
            rows.append(dict(unit=unit,baseline=baseline,candidate=arm,processes=len(usable),
                baseline_ms=statistics.median(w[baseline] for w in usable) if usable else None,
                candidate_ms=statistics.median(w[arm] for w in usable) if usable else None,
                ratio=statistics.median(ratios) if ratios else None,
                low=min(ratios) if ratios else None,high=max(ratios) if ratios else None,outcome=decide(ratios),
                ratio_without_release=statistics.median(free) if free else None,
                outcome_without_release=decide(free) if free else ""))
    return rows


def order(unit):
    width,_,rest = unit.partition("_")
    return (("GLM","QWEN35","DSV2L").index(width),int(rest[1:]) if rest[1:].isdigit() else {"DECODE":0,"PREFILL":1}[rest])


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--csv",type=Path)
    a = ap.parse_args()
    rows = (collect(a.folder/"snapshots",r"([A-Z0-9]+_(?:DECODE|PREFILL))-repeat\d+","NATIVE")
            + collect(a.folder/"variants",r"([A-Z0-9]+_M\d+)-seed\d+","SLT"))
    rows.sort(key=lambda r:(r["unit"].endswith(("DECODE","PREFILL")) is False,order(r["unit"]),r["candidate"]))
    if a.csv:
        with a.csv.open("w",newline="") as f:
            writer = csv.DictWriter(f,list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    last = None
    for r in rows:
        if r["unit"]!=last:
            last = r["unit"]
            print(f"{r['unit']}  {r['baseline']}={r['baseline_ms']:.3f} ms" if r["baseline_ms"] else r["unit"])
        if r["ratio"] is None:
            print(f"    {r['candidate']:24s} incomplete")
            continue
        extra = (f"   without release {r['ratio_without_release']:.2f} {r['outcome_without_release']}"
                 if r["ratio_without_release"] else "")
        print(f"    {r['candidate']:24s} {r['candidate_ms']:.3f} ms  ratio {r['ratio']:.2f} [{r['low']:.2f}, {r['high']:.2f}] "
              f"{r['outcome']:9s}{extra}")


if __name__=="__main__":
    main()
