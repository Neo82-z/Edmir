#!/usr/bin/env python3
"""Decide the confirmatory run against the hypotheses frozen in arms.json.

A pair is one forward process and one reverse process of the same unit; its value for an arm
is the geometric mean of the two process medians. Three pairs give three ratios of baseline
time over candidate time. A side wins when all three are beyond the band on its side.

  judge.py <folder with the process outputs> [--csv out.csv]
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def decide(ratios, band):
    if len(ratios)!=3:
        return "incomplete"
    if all(r>band for r in ratios):
        return "candidate"
    if all(r<1/band for r in ratios):
        return "baseline"
    if all(1/band<=r<=band for r in ratios):
        return "tie"
    return "undecided"


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--csv",type=Path)
    a = ap.parse_args()
    frozen = json.loads((ROOT/"arms.json").read_text())
    band = frozen["decision"]["band"]
    # unit -> pair -> orientation -> arm -> median wall time of that process
    walls = defaultdict(lambda:defaultdict(dict))
    failures = []
    for path in sorted(a.folder.glob("C*.json")):
        match = re.fullmatch(r"C\d+-([A-Z0-9]+_[A-Z0-9]+)-p(\d)-(forward|reverse)",path.stem)
        data = json.loads(path.read_text())
        failures += [f"{path.stem}:{arm}:{v['status']}" for arm,v in data["admission"].items() if v["status"]!="ok"]
        if any(r["launch_mode"]!="graph" for r in data["rows"] if r["arm"].startswith(("D_","BNEW_")) or r["arm"]=="NATIVE"):
            failures.append(f"{path.stem}:eager launch")
            continue
        walls[match[1]][int(match[2])][match[3]] = {r["arm"]:r["wall_ms"] for r in data["rows"]}
    rows = []
    for unit,pairs in walls.items():
        stage = "snapshot" if unit.endswith(("DECODE","PREFILL")) else "grid"
        base,cand = frozen[stage]["baseline"],frozen[stage]["candidate"]
        value = lambda p,arm: math.sqrt(p["forward"][arm]*p["reverse"][arm])
        complete = [p for p in pairs.values() if set(p)=={"forward","reverse"}
                    and all(arm in p[o] for arm in (base,cand) for o in p)]
        ratios = [value(p,base)/value(p,cand) for p in complete]
        rows.append(dict(unit=unit,baseline=base,candidate=cand,pairs=len(complete),
            baseline_ms=statistics.median(value(p,base) for p in complete) if complete else None,
            candidate_ms=statistics.median(value(p,cand) for p in complete) if complete else None,
            ratio=statistics.median(ratios) if ratios else None,low=min(ratios,default=None),
            high=max(ratios,default=None),outcome=decide(ratios,band)))
    outcome = {r["unit"]:r["outcome"] for r in rows}
    if a.csv:
        with a.csv.open("w",newline="") as f:
            writer = csv.DictWriter(f,list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for r in sorted(rows,key=lambda r:r["unit"]):
        if r["ratio"] is not None:
            print(f"{r['unit']:15s} {r['baseline']} {r['baseline_ms']:.3f} ms  {r['candidate']} {r['candidate_ms']:.3f} ms  "
                  f"ratio {r['ratio']:.2f} [{r['low']:.2f}, {r['high']:.2f}]  {r['outcome']}")
    print()
    for name,expected in frozen["hypotheses"].items():
        (side,units), = expected.items()
        got = [outcome.get(u,"missing") for u in units]
        verdict = "confirmed" if all(g==side for g in got) else "not confirmed"
        print(f"{name}: {verdict} ({sum(g==side for g in got)}/{len(units)} units as predicted)")
        for u,g in zip(units,got):
            if g!=side:
                print(f"    {u}: {g}")
    print("no prediction:",{u:outcome.get(u,"missing") for u in frozen["no_prediction"]})
    if failures:
        print("excluded or failed:",failures[:20])


if __name__=="__main__":
    main()
