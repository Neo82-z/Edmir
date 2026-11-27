#!/usr/bin/env python3
"""Decide the confirmatory run against the hypotheses frozen in arms.json.

A pair is one forward process and one reverse process of the same unit; its value for an arm
is the geometric mean of the two process medians. Three pairs give three ratios of baseline
time over candidate time. A side wins when all three are beyond the band on its side.

Every hypothesis names its own baseline arm, candidate arm, the side it expects to win and
the units it covers. It is confirmed when every unit comes out on that side.

  judge.py <folder with the process outputs> [--csv out.csv] [--arms arms.json]
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
GRAPH_ARMS = ("D_","BNEW_","PULL","NATIVE")


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


def load(folder):
    """unit -> pair -> orientation -> arm -> median wall time of that process, and what was left out."""
    walls = defaultdict(lambda:defaultdict(dict))
    failures = []
    for path in sorted(folder.glob("C*.json")):
        match = re.fullmatch(r"C\d+-([A-Z0-9]+_[A-Z0-9]+)-p(\d)-(forward|reverse)",path.stem)
        data = json.loads(path.read_text())
        failures += [f"{path.stem}:{arm}:{v['status']}" for arm,v in data["admission"].items() if v["status"]!="ok"]
        eager = {r["arm"] for r in data["rows"] if r["launch_mode"]!="graph" and r["arm"].startswith(GRAPH_ARMS)}
        failures += [f"{path.stem}:{arm}:eager launch" for arm in sorted(eager)]
        walls[match[1]][int(match[2])][match[3]] = {r["arm"]:r["wall_ms"] for r in data["rows"] if r["arm"] not in eager}
    return walls,failures


def compare(pairs, base, cand, band):
    value = lambda p,arm: math.sqrt(p["forward"][arm]*p["reverse"][arm])
    complete = [p for p in pairs.values() if set(p)=={"forward","reverse"} and all(arm in p[o] for arm in (base,cand) for o in p)]
    ratios = [value(p,base)/value(p,cand) for p in complete]
    return dict(baseline=base,candidate=cand,pairs=len(complete),
        baseline_ms=statistics.median(value(p,base) for p in complete) if complete else None,
        candidate_ms=statistics.median(value(p,cand) for p in complete) if complete else None,
        ratio=statistics.median(ratios) if ratios else None,low=min(ratios,default=None),
        high=max(ratios,default=None),outcome=decide(ratios,band))


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--csv",type=Path)
    ap.add_argument("--arms",type=Path,default=ROOT/"arms.json")
    a = ap.parse_args()
    frozen = json.loads(a.arms.read_text())
    band = frozen["decision"]["band"]
    walls,failures = load(a.folder)
    rows = []
    for item in frozen["hypotheses"]+frozen.get("no_prediction",[]):
        name,expected = item.get("name","no prediction"),item.get("winner","")
        results = [dict(hypothesis=name,expected=expected,unit=unit,
                        **compare(walls.get(unit,{}),item["baseline"],item["candidate"],band)) for unit in item["units"]]
        rows += results
        if expected:
            hits = sum(r["outcome"]==expected for r in results)
            print(f"{name}: {'confirmed' if hits==len(results) else 'not confirmed'} ({hits}/{len(results)} units as predicted)")
        else:
            print(f"{name}:")
        for r in results:
            mark = "" if not expected or r["outcome"]==expected else "   <-- not as predicted"
            if r["ratio"] is None:
                print(f"    {r['unit']:15s} missing{mark}")
            else:
                print(f"    {r['unit']:15s} {r['baseline']} {r['baseline_ms']:.3f} ms  {r['candidate']} {r['candidate_ms']:.3f} ms  "
                      f"ratio {r['ratio']:.2f} [{r['low']:.2f}, {r['high']:.2f}]  {r['outcome']}{mark}")
    if a.csv and rows:
        with a.csv.open("w",newline="") as f:
            writer = csv.DictWriter(f,list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if failures:
        print("excluded or failed:",failures[:30])


if __name__=="__main__":
    main()
