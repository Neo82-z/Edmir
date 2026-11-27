#!/usr/bin/env python3
"""One table of the exploratory outputs: units down, arms across, baseline time over arm time.

A process that timed both orientations gives the geometric mean of its two medians per arm.
Several files of one unit (seeds, repeats) are shown as their median ratio with the range.
With --propose the units are sorted by where the first listed arm stands against the
baseline: beyond the margin on either side, or in between (no prediction).

  table.py <folder> [--tag TAG] [--baseline SLT] [--arms A,B] [--propose 1.10] [--no-release]

<folder> is exploratory/variants, exploratory/snapshots or exploratory/layers; a file is
named <unit><tag>.json. Exploratory: not part of the registered design.
"""
import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

MODELS = ("GLM","QWEN35","DSV2L")


def order(unit):
    model,_,rest = unit.partition("_")
    digits = re.sub(r"\D","",rest.split("-")[0])
    return (MODELS.index(model) if model in MODELS else 9,rest.startswith("PREFILL"),int(digits or 0),unit)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--tag",default="",help="only files named <unit><tag>*.json")
    ap.add_argument("--baseline")
    ap.add_argument("--arms",help="columns, in this order; default: every arm seen")
    ap.add_argument("--propose",type=float,help="margin for sorting units by the first listed arm")
    ap.add_argument("--no-release",action="store_true",help="use the time without the direct arm's release barrier")
    a = ap.parse_args()
    key = "wall_without_release_ms" if a.no_release else "wall_ms"
    units,bad = defaultdict(list),[]
    for path in sorted(a.folder.glob("*.json")):
        match = re.fullmatch(r"([A-Z0-9]+_(?:M\d+|DECODE|PREFILL)(?:-L\d+)?)"+re.escape(a.tag)+r".*",path.stem)
        if not match:
            continue
        data = json.loads(path.read_text())
        by_arm = defaultdict(list)
        for row in data["rows"]:
            by_arm[row["arm"]].append(row["wall_ms"] if row["arm"] in ("SLT","NATIVE") else row[key])
        units[match[1]].append({arm:math.exp(statistics.fmean(map(math.log,v))) for arm,v in by_arm.items()})
        bad += [f"{path.stem}:{arm}:{v['status']}" for arm,v in data["admission"].items() if v["status"]!="ok"]
        bad += [f"{path.stem}:{r['arm']}:eager" for r in data["rows"]
                if r["launch_mode"]!="graph" and r["arm"] not in ("SLT","BOLD_Q1_G1")]
    if not units:
        raise SystemExit("no matching files")
    seen = sorted({arm for files in units.values() for f in files for arm in f})
    baseline = a.baseline or next(arm for arm in ("SLT","NATIVE") if arm in seen)
    arms = a.arms.split(",") if a.arms else [arm for arm in seen if arm!=baseline]
    short = lambda arm: arm.removeprefix("D_").replace("_G2","").removesuffix("_R")
    print(f"{'unit':16s}{baseline+' ms':>10s} "+" ".join(f"{short(arm):>14s}" for arm in arms))
    sides = dict(candidate=[],baseline=[],between=[])
    for unit in sorted(units,key=order):
        files = [f for f in units[unit] if baseline in f]
        if not files:
            continue
        cells = []
        for arm in arms:
            ratios = [f[baseline]/f[arm] for f in files if arm in f]
            if not ratios:
                cells.append(f"{'-':>14s}")
            elif len(ratios)==1:
                cells.append(f"{ratios[0]:14.2f}")
            else:
                cells.append(f"{statistics.median(ratios):.2f} {min(ratios):.2f}-{max(ratios):.2f}".rjust(14))
        print(f"{unit:16s}{statistics.median(f[baseline] for f in files):10.3f} "+" ".join(cells))
        first = [f[baseline]/f[arms[0]] for f in files if arms[0] in f]
        if a.propose and first:
            side = ("candidate" if min(first)>a.propose else "baseline" if max(first)<1/a.propose else "between")
            sides[side].append(unit)
    if a.propose:
        print(f"\n{arms[0]} against {baseline}, margin {a.propose}:")
        for side,listed in sides.items():
            print(f"  {side}: {json.dumps(listed)}")
    if bad:
        print("\nnot admitted or eager:",sorted(set(bad))[:20])


if __name__=="__main__":
    main()
