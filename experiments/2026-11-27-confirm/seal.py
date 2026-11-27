#!/usr/bin/env python3
"""Write arms.json from the screening outputs, by a fixed rule.

Every unit was screened by two separate processes before the confirmatory run. A process
gives, per arm, the geometric mean of its forward and reverse medians. For a comparison
(baseline arm, candidate arm) a unit goes into the hypothesis "candidate wins" when the ratio
baseline time over candidate time is above the margin in both processes, into "baseline wins"
when it is below 1/margin in both, and into no_prediction otherwise.

  seal.py <exploratory folder> [--margin 1.10] [--out arms.json]
  seal.py <exploratory folder> --layers GLM:3,44 QWEN35:3,37 DSV2L:3,24 --out arms-layers.json

The second form seals the snapshot comparisons on exploratory captures of other layers, as a
separate file for a separate queue; its screening files are layers/<unit>-L<layer><tag>.json.

The comparisons and the arm sets are listed below; nothing else is chosen by hand.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("GLM","QWEN35","DSV2L")
M = (1,2,4,8,16,32,64,128,256,512,1024)
GRID = [f"{w}_M{m:04d}" for w in MODELS for m in M]
SNAPSHOTS = [f"{w}_{p}" for p in ("DECODE","PREFILL") for w in MODELS]

GRID_ARMS = ["SLT","PULLW_SLT","PULLR_SLT","D_MRG16_Q8_G2_K64_W","D_Q4_G2","BNEW_Q4_G2"]
SNAPSHOT_ARMS = ["NATIVE","D_Q1_G2_R","D_Q4_G2_R","D_GATHER16_Q8_G2_K64_W_R","D_GATHER64_Q1_G4_K64_W_R","PULL_BLAS_R","BNEW_BLAS_R"]
# (short name, what is compared, baseline arm, candidate arm, folder, the two screening tags, units)
COMPARISONS = [
    ("G-pull","rows pulled from peer memory against All-to-All, same tuned staged consumer",
     "SLT","PULLW_SLT","variants",("-sweep","-sweep2"),GRID),
    ("G-pull-blocks","the same, pulled as one block per source in rank-rotated order",
     "SLT","PULLR_SLT","variants",("-sweep","-sweep2"),GRID),
    ("G-inplace","one-launch consumer reading peer memory in place against the tuned staged path",
     "SLT","D_MRG16_Q8_G2_K64_W","variants",("-sweep","-sweep2"),GRID),
    ("G-inplace-or-pull","reading in place against pulling first, both from peer memory",
     "PULLW_SLT","D_MRG16_Q8_G2_K64_W","variants",("-sweep","-sweep2"),GRID),
    ("S-pull","rows pulled from peer memory, then a library GEMM per expert, against the native path",
     "NATIVE","PULL_BLAS_R","snapshots",("-final","-final2"),SNAPSHOTS),
    ("S-pull-or-stage","the same library GEMM on rows pulled from peer memory against rows staged by All-to-All",
     "BNEW_BLAS_R","PULL_BLAS_R","snapshots",("-final","-final2"),SNAPSHOTS),
    ("S-inplace-small","one-launch consumer in place, small tiles, against the native path",
     "NATIVE","D_GATHER16_Q8_G2_K64_W_R","snapshots",("-final","-final2"),SNAPSHOTS),
    ("S-inplace-large","one-launch consumer in place, large tiles, against the native path",
     "NATIVE","D_GATHER64_Q1_G4_K64_W_R","snapshots",("-final","-final2"),SNAPSHOTS),
]


def process(path):
    """arm -> geometric mean of the medians of one screening process; None unless every arm was admitted from a graph."""
    data = json.loads(path.read_text())
    by_arm = defaultdict(list)
    for row in data["rows"]:
        by_arm[row["arm"]].append(row["wall_ms"])
    bad = {arm for arm,v in data["admission"].items() if v["status"]!="ok"}
    bad |= {r["arm"] for r in data["rows"] if r["launch_mode"]!="graph" and r["arm"]!="SLT"}
    return {arm:math.exp(statistics.fmean(map(math.log,v))) for arm,v in by_arm.items() if arm not in bad}


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--margin",type=float,default=1.10)
    ap.add_argument("--out",type=Path,default=ROOT/"arms.json")
    ap.add_argument("--layers",nargs="+",metavar="MODEL:LAYER,LAYER")
    a = ap.parse_args()
    comparisons,layers = COMPARISONS,{}
    if a.layers:
        layers = {model:[int(x) for x in listed.split(",")] for model,listed in (item.split(":") for item in a.layers)}
        units = [f"{w}_{p}-L{layer}" for p in ("DECODE","PREFILL") for w,listed in layers.items() for layer in listed]
        comparisons = [(n,what,b,c,"layers",tags,units) for n,what,b,c,folder,tags,_ in COMPARISONS if folder=="snapshots"]
    hypotheses,open_units,screening = [],[],{}
    for name,what,base,cand,folder,tags,units in comparisons:
        sides = dict(candidate=[],baseline=[],none=[])
        for unit in units:
            runs = [process(a.folder/folder/f"{unit}{tag}.json") for tag in tags]
            ratios = [r[base]/r[cand] for r in runs]   # a missing arm stops the seal
            screening[f"{name}:{unit}"] = [round(r,3) for r in ratios]
            side = "candidate" if min(ratios)>a.margin else "baseline" if max(ratios)<1/a.margin else "none"
            sides[side].append(unit)
        for winner in ("candidate","baseline"):
            if sides[winner]:
                hypotheses.append(dict(name=f"{name}/{winner}: {what}; expected faster: {cand if winner=='candidate' else base}",
                                       baseline=base,candidate=cand,winner=winner,units=sides[winner]))
        if sides["none"]:
            open_units.append(dict(baseline=base,candidate=cand,units=sides["none"]))
    sealed = dict(
        note="Sealed by the commit that contains this file. Written by seal.py from two screening processes per unit; "
             "the confirmatory run uses new processes and, on the grids, new input seeds.",
        decision=dict(band=1.03,pairs=3),policy=dict(warmup=20,timed=100),margin=a.margin,
        **(dict(layers=layers) if layers else dict(grid=dict(seeds=["20261127","20261128","20261129"],arms=GRID_ARMS))),
        snapshot=dict(arms=SNAPSHOT_ARMS),
        hypotheses=hypotheses,no_prediction=open_units,screening_ratios=screening)
    a.out.write_text(json.dumps(sealed,indent=2)+"\n")
    for h in hypotheses:
        print(f"{h['name']}\n    {len(h['units'])} units: {' '.join(h['units'])}")
    for o in open_units:
        print(f"no prediction, {o['baseline']} against {o['candidate']}: {' '.join(o['units'])}")


if __name__=="__main__":
    main()
