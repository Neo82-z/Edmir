#!/usr/bin/env python3
"""Write queue.csv: three order-balanced pairs of processes per unit, as in the registered design.

Registered snapshots come first, then the grids model by model, then the snapshots of the
exploratory captures of other layers (arms.json, "layers"), so a run that is cut short loses
whole blocks from the end. Inside a block the units of one pair are shuffled with a fixed seed and
each unit runs its forward process and then its reverse process.
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("GLM","QWEN35","DSV2L")
M = (1,2,4,8,16,32,64,128,256,512,1024)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms",type=Path,default=ROOT/"arms.json")
    ap.add_argument("--out",type=Path,default=ROOT/"queue.csv")
    a = ap.parse_args()
    arms = json.loads(a.arms.read_text())
    rng = random.Random(20261127)
    rows = []

    def block(stage, units, labels):
        for pair,label in enumerate(labels,1):
            order = list(units)
            rng.shuffle(order)
            for unit in order:
                for orientation in ("forward","reverse"):
                    rows.append(dict(process=f"C{len(rows)+1:04d}",stage=stage,unit=unit,pair=pair,rep=label,
                                     orientation=orientation))

    if "grid" in arms:
        block("snapshot",[f"{w}_{p}" for w in MODELS for p in ("DECODE","PREFILL")],("1","2","3"))
        for width in MODELS:
            block("grid",[f"{width}_M{m:04d}" for m in M],arms["grid"]["seeds"])
    layers = [f"{w}_{p}-L{layer}" for w,listed in arms.get("layers",{}).items() for layer in listed
              for p in ("DECODE","PREFILL")]
    if layers:
        block("layer",layers,("1","2","3"))
    with a.out.open("w",newline="") as f:
        writer = csv.DictWriter(f,list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows),"processes")


if __name__=="__main__":
    main()
