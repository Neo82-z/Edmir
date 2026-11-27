#!/usr/bin/env python3
"""Strict check that the process outputs are the ones the sealed files asked for.

judge.py pairs outputs by file name. This script binds each output to queue.csv and arms.json
first, so a misplaced, missing, repeated or differently configured process is found before any
unit is decided:

  - one output per queue row and no other output;
  - the unit, orientation, input seed, arm list and timing policy inside the file equal the
    queue row and the sealed arms.json;
  - every arm is admitted, and every arm except the tuned staged path replays a CUDA graph;
  - the outputs of a stage carry one harness hash, and all carry one hardware hash and one set
    of GPUs; with --harness the harness hashes must equal those of the two harness files given;
  - every unit has three pairs, each with one forward and one reverse process.

  verify.py <folder with the process outputs> [--arms arms.json] [--queue queue.csv]
            [--harness consumer_variants.py snapshot_variants.py]

Exit status 0 means every check passed.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EAGER_BY_DESIGN = ("SLT",)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--arms",type=Path,default=ROOT/"arms.json")
    ap.add_argument("--queue",type=Path,default=ROOT/"queue.csv")
    ap.add_argument("--harness",type=Path,nargs=2,metavar=("GRID","SNAPSHOT"))
    a = ap.parse_args()
    sealed = json.loads(a.arms.read_text())
    queue = list(csv.DictReader(a.queue.open()))
    problems,seen = [],defaultdict(set)
    hashes = defaultdict(Counter)
    expected = {f"{q['process']}-{q['unit']}-p{q['pair']}-{q['orientation']}":q for q in queue}
    found = {p.stem:p for p in a.folder.glob("C*.json")}
    problems += [f"missing output: {name}" for name in sorted(set(expected)-set(found))]
    problems += [f"output not in the queue: {name}" for name in sorted(set(found)-set(expected))]
    for name in sorted(set(expected)&set(found)):
        q,data = expected[name],json.loads(found[name].read_text())
        grid = q["stage"]=="grid"
        arms = sealed["grid" if grid else "snapshot"]["arms"]
        want = dict(unit=q["unit"],orientation=q["orientation"],arms=arms,policy=sealed["policy"])
        if grid:
            # Pair p of a grid cell uses the p-th sealed seed.
            want["rep"] = sealed["grid"]["seeds"][int(q["pair"])-1]
            if q["rep"]!=want["rep"]:
                problems.append(f"{name}: queue seed {q['rep']} is not the sealed seed {want['rep']}")
        for key,value in want.items():
            if data.get(key)!=value:
                problems.append(f"{name}: {key} is {data.get(key)!r}, expected {value!r}")
        problems += [f"{name}: {arm} not admitted ({v['status']})" for arm,v in data["admission"].items() if v["status"]!="ok"]
        if set(data["admission"])!=set(arms):
            problems.append(f"{name}: admission covers {sorted(data['admission'])}")
        timed = [r["arm"] for r in data["rows"]]
        if sorted(timed)!=sorted(arms) or any(r["orientation"]!=q["orientation"] for r in data["rows"]):
            problems.append(f"{name}: timed rows are {timed}")
        problems += [f"{name}: {r['arm']} launched as {r['launch_mode']}" for r in data["rows"]
                     if r["launch_mode"]!="graph" and r["arm"] not in EAGER_BY_DESIGN]
        hashes[f"harness of the {q['stage']} stage"][data["implementation_sha256"]] += 1
        hashes["hardware_sha256"][data["hardware_sha256"]] += 1
        hashes["gpu_uuids"][json.dumps(sorted(data["gpu_uuids"]))] += 1
        seen[q["unit"]].add((int(q["pair"]),q["orientation"]))
    full = {(p,o) for p in range(1,sealed["decision"]["pairs"]+1) for o in ("forward","reverse")}
    problems += [f"{unit}: processes {sorted(have)}" for unit,have in sorted(seen.items()) if have!=full]
    problems += [f"{key}: {len(values)} different values" for key,values in hashes.items() if len(values)!=1]
    for stage,path in zip(("grid","snapshot"),a.harness or ()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if set(hashes[f"harness of the {stage} stage"])!={digest}:
            problems.append(f"{path.name} has hash {digest}, not the one recorded by the {stage} outputs")
    for line in problems:
        print("FAIL",line)
    print(f"{len(found)} outputs, {len(queue)} queue rows, {len(seen)} units, {len(problems)} problems")
    for key,values in hashes.items():
        if key!="gpu_uuids" and len(values)==1:
            print(f"{key}: {next(iter(values))}")
    raise SystemExit(1 if problems else 0)


if __name__=="__main__":
    main()
