#!/usr/bin/env python3
"""Write the verdicts of one sealed batch as a Markdown table per comparison.

  summarize.py <folder with the process outputs> --arms arms.json [--title "..."] > RESULTS.md

Uses the rule and the loader of judge.py. A row shows the medians of the three pairs, the
median ratio (baseline time over candidate time) with its range, the outcome, and what was
predicted for that unit, if anything.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from judge import compare, load

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder",type=Path)
    ap.add_argument("--arms",type=Path,default=ROOT/"arms.json")
    ap.add_argument("--title",default="Confirmatory run")
    a = ap.parse_args()
    frozen = json.loads(a.arms.read_text())
    band = frozen["decision"]["band"]
    walls,failures = load(a.folder)
    groups = defaultdict(list)   # (baseline, candidate) -> (unit, predicted)
    for item in frozen["hypotheses"]+frozen.get("no_prediction",[]):
        groups[item["baseline"],item["candidate"]] += [(unit,item.get("winner","")) for unit in item["units"]]
    print(f"# {a.title}\n\nSealed file: `{a.arms.name}`. Band {band}, three pairs; ratio = baseline time / candidate time.\n")
    for item in frozen["hypotheses"]:
        results = [compare(walls.get(u,{}),item["baseline"],item["candidate"],band)["outcome"] for u in item["units"]]
        hits = sum(r==item["winner"] for r in results)
        print(f"- {'confirmed' if hits==len(results) else 'not confirmed'} ({hits}/{len(results)}): {item['name']}")
    order = {u:i for i,u in enumerate(sorted({u for g in groups.values() for u,_ in g},
             key=lambda u:(("GLM","QWEN35","DSV2L").index(u.split("_")[0]),"PREFILL" in u,u.split("-")[0],int(u.split("-L")[1]) if "-L" in u else 0)))}
    for (base,cand),listed in groups.items():
        print(f"\n## {cand} against {base}\n\n| unit | {base} ms | {cand} ms | ratio | range | outcome | predicted |\n|---|---|---|---|---|---|---|")
        for unit,predicted in sorted(listed,key=lambda x:order[x[0]]):
            r = compare(walls.get(unit,{}),base,cand,band)
            if r["ratio"] is None:
                print(f"| {unit} | | | | | missing | {predicted} |")
                continue
            mark = "" if not predicted else " ✓" if r["outcome"]==predicted else " ✗"
            print(f"| {unit} | {r['baseline_ms']:.3f} | {r['candidate_ms']:.3f} | {r['ratio']:.2f} | {r['low']:.2f}–{r['high']:.2f} | {r['outcome']} | {predicted}{mark} |")
    if failures:
        print("\nExcluded or failed:",", ".join(failures))


if __name__=="__main__":
    main()
