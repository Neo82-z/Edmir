#!/usr/bin/env python3
"""Summarize a diagnostic router envelope; this command does not run a model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.moe_shape_profile import summarize

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise SystemExit("refuse to overwrite a profile")
    if a.trace.stat().st_size>64*1024**2:raise SystemExit("trace exceeds 64 MiB budget")
    protocol=json.loads(a.protocol.read_text())
    references=protocol["representativeness_reference_sets"]
    shapes=references["archived_p8"]["shapes"]
    result=summarize(json.loads(a.trace.read_text()),evaluated_shapes=shapes,
                     reference_status=references["archived_p8"]["status"])
    result["reference_set_coverage"]={}
    for name,ref in references.items():
        keys={tuple(s[-3:]) for s in ref["shapes"]}
        covered={"calls":0,"rows":0,"flops":0}
        for row in result["logical_expert_shapes"]:
            if (row["M"],row["K"],row["N"]) in keys:
                covered["calls"]+=row["occurrences"]
                covered["rows"]+=row["occurrences"]*row["M"]
                covered["flops"]+=row["occurrences"]*2*row["M"]*row["K"]*row["N"]
        result["reference_set_coverage"][name]={"status":ref["status"],
            "logical_MKN_only":{m:covered[m]/result["logical_totals"][m] if result["logical_totals"].get(m) else None
                                for m in covered}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("origin","complete_step_layers","logical_exact_MKN_coverage","measured_time_coverage")}))
