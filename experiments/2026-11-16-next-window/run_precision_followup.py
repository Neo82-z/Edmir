#!/usr/bin/env python3
"""Balanced large-N replication using the already compiled precision binaries.

Dry-run is local. Execution requires a fresh, verified platform spending guard.
The runner neither rents resources nor stops platform billing.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

FAMILIES = ("splitk2", "splitk4", "splitk4_groupedn2", "splitk8_groupedn2",
            "host_merged_cutlass", "host_merged_cublaslt")
SEEDS = (20261020,20261101,20261102)
BUNDLE_SHA = "da2f74ce4043e3b83f08b939e53ba840ec1c174d14c2b0fdd02b7da0ae9d32b3"
SCHEMA = "edmir.large-n-balanced-followup.v1"


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def jobs():
    # Even-order Williams design: each candidate occupies each position once;
    # every ordered distinct adjacent pair occurs once within the six orders.
    base=(0,1,5,2,4,3)
    orders=[[FAMILIES[(i+offset)%6] for i in base] for offset in range(6)]
    result=[dict(shape="large_n",phase="check",seed=seed,repetition=i,order=list(FAMILIES))
            for i,seed in enumerate(SEEDS)]
    blocks=[(seed,index) for seed in SEEDS for index in range(6)]
    random.Random(20261116).shuffle(blocks)
    result += [dict(shape="large_n",phase="measure",seed=seed,repetition=i,
                    order=orders[index]) for i,(seed,index) in enumerate(blocks)]
    return result


def plan():
    return dict(schema=SCHEMA,source_manifest_sha256=BUNDLE_SHA,jobs=jobs(),
        scope="one original development shape; no frozen shapes, fitting or new CUDA kernels",
        factors=dict(input_seed=list(SEEDS),candidate_order="six Williams orders per input seed"),
        warmup=20,iterations=100,profiler=False,expected_check_rows=18,
        expected_timing_rows_if_all_admitted=108,expected_raw_samples_if_all_admitted=10800,
        caveat="Process and temporal drift are recorded, not proven absent by order balance")


def validate_plan(value):
    if value != plan(): raise ValueError("follow-up plan differs from declared fixed design")
    for seed in SEEDS:
        orders=[j["order"] for j in value["jobs"] if j["phase"]=="measure" and j["seed"]==seed]
        for pos in range(6):
            if Counter(order[pos] for order in orders) != Counter(FAMILIES):
                raise ValueError("candidate positions are not balanced")
        pairs=Counter((a,b) for order in orders for a,b in zip(order,order[1:]))
        if len(pairs)!=30 or set(pairs.values())!={1}:raise ValueError("adjacent pairs are not balanced")


def cutoff_from_guard(guard,now,*,schema=SCHEMA):
    if guard.get("authorization_confirmed") is not True or guard.get("platform_verified") is not True:
        raise ValueError("fresh authorization and observed platform stop rule required")
    if not guard.get("instance_id") or guard.get("experiment_schema") != schema:
        raise ValueError("guard is not bound to this run")
    fields=("billed_from_epoch","hourly_total_usd","budget_usd","platform_stop_epoch")
    if any(type(guard.get(k)) not in (int,float) or not math.isfinite(guard[k]) for k in fields):
        raise ValueError("finite current billing values required")
    start,rate,budget,end=(guard[k] for k in fields)
    if not 0<budget<=15 or rate<=0 or not start<=now<end:
        raise ValueError("invalid billing window")
    if (end-start)*rate/3600>budget:raise ValueError("platform deadline exceeds total budget")
    cutoff=end-600
    if cutoff<=now:raise TimeoutError("data recovery reserve reached")
    return min(cutoff,now+3500)  # bounded collector; platform rule remains authoritative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("plan","bundle","inputs","build-dir","out","guard"):
        p.add_argument("--"+arg,type=Path,required=arg in ("plan","bundle"))
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    value=json.loads(a.plan.read_text());validate_plan(value)
    if sha(a.bundle/"files.json")!=BUNDLE_SHA:raise ValueError("wrong sealed binary/source package")
    if a.dry_run:
        print(json.dumps(dict(status="local_plan_valid_GPU_NOT_RUN",jobs=len(value["jobs"]),
                              check_rows=18,timing_rows=108,raw_samples=10800)))
        return
    if any(getattr(a,k) is None for k in ("inputs","build_dir","out","guard")):
        p.error("execution requires inputs, build-dir, out and fresh guard")
    guard=json.loads(a.guard.read_text());cutoff=cutoff_from_guard(guard,time.time())
    sys.path.insert(0,str(a.bundle/"source/EDM-IR/tools"))
    sys.path.insert(0,str(a.bundle/"source/EDM-IR"))
    import run_w1_gpu as w1
    clock=w1.Deadline(cutoff)
    source,records=w1.verify(a.bundle,a.inputs)
    context=w1.preflight(a,source,clock)
    context["followup_identity"]=dict(plan=sha(a.plan),runner=sha(__file__),instance_id=guard["instance_id"])
    a.out.mkdir(parents=True,exist_ok=True)
    intent=dict(plan=value,context=context,guard=guard,cutoff_epoch=cutoff)
    # The exact cutoff is recorded but may shorten on a same-identity resume.
    old=a.out/"intent.json"
    if old.exists():
        prev=json.loads(old.read_text())
        if any(prev[k]!=intent[k] for k in ("plan","context","guard")):
            raise ValueError("resume identity differs")
    else:w1.save(old,intent)
    run_plan=dict(source,jobs=value["jobs"])
    args=SimpleNamespace(out=a.out,inputs=a.inputs,build_dir=a.build_dir,action="pilot")
    status=dict(state="running",GPU_started_by_runner=False)
    try:
        w1.collect(args,run_plan,records,clock,context)
        args.action="collect";w1.collect(args,run_plan,records,clock,context)
        status["state"]="collection_complete_requires_local_analysis"
    except BaseException as exc:
        status.update(state="interrupted",error=str(exc));raise
    finally:
        status["finished_epoch"]=time.time();w1.save(a.out/"status.json",status)


if __name__=="__main__":main()
