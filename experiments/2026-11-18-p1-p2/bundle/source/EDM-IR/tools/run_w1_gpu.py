#!/usr/bin/env python3
"""Verify, build, or collect a prepared W1 package. Deadline is absolute UTC epoch."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import sha256
from prepare_selector_gpu import implementation_id, MERGED_FAMILIES
from run_p8_gpu import plain_topology
from run_selector_gpu import command, environment, measurement_environment


def save(path, obj):
    tmp=path.with_suffix(".partial")
    with tmp.open("w") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def verify(bundle, inputs):
    manifest=json.loads((bundle/"files.json").read_text())
    for name,expected in manifest.items():
        if sha256(bundle/name)!=expected:
            raise ValueError(f"bundle hash mismatch: {name}")
    plan=json.loads((bundle/"plan.json").read_text())
    records=json.loads((bundle/"inputs.json").read_text())
    for tag,shape in plan["shapes"].items():
        for seed in plan["seeds"]:
            record=records[tag][str(seed)]
            if record["shape"]!=shape or record["input_seed"]!=seed:
                raise ValueError("input identity mismatch")
            for name,key in [("inputs.bin","input_sha256"),("reference.bin","reference_sha256")]:
                if sha256(inputs/tag/str(seed)/name)!=record[key]:
                    raise ValueError(f"input hash mismatch: {tag}/{seed}/{name}")
    return plan, records


class Deadline:
    def __init__(self,end):
        if not time.time()<end<=time.time()+3600:
            raise ValueError("collection cutoff must be within this hour, with time left for recovery")
        self.end=end

    def run(self,args,limit,**kwargs):
        remaining=self.end-time.time()
        if remaining<1: raise TimeoutError("collection cutoff reached")
        return command(args,timeout=min(limit,remaining),**kwargs)

    def text(self,args,limit=20):
        return self.run(args,limit,capture_output=True).stdout.strip()


def build(args,plan,clock):
    if args.dependencies is None: raise ValueError("build needs --dependencies")
    cutlass=args.dependencies/"cutlass"
    nccl=args.dependencies/"nccl"
    for name,path in (("cutlass",cutlass),("nccl",nccl)):
        if clock.text(["git","-C",path,"rev-parse","HEAD"])!=plan[name] or clock.text(
                ["git","-C",path,"status","--porcelain","--untracked-files=no"]):
            raise ValueError(f"wrong dependency identity: {name}")
    inc=nccl/"build/include"
    lib=(nccl/"build/lib/libnccl.so").resolve(strict=True)
    lock=json.loads((args.bundle/"nccl-artifacts.json").read_text())
    for name,path in [("libnccl.so",lib),("nccl.h",inc/"nccl.h"),("nccl_device.h",inc/"nccl_device.h")]:
        if sha256(path)!=lock[name]:raise ValueError("historical NCCL artifact differs: "+name)
    b=args.build_dir
    b.mkdir(parents=True,exist_ok=False)
    flags=["-DCMAKE_BUILD_TYPE=Release","-DEDMIR_ENABLE_CUDA=ON","-DEDMIR_ENABLE_CUTLASS=ON",
           "-DEDMIR_ENABLE_NCCL_DEVICE=ON",f"-DEDMIR_CUTLASS_DIR={cutlass}",
           f"-DEDMIR_EXPERIMENT_DIR={args.bundle}",f"-DNCCL_INCLUDE_DIR={inc}",f"-DNCCL_LIBRARY={lib}",
           f"-DCMAKE_BUILD_RPATH={lib.parent}"]
    nvcc=clock.text(["nvcc","--version"])
    if "release 12.8," not in nvcc: raise ValueError("W1 requires CUDA 12.8")
    with (b/"build.log").open("w") as log:
        for cmd in [["cmake","-S",args.bundle/"source/EDM-IR/runtime","-B",b,*flags],
                    ["cmake","--build",b,"-j",str(args.jobs)],["ctest","--test-dir",b,"--output-on-failure"]]:
            clock.run(cmd,600,stdout=log,stderr=subprocess.STDOUT)
    save(b/"receipt.json",dict(package=sha256(args.bundle/"files.json"),nvcc=nvcc,
        compiler=clock.text(["c++","--version"]),cmake=clock.text(["cmake","--version"]),
        dependencies={str(p):sha256(p) for p in (lib,inc/"nccl.h",inc/"nccl_device.h")},
        nccl_library=sha256(lib),binaries={tag:sha256(b/f"w1_{tag}_batch") for tag in plan["shapes"]}))


def preflight(args,plan,clock):
    b=args.build_dir
    receipt=json.loads((b/"receipt.json").read_text())
    if receipt['package']!=sha256(args.bundle/"files.json"): raise ValueError("build/package mismatch")
    for path,value in receipt["dependencies"].items():
        if sha256(Path(path))!=value: raise ValueError("build dependency changed")
    loaded={}
    for tag in plan["shapes"]:
        binary=b/f"w1_{tag}_batch"
        if sha256(binary)!=receipt["binaries"][tag]: raise ValueError("binary hash mismatch")
        linked=clock.text(["ldd",binary])
        libs=re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)\s+\(",linked,re.M)
        for _,p in libs:
            if p not in loaded: loaded[p]=sha256(Path(p))
        if "not found" in linked or [loaded[p] for lib,p in libs if lib.startswith("libnccl.so")]!=[receipt["nccl_library"]]:
            raise ValueError("loaded NCCL differs from build")
    runtime=json.loads(clock.text([b/"w1_grouped_batch","--inspect"]))
    inventory=clock.text(["nvidia-smi","--query-gpu=index,pci.bus_id","--format=csv,noheader"])
    topo=plain_topology(clock.text(["nvidia-smi","topo","-m"]))
    hardware=environment(runtime,topo,inventory,8)
    if runtime["cuda_version"]!="12080" or runtime["nccl_version"]!="22809":
        raise ValueError("CUDA/NCCL runtime identity mismatch")
    active=clock.text(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"])
    if active: raise ValueError("visible GPUs already have compute processes: "+active)
    return dict(package=receipt['package'],build=sha256(b/"receipt.json"),hardware=hardware,
        gpu_inventory=clock.text(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,name","--format=csv,noheader"]),
        loaded_libraries=loaded,topology=topo,runtime=runtime,measurement_env=measurement_environment(os.environ))


def validate_rows(rows,job,order,shape,runtime):
    if [r.get("family") for r in rows]!=order: raise ValueError("missing, reordered or duplicate rows")
    for r in rows:
        # Legacy W1 rows predate explicit precision fields. New FP32 storage
        # must never inherit that legacy allowance or an FP16 timing identity.
        if r.get("family", "").endswith("_fp32reduce") or "precision" in r:
            from edmir.compiler.w1_candidate_selection import precision, ERROR_CONTRACT
            family=r["family"]
            numerical_family="host" if family.startswith("pipeline") else family
            if r.get("precision")!=precision(numerical_family) or r.get("error_contract")!=ERROR_CONTRACT:
                raise ValueError("precision or error contract mismatch")
            mode="merged" if family in MERGED_FAMILIES else "prepared"
            policy="uniform_immutable_peer_v1" if family.startswith(("direct","splitk")) else "resource_queues_v1"
            expected=(family+"_cutlass_sm80_double_buffer_v1_prepared_metadata_v2" if family.startswith("pipeline")
                      else implementation_id(family,mode,dependency_policy=policy))
            if r.get("implementation_id")!=expected:
                raise ValueError("correctness/timing implementation identity mismatch")
        if r.get("input_seed")!=job["seed"] or type(r.get("correctness_passed")) is not bool:
            raise ValueError("incorrect seed or validation status")
        if r.get("checked_elements")!=shape[0]**2*shape[1]*shape[3]:raise ValueError("incomplete output check")
        if not r["correctness_passed"]:
            if not r.get("failure") or r.get("median_ms") is not None:raise ValueError("invalid failure record")
        elif job["phase"]=="check":
            if r.get("check_only") is not True or "samples_ms" in r:raise ValueError("expected correctness-only row")
        else:
            family=r["family"]
            mode="merged" if family in MERGED_FAMILIES else "prepared"
            policy="uniform_immutable_peer_v1" if family.startswith(("direct","splitk")) else "resource_queues_v1"
            expected=(family+"_cutlass_sm80_double_buffer_v1_prepared_metadata_v2" if family.startswith("pipeline")
                      else implementation_id(family,mode,dependency_policy=policy))
            if (r.get("implementation_id")!=expected or r.get("dependency_policy")!=policy
                or r.get("metadata_mode")!=mode or r.get("nonfinite_outputs")!=0
                or r.get("measurement_policy")!={"profiler":False,"nvtx":False,"discarded_samples":0}):
                raise ValueError("candidate or timing identity mismatch")
            samples=r.get("samples_ms",[])
            if (r["shape"]!=shape or r["runtime"]!=runtime or r["warmup"]!=20 or r["iterations"]!=100
                or len(samples)!=100 or any(not math.isfinite(t) or t<=0 for t in samples)):
                raise ValueError("timing identity or sample contract failed")
            import statistics
            if not math.isclose(r["median_ms"],statistics.median(samples),rel_tol=1e-12):
                raise ValueError("median differs from raw samples")


def collect(args,plan,records,clock,context):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    with (out/"collector.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=out/"context.json"
        if state.exists() and json.loads(state.read_text())!=context:
            raise ValueError("resume requires identical binary, package, hardware and measurement environment")
        save(state,context)
        invalid=set()
        if args.action=="collect":
            require_successful_pilot(out,sha256(state),plan,context)
        jobs=plan["jobs"]
        if args.action=="pilot":
            jobs=[dict(shape="grouped",phase="pilot",seed=plan["seeds"][0],repetition=0,order=["host","direct"])]
        for i,job in enumerate(jobs):
            tag=job["shape"]
            final=out/f"{args.action}-{i:03d}.json"
            order=[f for f in job["order"] if (tag,f) not in invalid or job["phase"]=="check"]
            if final.exists():
                done=json.loads(final.read_text())
                if done["job"]!=job or done["context"]!=sha256(state):raise ValueError("resume identity mismatch")
                rows=done["rows"]
            else:
                if clock.end-time.time()<5:raise TimeoutError("collection cutoff reached")
                stem=f"{args.action}-{i:03d}-{time.time_ns()}"
                stdout=out/(stem+".stdout");stderr=out/(stem+".stderr")
                data=args.inputs/tag/str(job["seed"])
                cmd=[args.build_dir/f"w1_{tag}_batch","--check" if job["phase"]=="check" else "--measure",
                    data/"inputs.bin",data/"reference.bin",str(job["seed"]),*order]
                began=time.time()
                with stdout.open("x") as o,stderr.open("x") as e:
                    if order:clock.run(cmd,300,stdout=o,stderr=e)
                rows=[json.loads(line) for line in stdout.read_text().splitlines()]
                validate_rows(rows,job,order,plan["shapes"][tag],context["runtime"])
                save(final,dict(job=job,context=sha256(state),input=records[tag][str(job["seed"])],
                    rows=rows,order=order,excluded_by_numerical_check=[f for f in job["order"] if f not in order],
                    stdout=stdout.name,stdout_sha256=sha256(stdout),stderr=stderr.name,process_seconds=time.time()-began))
            if done_raw_changed(final,out): raise ValueError("raw output no longer matches completed block")
            validate_rows(rows,job,order,plan["shapes"][tag],context["runtime"])
            for row in rows:
                if not row["correctness_passed"]:invalid.add((tag,row['family']))
            print(f"{args.action} {i+1}/{len(jobs)} {tag} {job['phase']} {len(rows)} rows",flush=True)
        if args.action=="pilot" and invalid:
            raise ValueError("pilot numerical failure; full collection blocked")
        save(out/f"{args.action}-complete.json",dict(attempted_groups=len(jobs),invalid=sorted(invalid),
            complete_at=datetime.now(timezone.utc).isoformat(),valid_set_only=True))


def require_successful_pilot(out,context_hash,plan,context):
    path=out/"pilot-000.json"
    if not (out/"pilot-complete.json").is_file() or not path.is_file():
        raise ValueError("run the two-candidate pilot in this result directory first")
    record=json.loads(path.read_text())
    job=dict(shape="grouped",phase="pilot",seed=plan["seeds"][0],repetition=0,order=["host","direct"])
    if (record["context"]!=context_hash or record["job"]!=job or done_raw_changed(path,out)
        or not all(row.get("correctness_passed") is True for row in record["rows"])):
        raise ValueError("pilot evidence invalid for full collection")
    validate_rows(record["rows"],job,job["order"],plan["shapes"]["grouped"],context["runtime"])


def done_raw_changed(path,out):
    row=json.loads(path.read_text())
    raw=out/row["stdout"]
    return sha256(raw)!=row["stdout_sha256"] or row["rows"]!=[json.loads(s) for s in raw.read_text().splitlines()]


TRACE_CASES={"grouped-g1":("grouped","splitk8"),"grouped-g2":("grouped","splitk8_groupedn2"),
             "m512-serial":("m512","pipeline_serial"),"m512-pipeline":("m512","pipeline")}


def trace(args,plan,clock,context):
    cases=plan.get("trace_cases",TRACE_CASES)
    if args.trace_case not in cases:
        raise ValueError("trace case is not prescribed in this package")
    tag,family=cases[args.trace_case]
    if json.loads((args.out/"context.json").read_text())!=context:
        raise ValueError("trace requires the primary run context")
    passed=False
    for path in sorted(args.out.glob("collect-[0-9][0-9][0-9].json")):
        block=json.loads(path.read_text())
        if block["job"]["shape"]!=tag:continue
        for row in block["rows"]:
            if row["family"]!=family:continue
            if not row["correctness_passed"]:raise ValueError("trace candidate failed validation")
            passed |= block["job"]["phase"] in ("measure","pair")
    if not passed:raise ValueError("trace requires a valid unprofiled measurement first")
    dest=args.out/"diagnostic"/args.trace_case
    dest.mkdir(parents=True,exist_ok=False)
    seed=plan["seeds"][0]
    data=args.inputs/tag/str(seed)
    cmd=["nsys","profile","--trace=cuda,nvtx,osrt","--sample=none","--cpuctxsw=none",
         "--capture-range=nvtx","--capture-range-end=stop","-p","edmir.w1.capture",
         "-e","NSYS_NVTX_PROFILER_REGISTER_ONLY=0","-o",dest/"timeline",
         args.build_dir/f"w1_{tag}_batch","--trace",data/"inputs.bin",data/"reference.bin",str(seed),family]
    save(dest/"intent.json",dict(kind="profiled_diagnostic_only",case=args.trace_case,
         version=clock.text(["nsys","--version"]),command=list(map(str,cmd)),context=sha256(args.out/"context.json")))
    with (dest/"stdout").open("x") as o,(dest/"stderr").open("x") as e:
        clock.run(cmd,120,stdout=o,stderr=e)
    report=dest/"timeline.nsys-rep"
    if not report.is_file() or not report.stat().st_size or '"diagnostic_only":true' not in (dest/"stdout").read_text():
        raise ValueError("incomplete diagnostic capture")
    save(dest/"receipt.json",dict(kind="diagnostic_only",report_sha256=sha256(report),
         trace_contents_require_review=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["verify","build","pilot","collect","trace"])
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--build-dir",type=Path)
    p.add_argument("--dependencies",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--cutoff-epoch",type=float)
    p.add_argument("--trace-case")
    p.add_argument("--guard-log",type=Path)
    p.add_argument("--jobs",type=int,default=8)
    args=p.parse_args()
    for name in ("bundle","inputs","build_dir","dependencies","out"):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if args.action!="verify" and (args.build_dir is None or args.cutoff_epoch is None):
        p.error("build/measurement needs --build-dir and an absolute --cutoff-epoch")
    if args.action in ("pilot","collect","trace") and args.out is None:p.error("measurement needs --out")
    if not 1<=args.jobs<=16:p.error("--jobs must be 1..16")
    if args.action!="verify" and Path(__file__).resolve()!=args.bundle/"source/EDM-IR/tools/run_w1_gpu.py":
        p.error("run the collector from the sealed bundle")
    if args.action=="trace" and not args.trace_case:p.error("trace requires --trace-case")
    if args.action!="verify" and os.environ.get("RUNPOD_POD_ID"):
        if args.guard_log is None:p.error("RunPod execution requires the live stop-guard log")
        guard=json.loads(args.guard_log.read_text().splitlines()[0])
        if (guard["state"]!="armed" or guard["pod"]!=os.environ["RUNPOD_POD_ID"] or
            args.cutoff_epoch>guard["collection_cutoff"] or args.cutoff_epoch<=time.time()):
            p.error("wrong guard identity or unsafe collection cutoff")
        os.kill(guard["pid"],0)
    plan,records=verify(args.bundle,args.inputs)
    if args.action=="verify":
        print("W1 source, programs, input and reference hashes verified")
        return
    clock=Deadline(args.cutoff_epoch)
    if args.action=="build":build(args,plan,clock)
    elif args.action=="trace":trace(args,plan,clock,preflight(args,plan,clock))
    else:collect(args,plan,records,clock,preflight(args,plan,clock))


if __name__=="__main__": main()
