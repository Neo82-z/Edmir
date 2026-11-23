#!/usr/bin/env python3
"""Run the GLM balanced grid and BF16 replay in a shared eight-rank protocol."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from balanced import BalancedPaths, cells, input_digest, combine_input_digests
from make_design import ROOT, SNAPSHOT, BALANCED, BRIDGE, read, sha
from snapshots import WORLD, load_rank, load_receipts, layout, reference, correctness, tiles

SMOKE_CELLS = {"SMOKE_M0001":(1,2048,3072),"SMOKE_M0064":(64,2048,3072)}


def canonical_digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def append_row(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row,allow_nan=False)+"\n")


def hardware(path):
    gate = json.loads(path.read_text())
    if gate.get("verdict")!="PASS" or not gate.get("load_test") or gate.get("failures") or gate.get("unchecked"):
        raise ValueError("passing eight-GPU load/topology/power gate required")
    return gate


def inventory():
    return subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],
                                   text=True,timeout=20).splitlines()


def verify_captures(a):
    inputs,coverage,captures = {},{},[]
    for name in ("GLM_PREFILL","GLM_DECODE"):
        rows = load_receipts(a.snapshots,a.design,name)
        for row in rows:
            load_rank(a.snapshots,name,row)
        captures.append(rows[0])
        inputs[name+":real"] = canonical_digest(rows)
        coverage[name] = dict(step=rows[0]["step"],rows_by_source=[len(r["tokens"]) for r in rows],
                              padded_rows_by_source=rows[0]["padded_sizes"],native_tune=rows[0]["native_tune"])
    for f in ("run_id","source_lock_sha256","request_sha256"):
        if captures[0][f]!=captures[1][f]:
            raise ValueError(f"prefill and decode disagree on {f}")
    return inputs,captures,coverage


def source_paths(library):
    from importlib.util import find_spec

    package = Path(find_spec("vllm").origin).parent
    paths = {p.name:p for p in ROOT.glob("*.py")}
    paths["lt_bridge.cu"] = ROOT/"lt_bridge.cu"
    paths["build_lt.sh"] = ROOT/"build_lt.sh"
    paths["library"] = library
    paths["cutlass_gemm.cu"] = ROOT.parents[1]/"EDM-IR/runtime/src/cutlass_gemm.cu"
    paths["cutlass_sm80_gemm.cuh"] = ROOT.parents[1]/"EDM-IR/runtime/src/cutlass_sm80_gemm.cuh"
    for p in ("model_executor/layers/fused_moe/fused_moe.py",
              "distributed/device_communicators/pynccl.py"):
        paths[p] = package/p
    return paths


def prepare(a):
    import torch
    from analyze import load_prediction
    from microbench import check_receipt as check_microbench

    hardware(a.gate)
    uuids = inventory()
    if len(uuids)!=WORLD or len(set(uuids))!=WORLD:
        raise ValueError("eight distinct GPU UUIDs required")
    check_smoke(json.loads(a.smoke_receipt.read_text()),a.gate,uuids,a.library)
    inputs,captures,coverage = verify_captures(a)
    tuned = json.loads(a.slt_receipt.read_text())
    if (tuned.get("schema")!="edmir.glm-slt.v1" or tuned.get("hardware_sha256")!=sha(a.gate)
            or tuned.get("design_sha256")!=sha(a.design/"DESIGN.json")
            or tuned.get("library_sha256")!=sha(a.library) or tuned.get("gpu_uuids")!=uuids
            or tuned.get("balanced_sha256")!=sha(ROOT/"balanced.py")
            or set(tuned["cells"])!=set(cells(a.design)) or any(len(x)!=WORLD for x in tuned["cells"].values())):
        raise ValueError("SLT tuning receipt differs from the run")
    check_microbench(json.loads(a.microbench.read_text()),sha(a.gate),uuids,
                     [shape[0] for shape in cells(a.design).values()])
    load_prediction(a.prediction,a.design,"balanced")
    for row in captures:
        provenance = row["native_tune"]
        if a.native_tune and (provenance["kind"]!="tuned" or provenance["sha256"]!=sha(a.native_tune)):
            raise ValueError("capture did not use the frozen native tuning file")
        if provenance["kind"]=="tuned" and not a.native_tune:
            raise ValueError("include the exact tuning file used by capture")
    registered = {(b["unit_id"],b["rep"]) for b in read(a.design/"blocks.csv") if b["stage"]=="balanced"}
    by_unit = cells(a.design)
    for unit,rep in sorted(registered):
        inputs[f"{unit}:{rep}"] = input_digest(torch,unit,rep,by_unit[unit])
    runtime = {p:importlib.metadata.version(p) for p in ("torch","vllm","triton")}
    if runtime["vllm"]!="0.20.1":
        raise ValueError("pinned vLLM 0.20.1 is required")
    files = {name:sha(path) for name,path in source_paths(a.library).items()}
    artifacts = {"hw_gate.json":a.gate,"slt.json":a.slt_receipt,"smoke.json":a.smoke_receipt,
                 "microbench.json":a.microbench}
    if a.native_tune:
        artifacts["native_tune.json"] = a.native_tune
    frozen = {name:sha(path) for name,path in artifacts.items()}
    prediction = {p.name:sha(p) for p in a.prediction.iterdir() if p.is_file()}
    harness = canonical_digest(dict(runtime=runtime,files=files,artifacts=frozen,prediction=prediction,
                                     native_configs={r["snapshot"]:r["native_config"] for r in captures}))
    a.run.mkdir(parents=True,exist_ok=False)
    for name,path in artifacts.items():
        shutil.copyfile(path,a.run/name)
    shutil.copytree(a.prediction,a.run/"prediction")
    write_json(a.run/"RUN.json",dict(schema="edmir.glm-combined-run.v2",origin="device_measurement",
        run_id=a.run.name,design_sha256=sha(a.design/"DESIGN.json"),hardware_sha256=frozen["hw_gate.json"],
        harness=harness,inputs=inputs,builds={arm:canonical_digest([arm,harness]) for arm in SNAPSHOT+BALANCED},
        runtime=runtime,files=files,artifacts=frozen,prediction=prediction,gpu_uuids=uuids,
        coverage=coverage,capture_run_id=captures[0]["run_id"],scope="fixed input to gate/up output"))


class Paths:
    def __init__(self, payload, receipts, routes, rank, device, cpu_group, alignment):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symmetric
        from gate_up import launch
        from native import invoke

        self.torch,self.dist,self.launch,self.invoke = torch,dist,launch,invoke
        self.hidden = payload["hidden"].to(device)
        self.weight = payload["expert_weights"].to(device)
        self.rank,self.device,self.routes = rank,device,routes
        self.sizes = receipts[rank]["padded_sizes"]
        self.live = [len(r["tokens"]) for r in receipts]
        self.cpu_group = cpu_group
        self.nccl = None
        matrix = [[sum(map(len,dest)) for dest in source] for source in routes]
        self.send = matrix[rank]
        self.receive = [src[rank] for src in matrix]
        self.offsets = [sum(row[:rank]) for row in matrix]
        self.read_rows = torch.tensor([token for dest in routes[rank] for ex in dest for token,_ in ex],
                                      device=device,dtype=torch.long)
        self.staged = self.hidden.new_empty((sum(self.receive),self.hidden.shape[1]))
        cap = max(map(sum,matrix))
        self.peer_storage = symmetric.empty((cap,self.hidden.shape[1]),dtype=torch.bfloat16,device=device)
        self.handle = symmetric.rendezvous(self.peer_storage,dist.group.WORLD)
        if self.handle is None or self.handle.world_size!=WORLD:
            raise RuntimeError("eight symmetric peer mappings are required")
        self.peers = [self.handle.get_buffer(src,self.peer_storage.shape,torch.bfloat16,0) for src in range(WORLD)]
        self.output = self.hidden.new_empty((sum(self.receive),self.weight.shape[1]))
        self.partial = self.output.new_empty((4,max(self.receive),self.weight.shape[1]),dtype=torch.float32)
        self.work,flat = [],[]
        out_pos = 0
        for src,source in enumerate(routes):
            count = self.receive[src]
            if count:
                table = torch.tensor(tiles([len(e) for e in source[rank]]),device=device,dtype=torch.int32)
                self.work.append((self.peers[src].narrow(0,self.offsets[src],count),
                                  self.staged.narrow(0,out_pos,count),self.output.narrow(0,out_pos,count),table))
                flat.extend((sum(self.sizes[:src])+t)*4+s for ex in source[rank] for t,s in ex)
                out_pos += count
        self.flat = torch.tensor(flat,device=device,dtype=torch.long)
        self.cfg = receipts[rank]["native_config"]
        self.tune = receipts[rank]["native_tune"]
        self.aligned = tuple(t.to(device) for t in alignment)

    def open_native(self):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from native import module

        if self.nccl is not None:
            return
        torch,device = self.torch,self.device
        self.native = module()
        self.nccl = PyNcclCommunicator(self.cpu_group,device)
        if self.nccl.disabled:
            raise RuntimeError("native all-gatherv communicator disabled")
        self.gathered = self.hidden.new_empty((sum(self.sizes),self.hidden.shape[1]))
        self.native_out = self.hidden.new_empty((sum(self.sizes),4,self.weight.shape[1]))
        post = int(self.aligned[2].item())
        bm = self.cfg["BLOCK_SIZE_M"]
        ids = self.aligned[0][:post].view(-1,bm)
        ex = self.aligned[1][:post//bm]
        kept = ids[ex>=0].flatten()
        kept = kept[kept<4*sum(self.sizes)].long()
        live_mask = torch.tensor(list(itertools.chain.from_iterable(
            [True]*l+[False]*(s-l) for s,l in zip(self.sizes,self.live,strict=True))),device=device)
        live = live_mask[kept//4]
        self.native_live = int(live.sum().item())
        self.native_rows = int(kept.numel())
        self.native_zero = int((ids[ex<0]<4*sum(self.sizes)).sum().item())
        self.native_slots = post
        if not torch.equal(kept[live].sort().values,self.flat.sort().values):
            raise ValueError("native launch does not cover the captured live assignments")

    def close(self):
        pass

    def materialize(self, source, staged):
        pack = self.peer_storage[:self.read_rows.numel()]
        self.torch.index_select(source,0,self.read_rows,out=pack)
        if staged:
            self.dist.all_to_all_single(self.staged,pack,self.receive,self.send)
        else:
            self.handle.barrier(0,60_000)

    def run(self, arm):
        if arm=="NATIVE":
            self.open_native()
            self.nccl.all_gatherv(self.gathered,self.hidden,self.sizes)
            self.invoke(self.native,self.gathered,self.weight,self.native_out,self.aligned,self.cfg)
            return self.native_out
        q = 4 if "_Q4_" in arm else 1
        staged = arm.startswith("BNEW_")
        self.materialize(self.hidden,staged)
        for direct,local,out,table in self.work:
            partial = self.partial.flatten()[:4*out.numel()].view(4,*out.shape)
            self.launch(local if staged else direct,self.weight,out,partial,table,q)
        if not staged:
            self.handle.barrier(0,60_000)
        return self.output

    def admit(self, arm):
        torch = self.torch
        tags = torch.zeros_like(self.hidden)
        token = torch.arange(len(tags),device=self.device)
        tags[:,0],tags[:,1],tags[:,2] = self.rank,token//128,token%128
        if arm=="NATIVE":
            self.open_native()
            self.nccl.all_gatherv(self.gathered,tags,self.sizes)
            observed = self.gathered[:,:3].index_select(0,self.flat//4)
        else:
            staged = arm.startswith("BNEW_")
            self.materialize(tags,staged)
            views = [x[1 if staged else 0][:,:3] for x in self.work]
            observed = torch.cat(views) if views else self.hidden.new_empty((0,3))
            if not staged:
                self.handle.barrier(0,60_000)
        source = self.flat//4
        offset = 0
        expected = torch.empty_like(observed)
        for src,size in enumerate(self.sizes):
            mask = (offset<=source)&(source<offset+size)
            token = source[mask]-offset
            expected[mask,0],expected[mask,1],expected[mask,2] = src,token//128,token%128
            offset += size
        fingerprint = torch.equal(observed,expected)
        self.dist.barrier()
        original = []
        for src,size in enumerate(self.sizes):
            buf = self.hidden if self.rank==src else self.hidden.new_empty((size,self.hidden.shape[1]))
            if size:
                self.dist.broadcast(buf,src=src)
            original.append(buf)
        ref = reference(original,self.weight,self.routes,self.rank)
        result = self.run(arm)
        if arm=="NATIVE":
            result = result.flatten(0,1).index_select(0,self.flat)
        err,nans = correctness(result,ref)
        return err,nans,fingerprint

    def counts(self, arm):
        if arm=="NATIVE":
            return dict(input_rows=sum(self.sizes),live_assignments=self.native_live,projection_rows=self.native_rows,
                        projection_live_rows=self.native_live,padding_rows=self.native_rows-self.native_live,
                        zeroed_nonlocal_rows=self.native_zero,aligned_slots=self.native_slots,native_tune=self.tune,
                        projection_launches=1,merge_launches=0)
        return dict(input_rows=sum(self.receive),live_assignments=self.output.shape[0],projection_rows=self.output.shape[0],
                    projection_live_rows=self.output.shape[0],padding_rows=0,
                    projection_launches=len(self.work),merge_launches=len(self.work) if "_Q4_" in arm else 0)


@contextmanager
def gpu_session(expected_uuids):
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"])!=WORLD or int(os.environ["RANK"])!=rank:
        raise ValueError("the experiment requires one node with eight ordered ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda",rank)
    prop = torch.cuda.get_device_properties(device)
    uuids = {u.removeprefix("GPU-") for u in expected_uuids}
    if (torch.cuda.device_count()!=WORLD or "A100" not in prop.name or "SXM" not in prop.name
            or (prop.major,prop.minor)!=(8,0) or prop.total_memory<75*1024**3
            or str(prop.uuid).removeprefix("GPU-") not in uuids):
        raise ValueError("rank is outside the registered SM80 A100 SXM4 80GB node")
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl",timeout=timedelta(seconds=120))
    try:
        symmetric.set_backend("CUDA")
        cpu_group = dist.new_group(backend="gloo",timeout=timedelta(seconds=120))
        dist.barrier()
        yield rank,device,cpu_group
    finally:
        dist.destroy_process_group()


def spread(value,group):
    import torch.distributed as dist
    result = [None]*WORLD
    dist.all_gather_object(result,value,group=group)
    return result


def checked_admission(values):
    err = None if any(r[0] is None for r in values) else max(r[0] for r in values)
    nans = sum(r[1] for r in values)
    fp = all(r[2] for r in values)
    return dict(scaled_max_err=err,nonfinite=nans,fingerprint_ok=fp,rank_checks=values,
                status="ok" if err is not None and err<=.02 and not nans and fp else "numerical_failure")


def guarded_admission(paths, arm, group):
    """Record an arm's exception and continue; ranks fail together on deterministic errors."""
    try:
        local = dict(result=list(paths.admit(arm)))
    except Exception as exc:
        local = dict(error=repr(exc))
    values = spread(local,group)
    if any("error" in v for v in values):
        return dict(status="runtime_error",rank_errors=[v.get("error") for v in values])
    return checked_admission([v["result"] for v in values])


def smoke_identity(gate, uuids, library):
    return dict(schema="edmir.glm-smoke.v2",hardware_sha256=sha(gate),gpu_uuids=uuids,
                replay_sha256=sha(ROOT/"replay.py"),consumer_sha256=sha(ROOT/"gate_up.py"),
                native_sha256=sha(ROOT/"native.py"),balanced_sha256=sha(ROOT/"balanced.py"),
                library_sha256=sha(library))


def check_smoke(smoke, gate, uuids, library):
    """The five BF16 arms must pass; every FP16 arm must have been exercised on both shapes."""
    if (any(smoke.get(k)!=v for k,v in smoke_identity(gate,uuids,library).items())
            or set(smoke.get("arms",{}))!=set(SNAPSHOT)
            or any(r["status"]!="ok" for r in smoke["arms"].values())
            or set(smoke.get("balanced",{}))!=set(SMOKE_CELLS)
            or any(set(arms)!=set(BALANCED) for arms in smoke["balanced"].values())):
        raise ValueError("current synthetic smoke receipt required")


def smoke(a):
    import torch
    from native import module,synthetic_align

    hardware(a.gate)
    uuids = inventory()
    receipt = dict(origin="synthetic_device_smoke",**smoke_identity(a.gate,uuids,a.library),arms={},balanced={})
    with gpu_session(uuids) as (rank,device,cpu_group):
        def emit():
            if rank==0:
                write_json(a.out,receipt)

        gen = torch.Generator(device="cpu").manual_seed(800+rank)
        sizes = [5]*WORLD
        ids = [[[(src*9+t*3+e*11)%64 for e in range(4)] for t in range(5)] for src in range(WORLD)]
        routes = layout(ids,[4]*WORLD)
        cfg = module().get_default_config(sum(sizes),8,1536,2048,4,None,None)
        rows = [dict(padded_sizes=sizes,tokens=[{}]*4,native_config=cfg,native_tune={"kind":"default"})
                for _ in range(WORLD)]
        payload = dict(hidden=(torch.randn((5,2048),generator=gen)*.125).bfloat16(),
                       expert_weights=(torch.randn((8,3072,2048),generator=gen)*.125).bfloat16())
        alignment = synthetic_align(torch.tensor(list(itertools.chain.from_iterable(ids)),device=device,
                                                 dtype=torch.int32),cfg,rank)
        paths = Paths(payload,rows,routes,rank,device,cpu_group,alignment)
        for arm in SNAPSHOT:
            receipt["arms"][arm] = guarded_admission(paths,arm,cpu_group)
            emit()
        # The FP16 grid arms, the CUTLASS bridge and cuBLASLt run here before any model is loaded.
        for unit,shape in SMOKE_CELLS.items():
            grid = BalancedPaths(unit,"smoke",shape,rank,device,a.library,None)
            try:
                try:
                    chosen = grid.tune()["selected"]
                except Exception:
                    chosen = None
                grid.tuned = chosen
                everywhere = None not in spread(chosen,cpu_group)
                arms = receipt["balanced"][unit] = {}
                for arm in BALANCED:
                    if arm=="SLT" and not everywhere:
                        arms[arm] = dict(status="not_run",reason="no admissible cuBLASLt heuristic on every rank")
                    else:
                        arms[arm] = guarded_admission(grid,arm,cpu_group)
                    emit()
            finally:
                grid.close()
        failed = [arm for arm,r in receipt["arms"].items() if r["status"]!="ok"]
        failed += [f"{unit}:{arm}" for unit,arms in receipt["balanced"].items()
                   for arm,r in arms.items() if r["status"]!="ok"]
        if failed:
            raise ValueError("synthetic smoke failures: "+", ".join(failed))


def tune_slt(a):
    hardware(a.gate)
    uuids = inventory()
    with gpu_session(uuids) as (rank,device,cpu_group):
        results = {}
        for unit,shape in cells(a.design).items():
            paths = BalancedPaths(unit,"tune-only",shape,rank,device,a.library,None)
            try:
                local = paths.tune()
            except Exception as exc:
                local = dict(error=repr(exc),selected=None,algorithms=[])
            finally:
                paths.close()
            results[unit] = spread(local,cpu_group)
        if rank==0:
            write_json(a.out,dict(schema="edmir.glm-slt.v1",gpu_uuids=uuids,hardware_sha256=sha(a.gate),
                design_sha256=sha(a.design/"DESIGN.json"),balanced_sha256=sha(ROOT/"balanced.py"),
                library_sha256=sha(a.library),cells=results))


def record_identity(run, unit, rep, arm):
    return dict(run_id=run["run_id"],unit_id=unit,arm_id=arm,input_sha256=run["inputs"][f"{unit}:{rep}"],
                build_sha256=run["builds"][arm],hardware_sha256=run["hardware_sha256"],harness=run["harness"])


def timed(path, policy, arm, cpu_group):
    import torch
    import torch.distributed as dist

    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples,events = [],[]
    for i in range(policy["warmup"]+policy["timed"]):
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        start.record()
        path.run(arm)
        end.record()
        end.synchronize()
        if i>=policy["warmup"]:
            samples.append((time.perf_counter_ns()-begin)/1e6)
            events.append(start.elapsed_time(end))
    return spread(dict(host=samples,event=events,counts=path.counts(arm)),cpu_group)


def snapshot_receipts(a, run, unit):
    if a.snapshots is None or not (a.snapshots/unit).is_dir():
        raise ValueError("snapshot input directory required")
    rows = load_receipts(a.snapshots,a.design,unit)
    if canonical_digest(rows)!=run["inputs"][unit+":real"]:
        raise ValueError("snapshot changed after prepare")
    return rows


def claim_process(run_dir, process):
    claims = run_dir/"process-started"
    claims.mkdir(exist_ok=True)
    with (claims/process).open("x") as f:
        f.write("started\n")


def device_run(a):
    import analyze

    run = json.loads((a.run/"RUN.json").read_text())
    hardware(a.run/"hw_gate.json")
    if (run["schema"]!="edmir.glm-combined-run.v2" or run["design_sha256"]!=sha(a.design/"DESIGN.json")
            or run["gpu_uuids"]!=inventory()):
        raise ValueError("run design or GPU changed")
    for name,h in run["artifacts"].items():
        if sha(a.run/name)!=h:
            raise ValueError(f"frozen artifact changed: {name}")
    for name,h in run["prediction"].items():
        if sha(a.run/"prediction"/name)!=h:
            raise ValueError(f"frozen prediction changed: {name}")
    for package,version in run["runtime"].items():
        if importlib.metadata.version(package)!=version:
            raise ValueError(f"runtime changed: {package}")
    for name,path in source_paths(a.library).items():
        if sha(path)!=run["files"][name]:
            raise ValueError(f"execution source changed: {name}")

    apath,mpath = a.run/"admission.jsonl",a.run/"metrics.jsonl"
    admission = analyze.jsonl(apath) if apath.exists() else []
    previous = analyze.jsonl(mpath) if mpath.exists() else []
    _,policy,_ = analyze.check_rows(a.design,run,a.run,admission,previous)
    blocks = [b for b in read(a.design/"blocks.csv") if b["process_id"]==getattr(a,"process",None)]
    if a.action=="time":
        if not blocks or a.process in {r["process_id"] for r in previous}:
            raise ValueError("unknown or already recorded timing process")
        unit,rep = blocks[0]["unit_id"],blocks[0]["rep"]
        if not (a.run/"ADMISSION_LOCK.json").is_file():
            raise ValueError("finish and lock admission before any formal timing")
        lock = json.loads((a.run/"ADMISSION_LOCK.json").read_text())
        if lock["admission_sha256"]!=sha(apath):
            raise ValueError("admission changed after it was locked")
    else:
        unit,rep = a.unit,a.input_id
    balanced = unit in cells(a.design)
    rep = rep if balanced else "real"
    # Resolve snapshot receipts before claiming: a missing or wrong --snapshots
    # must not consume the single formal attempt of a timing process.
    rows = None if balanced else snapshot_receipts(a,run,unit)
    if a.action=="time" and int(os.environ["RANK"])==0:
        claim_process(a.run,a.process)
    admitted = {(r["unit_id"],r["input_id"],r["arm_id"]):analyze.numerically_valid(r) for r in admission}
    with gpu_session(run["gpu_uuids"]) as (rank,device,cpu_group):
        if balanced:
            tuned = json.loads((a.run/"slt.json").read_text())["cells"][unit][rank]["selected"]
            paths = BalancedPaths(unit,rep,cells(a.design)[unit],rank,device,a.library,tuned)
            actual = combine_input_digests(spread(paths.input_sha256,cpu_group))
            if actual!=run["inputs"][f"{unit}:{rep}"]:
                raise ValueError("generated FP16 inputs differ from prepare")
        else:
            payload = load_rank(a.snapshots,unit,rows[rank])
            ids = spread(payload["ids"].tolist(),cpu_group)
            routes = layout(ids,[len(r["tokens"]) for r in rows])
            aligned = tuple(payload[k] for k in ("sorted_token_ids","expert_ids","num_tokens_post_padded"))
            paths = Paths(payload,rows,routes,rank,device,cpu_group,aligned)
        try:
            if a.action=="admit-arm":
                choices = [arm for arm in BALANCED if arm not in BRIDGE] if a.arm=="balanced-main" else [a.arm]
                if any(arm not in (BALANCED if balanced else SNAPSHOT) for arm in choices):
                    raise ValueError("arm does not match the stage")
                for arm in choices:
                    if (unit,rep,arm) in admitted:
                        continue
                    result = checked_admission(spread(paths.admit(arm),cpu_group))
                    if rank==0:
                        append_row(apath,dict(**record_identity(run,unit,rep,arm),input_id=rep,**result))
                return
            for b in blocks:
                arm = b["arm_id"]
                rec = dict(**record_identity(run,unit,rep,arm),
                           **{k:b[k] for k in ("process_id","orientation","rep","arm_position")})
                if (arm in BRIDGE and not lock["bridge_enabled"]) or not admitted.get((unit,rep,arm)):
                    if rank==0:
                        append_row(mpath,dict(**rec,status="not_run",reason="bridge inactive or arm not admitted"))
                    continue
                try:
                    ranks = timed(paths,policy,arm,cpu_group)
                    if rank==0:
                        relative = f"{a.process}-{arm}-samples.json"
                        event_file = f"{a.process}-{arm}-events.json"
                        host = list(zip(*(x["host"] for x in ranks),strict=True))
                        events = list(zip(*(x["event"] for x in ranks),strict=True))
                        write_json(a.run/relative,host)
                        write_json(a.run/event_file,events)
                        append_row(mpath,dict(**rec,status="ok",warmup=policy["warmup"],timed=policy["timed"],
                            median_wall_ms=statistics.median(map(max,host)),median_event_ms=statistics.median(map(max,events)),
                            samples_file=relative,samples_sha256=sha(a.run/relative),
                            events_file=event_file,events_sha256=sha(a.run/event_file),rank_counts=[x["counts"] for x in ranks]))
                except Exception as exc:
                    if rank==0:
                        append_row(mpath,dict(**rec,status="runtime_error",error=repr(exc)))
                    raise
        finally:
            paths.close()


def launch_job(a, arguments, timeout):
    command = [sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=8",
               "--max_restarts=0",str(ROOT/"replay.py"),"--design",str(a.design),
               "--run",str(a.run),"--library",str(a.library)]
    if a.snapshots:
        command.extend(["--snapshots",str(a.snapshots)])
    folder = a.run/"job-logs"
    folder.mkdir(exist_ok=True)
    label = "-".join(word for word in arguments if not word.startswith("--"))
    logfile = folder/f"{label}.log"
    with logfile.open("ab") as log:
        proc = subprocess.Popen(command+arguments,start_new_session=True,stdout=log,
                                stderr=subprocess.STDOUT)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        return 124


def admit_jobs(a):
    import analyze

    if (a.run/"ADMISSION_LOCK.json").exists():
        raise ValueError("admission is already locked")
    run = json.loads((a.run/"RUN.json").read_text())
    path = a.run/"admission.jsonl"
    admission = analyze.jsonl(path) if path.exists() else []
    seen = {(r["unit_id"],r["input_id"],r["arm_id"]) for r in admission}
    desired = {(b["unit_id"],b["rep"] if b["stage"]=="balanced" else "real",b["arm_id"])
               for b in read(a.design/"blocks.csv")}
    if a.unit:
        desired = {key for key in desired if key[0]==a.unit}
    def execute(unit,rep,arm):
        key = unit,rep,arm
        if key in seen:
            return
        code = launch_job(a,["admit-arm","--unit",unit,"--input-id",rep,"--arm",arm],a.timeout)
        if not path.exists() or not any((r["unit_id"],r["input_id"],r["arm_id"])==key for r in analyze.jsonl(path)):
            append_row(path,dict(**record_identity(run,*key),input_id=rep,
                                 status="timeout" if code==124 else "runtime_error",exit_code=code))

    # Most FP16 seeds are admitted in one process. If that process stops,
    # isolate the remaining arms and retain the receipts already written.
    balanced = sorted({(u,s) for u,s,arm in desired-seen if s!="real" and arm not in BRIDGE})
    for unit,rep in balanced:
        launch_job(a,["admit-arm","--unit",unit,"--input-id",rep,"--arm","balanced-main"],a.timeout)
    if path.exists():
        seen.update((r["unit_id"],r["input_id"],r["arm_id"]) for r in analyze.jsonl(path))
    for unit,rep,arm in sorted(desired):
        if arm not in BRIDGE:
            execute(unit,rep,arm)
    rows = analyze.jsonl(path)
    failed = any(r["unit_id"] in cells(a.design) and r["arm_id"].startswith("BNEW_") and
                 not analyze.numerically_valid(r) for r in rows)
    if failed:
        for unit,rep,arm in sorted(desired):
            if arm in BRIDGE:
                execute(unit,rep,arm)


def time_all(a):
    if not (a.run/"ADMISSION_LOCK.json").is_file():
        raise ValueError("complete and lock admission before submitting the formal queue")
    claims = a.run/"process-started"
    for row in read(a.design/"queue.csv"):
        if row["job"]!="timing_block":
            continue
        pid = row["process_id"]
        if (claims/pid).exists():
            continue
        code = launch_job(a,["time","--process",pid],a.timeout)
        if not (claims/pid).exists():
            claim_process(a.run,pid)
        append_row(a.run/"process-exits.jsonl",
                   dict(process_id=pid,exit_code=code,status="ok" if code==0 else
                        "timeout" if code==124 else "runtime_error"))
        if code and not (a.run/"metrics.jsonl").exists():
            raise RuntimeError(f"first timing process {pid} exited {code}; review the retained logs")


def lock_admission(a):
    import analyze

    run = json.loads((a.run/"RUN.json").read_text())
    path = a.run/"admission.jsonl"
    rows = analyze.jsonl(path)
    blocks,_,_ = analyze.check_rows(a.design,run,a.run,rows,[])
    main = {(b["unit_id"],b["rep"] if b["stage"]=="balanced" else "real",b["arm_id"])
            for b in blocks if b["arm_id"] not in BRIDGE}
    seen = {(r["unit_id"],r["input_id"],r["arm_id"]) for r in rows}
    if not main<=seen:
        raise ValueError("all primary admissions must be recorded before timing")
    failed = [r for r in rows if r["unit_id"] in cells(a.design)
              and r["arm_id"].startswith("BNEW_") and not analyze.numerically_valid(r)]
    if failed and any((u,s,a0) not in seen for u,s in {(u,s) for u,s,_ in main if s!="real"} for a0 in BRIDGE):
        raise ValueError("BNEW failed; finish the g1 bridge admissions")
    target = a.run/"ADMISSION_LOCK.json"
    if target.exists():
        raise ValueError("admission lock already exists")
    write_json(target,dict(admission_sha256=sha(path),bridge_enabled=bool(failed),
                           bridge_trigger=[f"{r['unit_id']}:{r['input_id']}:{r['arm_id']}" for r in failed]))


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,default=ROOT/"design")
    p.add_argument("--snapshots",type=Path)
    p.add_argument("--run",type=Path)
    p.add_argument("--library",type=Path)
    sub = p.add_subparsers(dest="action",required=True)
    sub.add_parser("verify")
    for label in ("smoke","tune-slt","microbench"):
        cmd = sub.add_parser(label)
        cmd.add_argument("--gate",type=Path,required=True)
        cmd.add_argument("--out",type=Path,required=True)
    prep = sub.add_parser("prepare")
    for name in ("gate","slt-receipt","smoke-receipt","microbench","prediction","native-tune"):
        prep.add_argument("--"+name,type=Path,required=name!="native-tune")
    admit = sub.add_parser("admit")
    admit.add_argument("--unit")
    admit.add_argument("--timeout",type=int,default=240)
    arm = sub.add_parser("admit-arm")
    for field in ("unit","input-id","arm"):
        arm.add_argument("--"+field,required=True)
    sub.add_parser("lock-admission")
    sub.add_parser("time").add_argument("--process",required=True)
    sub.add_parser("time-all").add_argument("--timeout",type=int,default=600)
    a = p.parse_args()
    if a.action in {"verify","prepare","admit","admit-arm","time","time-all"} and not a.snapshots:
        p.error("--snapshots is required")
    if a.action in {"prepare","admit","admit-arm","lock-admission","time","time-all"} and not a.run:
        p.error("--run is required")
    if a.action in {"smoke","tune-slt","prepare","admit","admit-arm","time","time-all"} and not a.library:
        p.error("--library is required")
    if a.action=="verify":
        inputs,_,coverage = verify_captures(a)
        print(json.dumps(dict(inputs=inputs,coverage=coverage),indent=2))
    elif a.action=="microbench":
        from microbench import benchmark
        benchmark(a)
    else:
        actions = {"prepare":prepare,"smoke":smoke,"tune-slt":tune_slt,"admit":admit_jobs,
                   "lock-admission":lock_admission,"admit-arm":device_run,"time":device_run,"time-all":time_all}
        actions[a.action](a)


if __name__=="__main__":
    cli()
