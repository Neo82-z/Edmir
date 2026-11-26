#!/usr/bin/env python3
"""Run the balanced grids and the BF16 snapshot replays of all models in a shared eight-rank protocol."""
import argparse
from collections import defaultdict
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

from balanced import BalancedPaths, ConsumerGraphs, cells, consumer_arm, input_digest, combine_input_digests, no_mark
from hw_gate import inventory as gpu_inventory, load_passing_gate
from make_design import ROOT, SNAPSHOT, BALANCED, BRIDGE, read, sha
from native import SHAPES
from snapshots import (WORLD, load_rank, load_receipts, layout, locate, reference, correctness,
                       snapshot_spec, tiles)


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
    gate = load_passing_gate(path)
    uuids,driver = gpu_inventory()
    if uuids!=gate["gpu_uuids"] or driver!=gate["driver"]:
        raise ValueError("GPU inventory or driver changed after hardware qualification")
    return gate


def inventory():
    uuids,_ = gpu_inventory()
    return uuids


def verify_captures(a):
    """Verify every snapshot of the design that was captured, and name the ones that were not.
    A snapshot that is present but invalid raises: fix it or leave its directory out."""
    inputs,coverage,captures,absent = {},{},{},{}
    for name in [row["snapshot_id"] for row in read(a.design/"snapshots.csv")]:
        spec = snapshot_spec(a.design,name)
        try:
            root = locate(a.snapshots,name)
        except FileNotFoundError as exc:
            absent[name] = str(exc)
            continue
        rows = load_receipts(root,a.design,name)
        for row in rows:
            load_rank(root,name,row,spec)
        captures[name] = rows[0]
        inputs[name+":real"] = canonical_digest(rows)
        coverage[name] = dict(model=spec["model"],step=rows[0]["step"],rows_by_source=[len(r["tokens"]) for r in rows],
                              padded_rows_by_source=rows[0]["padded_sizes"],native_tune=rows[0]["native_tune"],
                              routing_layout=rows[0].get("routing_layout"),capture_run_id=rows[0]["run_id"])
    by_model = {}
    for name,row in captures.items():
        by_model.setdefault(snapshot_spec(a.design,name)["width"],[]).append(row)
    for width,rows in by_model.items():
        for f in ("run_id","source_lock_sha256","request_sha256"):
            if len({r[f] for r in rows})!=1:
                raise ValueError(f"{width}: prefill and decode disagree on {f}")
    return inputs,captures,coverage,absent


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


def check_slt(tuned, gate, design, library, uuids):
    if (tuned.get("schema")!="edmir.moe-path-slt.v1" or tuned.get("hardware_sha256")!=sha(gate)
            or tuned.get("design_sha256")!=sha(design/"DESIGN.json")
            or tuned.get("library_sha256")!=sha(library) or tuned.get("gpu_uuids")!=uuids
            or tuned.get("balanced_sha256")!=sha(ROOT/"balanced.py")
            or set(tuned["cells"])!=set(cells(design)) or any(len(x)!=WORLD for x in tuned["cells"].values())):
        raise ValueError("SLT tuning receipt differs from the run")


def check_calibrated(path, design, microbench):
    """The second prediction must come from this run's microbenchmark and the frozen predictor."""
    from analyze import load_prediction

    receipt = json.loads((path/"CALIBRATION.json").read_text())
    seal = json.loads((path/"PREDICTION_SEAL.json").read_text())
    if (receipt.get("status")!="sealed" or microbench is None or receipt.get("microbench_sha256")!=sha(microbench)
            or receipt.get("mapping_sha256")!=sha(ROOT/"calibrate_node.py")
            or receipt.get("params_sha256")!=sha(path/"params.json")
            or seal.get("params_source")!="microbench"
            or {h for name,h in seal["inputs"].items() if Path(name).name=="params.json"}!={sha(path/"params.json")}):
        raise ValueError("calibrated prediction does not belong to this microbenchmark")
    load_prediction(path,design,"balanced")


def registered_snapshots(run):
    """Snapshot units whose captures were verified when this run was prepared."""
    return {key.removesuffix(":real") for key in run["inputs"] if key.endswith(":real")}


def prepare(a):
    import torch
    from analyze import load_prediction
    from microbench import check_receipt as check_microbench

    hardware(a.gate)
    uuids = inventory()
    if len(uuids)!=WORLD or len(set(uuids))!=WORLD:
        raise ValueError("eight distinct GPU UUIDs required")
    # Only the hardware gate and the design stop a run. A missing capture, SLT receipt or
    # microbenchmark removes only what depends on it and is recorded in RUN.json.
    smoke_failed = check_smoke(json.loads(a.smoke_receipt.read_text()),a.gate,uuids,a.library)
    missing = {}
    if a.without_snapshots:
        if a.native_tune:
            raise ValueError("--native-tune binds a capture; omit it with --without-snapshots")
        inputs,captures,coverage = {},{},{}
        missing["snapshots"] = a.without_snapshots
    else:
        inputs,captures,coverage,absent = verify_captures(a)
        # A model whose capture could not be obtained keeps its grid cells; only its snapshots are missing.
        missing.update({f"snapshot:{name}":reason for name,reason in absent.items()})
    if a.slt_receipt:
        check_slt(json.loads(a.slt_receipt.read_text()),a.gate,a.design,a.library,uuids)
    else:
        missing["slt.json"] = "not provided: SLT is recorded as unsupported"
    if a.microbench:
        check_microbench(json.loads(a.microbench.read_text()),sha(a.gate),uuids,
                         sorted({shape[0] for shape in cells(a.design).values()}))
    else:
        missing["microbench.json"] = "not provided"
    load_prediction(a.prediction,a.design,"balanced")
    if a.calibrated_prediction:
        check_calibrated(a.calibrated_prediction,a.design,a.microbench)
    else:
        missing["prediction_calibrated"] = "not provided: only the pre-boot scenario prediction is scored"
    tuned_files = {sha(path):path for path in a.native_tune or ()}
    for name,row in captures.items():
        provenance = row["native_tune"]
        if provenance["kind"]=="tuned" and provenance["sha256"] not in tuned_files:
            raise ValueError(f"{name}: include the exact native tuning file used by its capture")
    registered = {(b["unit_id"],b["rep"]) for b in read(a.design/"blocks.csv") if b["stage"]=="balanced"}
    by_unit = cells(a.design)
    for unit,rep in sorted(registered):
        inputs[f"{unit}:{rep}"] = input_digest(torch,unit,rep,by_unit[unit])
    runtime = {p:importlib.metadata.version(p) for p in ("torch","vllm","triton")}
    if runtime["vllm"]!="0.20.1":
        raise ValueError("pinned vLLM 0.20.1 is required")
    files = {name:sha(path) for name,path in source_paths(a.library).items()}
    artifacts = {"hw_gate.json":a.gate,"smoke.json":a.smoke_receipt,"slt.json":a.slt_receipt,
                 "microbench.json":a.microbench}
    artifacts = {name:path for name,path in artifacts.items() if path}
    # One tuned config file and one trial record per model; vLLM's file names already differ by shape.
    for path in list(tuned_files.values())+list(a.native_trials or ()):
        if path.name in artifacts:
            raise ValueError(f"duplicate artifact name: {path.name}")
        artifacts[path.name] = path
    frozen = {name:sha(path) for name,path in artifacts.items()}
    prediction = {p.name:sha(p) for p in a.prediction.iterdir() if p.is_file()}
    calibrated = ({p.name:sha(p) for p in a.calibrated_prediction.iterdir() if p.is_file()}
                  if a.calibrated_prediction else {})
    harness = canonical_digest(dict(runtime=runtime,files=files,artifacts=frozen,prediction=prediction,
                                     prediction_calibrated=calibrated,
                                     native_configs={name:r["native_config"] for name,r in captures.items()}))
    a.run.mkdir(parents=True,exist_ok=False)
    for name,path in artifacts.items():
        shutil.copyfile(path,a.run/name)
    shutil.copytree(a.prediction,a.run/"prediction")
    if calibrated:
        # Sealed here, before admission and timing: parameters come from the microbenchmark only.
        shutil.copytree(a.calibrated_prediction,a.run/"prediction_calibrated")
    write_json(a.run/"RUN.json",dict(schema="edmir.moe-path-combined-run.v2",origin="device_measurement",
        run_id=a.run.name,design_sha256=sha(a.design/"DESIGN.json"),hardware_sha256=frozen["hw_gate.json"],
        harness=harness,inputs=inputs,builds={arm:canonical_digest([arm,harness]) for arm in SNAPSHOT+BALANCED},
        runtime=runtime,files=files,artifacts=frozen,prediction=prediction,
        prediction_calibrated=calibrated,gpu_uuids=uuids,
        coverage=coverage,
        missing=missing,smoke_failures=smoke_failed,scope="fixed input to gate/up output"))


class Paths:
    def __init__(self, payload, receipts, routes, rank, device, cpu_group, alignment, spec):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symmetric
        from gate_up import launch
        from native import invoke

        self.torch,self.dist,self.launch,self.invoke = torch,dist,launch,invoke
        self.hidden = payload["hidden"].to(device)
        self.weight = payload["expert_weights"].to(device)
        self.rank,self.device,self.routes = rank,device,routes
        self.top_k = spec["top_k"]
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
                flat.extend((sum(self.sizes[:src])+t)*self.top_k+s for ex in source[rank] for t,s in ex)
                out_pos += count
        self.flat = torch.tensor(flat,device=device,dtype=torch.long)
        self.cfg = receipts[rank]["native_config"]
        self.tune = receipts[rank]["native_tune"]
        self.aligned = tuple(t.to(device) for t in alignment)
        self.graphs = ConsumerGraphs(torch,lambda error:spread(error,cpu_group))

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
        self.native_out = self.hidden.new_empty((sum(self.sizes),self.top_k,self.weight.shape[1]))
        post = int(self.aligned[2].item())
        bm = self.cfg["BLOCK_SIZE_M"]
        ids = self.aligned[0][:post].view(-1,bm)
        ex = self.aligned[1][:post//bm]
        kept = ids[ex>=0].flatten()
        kept = kept[kept<self.top_k*sum(self.sizes)].long()
        live_mask = torch.tensor(list(itertools.chain.from_iterable(
            [True]*l+[False]*(s-l) for s,l in zip(self.sizes,self.live,strict=True))),device=device)
        live = live_mask[kept//self.top_k]
        self.native_live = int(live.sum().item())
        self.native_rows = int(kept.numel())
        self.native_zero = int((ids[ex<0]<self.top_k*sum(self.sizes)).sum().item())
        self.native_slots = post
        if not torch.equal(kept[live].sort().values,self.flat.sort().values):
            raise ValueError("native launch does not cover the captured live assignments")

    def close(self):
        pass

    def materialize(self, source, staged, mark=no_mark):
        pack = self.peer_storage[:self.read_rows.numel()]
        self.torch.index_select(source,0,self.read_rows,out=pack)
        mark("pack")
        if staged:
            self.dist.all_to_all_single(self.staged,pack,self.receive,self.send)
        else:
            self.handle.barrier(0,60_000)
        mark("transport")

    def run(self, arm, mark=no_mark):
        """mark(name) closes a phase on the device timeline: pack, transport, then consumer;
        whatever follows the consumer is the release phase."""
        if arm=="NATIVE":
            self.open_native()
            self.nccl.all_gatherv(self.gathered,self.hidden,self.sizes)
            mark("transport")
            self.graphs.run(arm,self.native_kernel)
            mark("consumer")
            return self.native_out
        staged = arm.startswith("BNEW_")
        self.materialize(self.hidden,staged,mark)
        self.graphs.run(arm,self.consumer(arm))
        mark("consumer")
        if not staged:
            self.handle.barrier(0,60_000)
        return self.output

    def native_kernel(self):
        """The captured vLLM gate/up launch. vLLM replays this kernel from CUDA graphs outside eager mode."""
        self.open_native()
        self.invoke(self.native,self.gathered,self.weight,self.native_out,self.aligned,self.cfg,self.top_k)

    def consumer(self, arm):
        """The per-source launches of a D or BNEW arm; only their input rows differ."""
        q = 4 if "_Q4_" in arm else 1
        staged = arm.startswith("BNEW_")

        def issue():
            for direct,local,out,table in self.work:
                partial = self.partial.flatten()[:4*out.numel()].view(4,*out.shape)
                self.launch(local if staged else direct,self.weight,out,partial,table,q)
        return issue

    def prepare_graphs(self, arms):
        self.graphs.prepare({arm:self.consumer(arm) for arm in arms if consumer_arm(arm)})
        # NATIVE is its own group: if its capture fails it runs eagerly and is reported as such,
        # without changing how D and BNEW launch.
        if "NATIVE" in arms:
            self.graphs.prepare({"NATIVE":self.native_kernel},group="native")

    def admit(self, arm):
        torch = self.torch
        # Admission runs the same launch mode that timing will use.
        self.prepare_graphs([arm])
        tags = torch.zeros_like(self.hidden)
        token = torch.arange(len(tags),device=self.device)
        tags[:,0],tags[:,1],tags[:,2] = self.rank,token//128,token%128
        if arm=="NATIVE":
            self.open_native()
            self.nccl.all_gatherv(self.gathered,tags,self.sizes)
            observed = self.gathered[:,:3].index_select(0,self.flat//self.top_k)
        else:
            staged = arm.startswith("BNEW_")
            self.materialize(tags,staged)
            views = [x[1 if staged else 0][:,:3] for x in self.work]
            observed = torch.cat(views) if views else self.hidden.new_empty((0,3))
            if not staged:
                self.handle.barrier(0,60_000)
        source = self.flat//self.top_k
        offset = 0
        expected = torch.empty_like(observed)
        for src,size in enumerate(self.sizes):
            mask = (offset<=source)&(source<offset+size)
            token = source[mask]-offset
            # Masked assignment on CUDA requires matching dtypes; these small integers are exact in BF16.
            expected[mask,0] = src
            expected[mask,1] = (token//128).to(expected.dtype)
            expected[mask,2] = (token%128).to(expected.dtype)
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
                        projection_launches=1,merge_launches=0,launch_mode=self.graphs.launch_mode(arm))
        return dict(input_rows=sum(self.receive),live_assignments=self.output.shape[0],projection_rows=self.output.shape[0],
                    projection_live_rows=self.output.shape[0],padding_rows=0,
                    projection_launches=len(self.work),merge_launches=len(self.work) if "_Q4_" in arm else 0,
                    launch_mode=self.graphs.launch_mode(arm))


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
        # CUDA is the default backend of the locked build, and set_backend rejects the name of
        # the backend already in place ("does not find allocation backend CUDA").
        if symmetric.get_backend(device)!="CUDA":
            symmetric.set_backend("CUDA")
        if symmetric.get_backend(device)!="CUDA":
            raise RuntimeError("symmetric memory is not on the CUDA backend")
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
    return dict(schema="edmir.moe-path-smoke.v2",hardware_sha256=sha(gate),gpu_uuids=uuids,
                replay_sha256=sha(ROOT/"replay.py"),consumer_sha256=sha(ROOT/"gate_up.py"),
                native_sha256=sha(ROOT/"native.py"),balanced_sha256=sha(ROOT/"balanced.py"),
                library_sha256=sha(library))


def smoke_cells():
    """Two grid shapes per model width: a single row and one full M tile."""
    return {f"{width}_SMOKE_M{m:04d}":(m,k,n) for width,(_,n,k,_,_) in SHAPES.items() for m in (1,64)}


def smoke_failures(smoke):
    failed = [f"{width}:{arm}" for width in SHAPES for arm in SNAPSHOT
              if smoke.get("arms",{}).get(width,{}).get(arm,{}).get("status")!="ok"]
    failed += [f"{unit}:{arm}" for unit in smoke_cells() for arm in BALANCED
               if smoke.get("balanced",{}).get(unit,{}).get(arm,{}).get("status")!="ok"]
    return failed


def check_smoke(smoke, gate, uuids, library):
    """The smoke must be current. Arm statuses are diagnostic: formal admission decides each arm,
    so a failing smoke arm never blocks the arms that work."""
    if any(smoke.get(k)!=v for k,v in smoke_identity(gate,uuids,library).items()):
        raise ValueError("current synthetic smoke receipt required")
    return smoke_failures(smoke)


def attempt(fn, group):
    """Run fn on every rank; if any rank raised, every rank gets None and the errors."""
    try:
        value,error = fn(),None
    except Exception as exc:
        value,error = None,repr(exc)
    errors = spread(error,group)
    return (None,errors) if any(errors) else (value,None)


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

        emit()
        gen = torch.Generator(device="cpu").manual_seed(800+rank)
        # The five BF16 replay arms on each model's geometry: local experts, width and top-k.
        for width,(local,n,k,top_k,experts) in SHAPES.items():
            spec = dict(width=width,local=local,n=n,k=k,top_k=top_k,experts=experts)
            sizes = [5]*WORLD
            ids = [[[(src*9+t*3+e*11)%experts for e in range(top_k)] for t in range(5)] for src in range(WORLD)]
            routes = layout(ids,[4]*WORLD,local)

            def build():
                cfg = module().get_default_config(sum(sizes),local,n//2,k,top_k,None,None)
                rows = [dict(padded_sizes=sizes,tokens=[{}]*4,native_config=cfg,native_tune={"kind":"default"})
                        for _ in range(WORLD)]
                payload = dict(hidden=(torch.randn((5,k),generator=gen)*.125).bfloat16(),
                               expert_weights=(torch.randn((local,n,k),generator=gen)*.125).bfloat16())
                alignment = synthetic_align(torch.tensor(list(itertools.chain.from_iterable(ids)),device=device,
                                                         dtype=torch.int32),cfg,rank,experts)
                return Paths(payload,rows,routes,rank,device,cpu_group,alignment,spec)

            paths,errors = attempt(build,cpu_group)
            arms = receipt["arms"][width] = {}
            for arm in SNAPSHOT:
                arms[arm] = (dict(**guarded_admission(paths,arm,cpu_group),
                                  launch_mode=paths.graphs.launch_mode(arm),graph_errors=paths.graphs.errors)
                             if paths is not None else dict(status="runtime_error",rank_errors=errors))
                emit()
        # The FP16 grid arms, the CUTLASS bridge and cuBLASLt run here before any model is loaded.
        for unit,shape in smoke_cells().items():
            grid,errors = attempt(lambda:BalancedPaths(unit,"smoke",shape,rank,device,a.library,None,
                                                       agree=lambda error:spread(error,cpu_group)),cpu_group)
            if grid is None:
                receipt["balanced"][unit] = {arm:dict(status="runtime_error",rank_errors=errors) for arm in BALANCED}
                emit()
                continue
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
                        arms[arm] = dict(**guarded_admission(grid,arm,cpu_group),
                                         launch_mode=grid.graphs.launch_mode(arm),graph_errors=grid.graphs.errors)
                    emit()
            finally:
                grid.close()
        failed = smoke_failures(receipt)
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
            write_json(a.out,dict(schema="edmir.moe-path-slt.v1",gpu_uuids=uuids,hardware_sha256=sha(a.gate),
                design_sha256=sha(a.design/"DESIGN.json"),balanced_sha256=sha(ROOT/"balanced.py"),
                library_sha256=sha(a.library),cells=results))


def record_identity(run, unit, rep, arm):
    return dict(run_id=run["run_id"],unit_id=unit,arm_id=arm,input_sha256=run["inputs"][f"{unit}:{rep}"],
                build_sha256=run["builds"][arm],hardware_sha256=run["hardware_sha256"],harness=run["harness"])


def timed(path, policy, arm, cpu_group):
    import torch
    import torch.distributed as dist

    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples,events,enqueue,phases = [],[],[],[]
    boundaries,seen,names = {},[],[]

    def mark(name):
        # One event per phase boundary on this rank's stream; every arm pays the same record cost.
        if name not in boundaries:
            boundaries[name] = torch.cuda.Event(enable_timing=True)
        boundaries[name].record()
        seen.append(name)

    for i in range(policy["warmup"]+policy["timed"]):
        dist.barrier()
        torch.cuda.synchronize()
        seen.clear()
        begin = time.perf_counter_ns()
        start.record()
        path.run(arm,mark)
        # Host time to issue all work; close to the wall time means the iteration is launch-bound.
        queued = time.perf_counter_ns()
        end.record()
        end.synchronize()
        if i>=policy["warmup"]:
            samples.append((time.perf_counter_ns()-begin)/1e6)
            events.append(start.elapsed_time(end))
            enqueue.append((queued-begin)/1e6)
            if seen:
                names = seen+["release"]
                points = [start]+[boundaries[n] for n in seen]+[end]
                phases.append([a.elapsed_time(b) for a,b in zip(points,points[1:])])
    return spread(dict(host=samples,event=events,enqueue=enqueue,phase_names=names,phases=phases,
                       counts=path.counts(arm)),cpu_group)


def snapshot_receipts(a, run, unit):
    if unit not in registered_snapshots(run):
        raise ValueError(f"this run was prepared without {unit}")
    try:
        root = locate(a.snapshots,unit)
    except FileNotFoundError as exc:
        raise ValueError(f"snapshot input directory required: {exc}") from None
    rows = load_receipts(root,a.design,unit)
    if canonical_digest(rows)!=run["inputs"][unit+":real"]:
        raise ValueError("snapshot changed after prepare")
    return root,rows


def claim_process(run_dir, process):
    claims = run_dir/"process-started"
    claims.mkdir(exist_ok=True)
    with (claims/process).open("x") as f:
        f.write("started\n")


def device_run(a):
    import analyze

    run = json.loads((a.run/"RUN.json").read_text())
    hardware(a.run/"hw_gate.json")
    if (run["schema"]!="edmir.moe-path-combined-run.v2" or run["design_sha256"]!=sha(a.design/"DESIGN.json")
            or run["gpu_uuids"]!=inventory()):
        raise ValueError("run design or GPU changed")
    for name,h in run["artifacts"].items():
        if sha(a.run/name)!=h:
            raise ValueError(f"frozen artifact changed: {name}")
    for folder in ("prediction","prediction_calibrated"):
        for name,h in run.get(folder,{}).items():
            if sha(a.run/folder/name)!=h:
                raise ValueError(f"frozen prediction changed: {folder}/{name}")
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
    attempt_no = 2 if getattr(a,"retry",False) else 1
    if a.action=="time":
        mine = [r for r in previous if r["process_id"]==a.process]
        if not blocks or any(r.get("attempt",1)==attempt_no for r in mine):
            raise ValueError("unknown or already recorded timing process")
        if attempt_no==2 and (not (a.run/"process-started"/a.process).exists()
                              or not analyze.first_attempt_failed(mine,len(blocks))):
            raise ValueError("a retry needs a started first attempt that ended in an infrastructure failure")
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
    root,rows = (None,None) if balanced else snapshot_receipts(a,run,unit)
    if a.action=="time" and int(os.environ["RANK"])==0:
        claim_process(a.run,a.process if attempt_no==1 else a.process+".retry")
    admitted = {(r["unit_id"],r["input_id"],r["arm_id"]):analyze.numerically_valid(r) for r in admission}
    with gpu_session(run["gpu_uuids"]) as (rank,device,cpu_group):
        if balanced:
            tuned = (json.loads((a.run/"slt.json").read_text())["cells"][unit][rank]["selected"]
                     if "slt.json" in run["artifacts"] else None)
            # Decide SLT on every rank together: one rank without an algorithm would hang the others.
            slt_ready = all(spread(tuned is not None,cpu_group))
            paths = BalancedPaths(unit,rep,cells(a.design)[unit],rank,device,a.library,tuned,
                                  agree=lambda error:spread(error,cpu_group))
            actual = combine_input_digests(spread(paths.input_sha256,cpu_group))
            if actual!=run["inputs"][f"{unit}:{rep}"]:
                raise ValueError("generated FP16 inputs differ from prepare")
        else:
            spec = snapshot_spec(a.design,unit)
            payload = load_rank(root,unit,rows[rank],spec)
            ids = spread(payload["ids"].tolist(),cpu_group)
            routes = layout(ids,[len(r["tokens"]) for r in rows],spec["local"])
            aligned = tuple(payload[k] for k in ("sorted_token_ids","expert_ids","num_tokens_post_padded"))
            paths = Paths(payload,rows,routes,rank,device,cpu_group,aligned,spec)
        try:
            if a.action=="admit-arm":
                choices = [arm for arm in BALANCED if arm not in BRIDGE] if a.arm=="balanced-main" else [a.arm]
                if any(arm not in (BALANCED if balanced else SNAPSHOT) for arm in choices):
                    raise ValueError("arm does not match the stage")
                for arm in choices:
                    if (unit,rep,arm) in admitted:
                        continue
                    if balanced and arm=="SLT" and not slt_ready:
                        if rank==0:
                            append_row(apath,dict(**record_identity(run,unit,rep,arm),input_id=rep,
                                status="unsupported",reason="no frozen cuBLASLt algorithm on every rank",
                                scaled_max_err=None,nonfinite=None,fingerprint_ok=False))
                        continue
                    result = checked_admission(spread(paths.admit(arm),cpu_group))
                    if rank==0:
                        append_row(apath,dict(**record_identity(run,unit,rep,arm),input_id=rep,**result))
                return
            # Capture every consumer arm this process will time together, before any timing,
            # so the arms of one process share a launch mode.
            paths.prepare_graphs([b["arm_id"] for b in blocks if admitted.get((unit,rep,b["arm_id"]))
                                  and (b["arm_id"] not in BRIDGE or lock["bridge_enabled"])])
            for b in blocks:
                arm = b["arm_id"]
                rec = dict(**record_identity(run,unit,rep,arm),attempt=attempt_no,
                           **{k:b[k] for k in ("process_id","orientation","rep","arm_position")})
                if (arm in BRIDGE and not lock["bridge_enabled"]) or not admitted.get((unit,rep,arm)):
                    if rank==0:
                        append_row(mpath,dict(**rec,status="not_run",reason="bridge inactive or arm not admitted"))
                    continue
                try:
                    ranks = timed(paths,policy,arm,cpu_group)
                    if rank==0:
                        tag = a.process if attempt_no==1 else a.process+"-retry"
                        relative = f"{tag}-{arm}-samples.json"
                        event_file = f"{tag}-{arm}-events.json"
                        enqueue_file = f"{tag}-{arm}-enqueue.json"
                        host = list(zip(*(x["host"] for x in ranks),strict=True))
                        events = list(zip(*(x["event"] for x in ranks),strict=True))
                        enqueue = list(zip(*(x["enqueue"] for x in ranks),strict=True))
                        write_json(a.run/relative,host)
                        write_json(a.run/event_file,events)
                        write_json(a.run/enqueue_file,enqueue)
                        phase = {}
                        names = ranks[0]["phase_names"]
                        if names and all(x["phase_names"]==names and len(x["phases"])==policy["timed"] for x in ranks):
                            # iteration -> rank -> one duration per phase, in ms on the device timeline
                            per_iteration = list(zip(*(x["phases"] for x in ranks),strict=True))
                            phase_file = f"{tag}-{arm}-phases.json"
                            write_json(a.run/phase_file,per_iteration)
                            phase = dict(phase_names=names,phases_file=phase_file,phases_sha256=sha(a.run/phase_file),
                                phase_ms={n:statistics.median(max(r[j] for r in it) for it in per_iteration)
                                          for j,n in enumerate(names)})
                        append_row(mpath,dict(**rec,status="ok",warmup=policy["warmup"],timed=policy["timed"],
                            median_wall_ms=statistics.median(map(max,host)),median_event_ms=statistics.median(map(max,events)),
                            median_enqueue_ms=statistics.median(map(max,enqueue)),
                            samples_file=relative,samples_sha256=sha(a.run/relative),
                            events_file=event_file,events_sha256=sha(a.run/event_file),
                            enqueue_file=enqueue_file,enqueue_sha256=sha(a.run/enqueue_file),
                            rank_counts=[x["counts"] for x in ranks],**phase))
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
    for root in a.snapshots or ():
        command.extend(["--snapshots",str(root)])
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
    captured = registered_snapshots(run)
    if captured and not a.snapshots:
        raise ValueError("--snapshots is required for a run with captured snapshots")
    path = a.run/"admission.jsonl"
    admission = analyze.jsonl(path) if path.exists() else []
    seen = {(r["unit_id"],r["input_id"],r["arm_id"]) for r in admission}
    desired = {(b["unit_id"],b["rep"] if b["stage"]=="balanced" else "real",b["arm_id"])
               for b in read(a.design/"blocks.csv")}
    desired = {key for key in desired if key[1]!="real" or key[0] in captured}
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


def exit_status(code):
    return "ok" if code==0 else "timeout" if code==124 else "runtime_error"


def time_all(a):
    import analyze

    if not (a.run/"ADMISSION_LOCK.json").is_file():
        raise ValueError("complete and lock admission before submitting the formal queue")
    run = json.loads((a.run/"RUN.json").read_text())
    captured = registered_snapshots(run)
    if captured and not a.snapshots:
        raise ValueError("--snapshots is required for a run with captured snapshots")
    claims,exits,mpath = a.run/"process-started",a.run/"process-exits.jsonl",a.run/"metrics.jsonl"
    arms = defaultdict(int)
    for b in read(a.design/"blocks.csv"):
        arms[b["process_id"]] += 1
    grid = cells(a.design)
    order = [r["process_id"] for r in read(a.design/"queue.csv") if r["job"]=="timing_block"
             and (r["unit_id"] in grid or r["unit_id"] in captured)]
    for pid in order:
        if (claims/pid).exists():
            continue
        code = launch_job(a,["time","--process",pid],a.timeout)
        if not (claims/pid).exists():
            claim_process(a.run,pid)
        append_row(exits,dict(process_id=pid,attempt=1,exit_code=code,status=exit_status(code)))
        # Stop once, at the very first formal process, so a systematic failure is reviewed.
        # Rerunning time-all continues past it; failed processes get their retry at the end.
        if code and pid==order[0] and not mpath.exists():
            raise RuntimeError(f"first timing process {pid} exited {code}; review the retained logs")
    # One retry per process after the whole queue, decided only from exit status and row
    # completeness, never from timing values. The retry replaces that process's first attempt.
    for pid in order:
        mine = [r for r in analyze.jsonl(mpath) if r["process_id"]==pid] if mpath.exists() else []
        if (claims/(pid+".retry")).exists() or not analyze.first_attempt_failed(mine,arms[pid]):
            continue
        code = launch_job(a,["time","--process",pid,"--retry"],a.timeout)
        if not (claims/(pid+".retry")).exists():
            claim_process(a.run,pid+".retry")
        append_row(exits,dict(process_id=pid,attempt=2,exit_code=code,status=exit_status(code)))


def lock_admission(a):
    import analyze

    run = json.loads((a.run/"RUN.json").read_text())
    path = a.run/"admission.jsonl"
    rows = analyze.jsonl(path)
    blocks,_,_ = analyze.check_rows(a.design,run,a.run,rows,[])
    main = {(b["unit_id"],b["rep"] if b["stage"]=="balanced" else "real",b["arm_id"])
            for b in blocks if b["arm_id"] not in BRIDGE
            and (b["stage"]=="balanced" or b["unit_id"] in registered_snapshots(run))}
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
    p.add_argument("--snapshots",type=Path,action="append",
                   help="a capture's workers/snapshots directory; repeat once per captured model")
    p.add_argument("--run",type=Path)
    p.add_argument("--library",type=Path)
    sub = p.add_subparsers(dest="action",required=True)
    sub.add_parser("verify")
    for label in ("smoke","tune-slt","microbench"):
        cmd = sub.add_parser(label)
        cmd.add_argument("--gate",type=Path,required=True)
        cmd.add_argument("--out",type=Path,required=True)
    prep = sub.add_parser("prepare")
    for name in ("gate","smoke-receipt","prediction"):
        prep.add_argument("--"+name,type=Path,required=True)
    for name in ("slt-receipt","microbench","calibrated-prediction"):
        prep.add_argument("--"+name,type=Path)
    for name in ("native-tune","native-trials"):
        prep.add_argument("--"+name,type=Path,action="append",help="repeat once per model")
    prep.add_argument("--without-snapshots",metavar="REASON",
                      help="register the FP16 grid only, because the capture could not be obtained")
    admit = sub.add_parser("admit")
    admit.add_argument("--unit")
    admit.add_argument("--timeout",type=int,default=240)
    arm = sub.add_parser("admit-arm")
    for field in ("unit","input-id","arm"):
        arm.add_argument("--"+field,required=True)
    sub.add_parser("lock-admission")
    timing = sub.add_parser("time")
    timing.add_argument("--process",required=True)
    timing.add_argument("--retry",action="store_true",help="the single retry after an infrastructure failure")
    sub.add_parser("time-all").add_argument("--timeout",type=int,default=600)
    a = p.parse_args()
    # admit, admit-arm, time and time-all check --snapshots against RUN.json before launching anything.
    if a.action=="verify" and not a.snapshots:
        p.error("--snapshots is required")
    if a.action=="prepare" and bool(a.snapshots)==bool(a.without_snapshots):
        p.error("prepare needs exactly one of --snapshots and --without-snapshots")
    if a.action in {"prepare","admit","admit-arm","lock-admission","time","time-all"} and not a.run:
        p.error("--run is required")
    if a.action in {"smoke","tune-slt","prepare","admit","admit-arm","time","time-all"} and not a.library:
        p.error("--library is required")
    if a.action=="verify":
        inputs,_,coverage,absent = verify_captures(a)
        print(json.dumps(dict(inputs=inputs,coverage=coverage,absent=absent),indent=2))
    elif a.action=="microbench":
        from microbench import benchmark
        benchmark(a)
    else:
        actions = {"prepare":prepare,"smoke":smoke,"tune-slt":tune_slt,"admit":admit_jobs,
                   "lock-admission":lock_admission,"admit-arm":device_run,"time":device_run,"time-all":time_all}
        actions[a.action](a)


if __name__=="__main__":
    cli()
