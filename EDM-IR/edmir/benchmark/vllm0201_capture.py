"""Version-bound vLLM worker instrumentation, loaded only in diagnostic jobs.

No model changes. Performance jobs use the stock worker. Route jobs use eager,
synchronous scheduling and additional CPU collectives; their timings are not a
performance baseline. GPU import is deferred until vLLM resolves CaptureWorker.
"""
import atexit
from collections import Counter
from contextlib import contextmanager
import functools
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time

from .real_moe import (ENGINE, LAYERS, WORLD, TOP_K, REVISION, JsonlWriter, StepSampler,
    classify, file_sha, input_rows, save, slice_routes, route_slice_bounds)


def verify_sources(root, lock):
    checked={}
    for name, expected in lock["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:raise ValueError("unsafe source path")
        value=file_sha(Path(root)/name)
        if value!=expected:raise ValueError("vLLM source differs from audited tag: "+name)
        checked[name]=value
    return checked


def route_block(tokens):
    blocks=set()
    for t in tokens:
        m=re.match(r"diag-(S1|S32|L1|L32)-\d{3}(?:$|[-_])",t["request_id"])
        if m:blocks.add("diag-"+m.group(1))
        elif t["request_id"].startswith("warmup-"):continue
        else:raise ValueError("unrecognized diagnostic request ID")
    if len(blocks)>1:raise ValueError("overlapping diagnostic workloads")
    return next(iter(blocks),None)


def padded_assignment(sorted_ids, expert_ids, post_padded, block_m, real_mask, *, top_k=TOP_K):
    """Count observed backend assignment slots, not a rounded logical estimate."""
    if type(post_padded) is not int or post_padded<0 or post_padded>len(sorted_ids):
        raise ValueError("bad backend padding extent")
    if type(block_m) is not int or block_m<=0 or post_padded%block_m:
        raise ValueError("bad block size")
    blocks=post_padded//block_m
    if blocks>len(expert_ids):raise ValueError("missing backend block IDs")
    result={}
    valid_assignment_ids=set()
    for block in range(blocks):
        expert=expert_ids[block]
        if type(expert) is not int or expert < -1:raise ValueError("bad local expert")
        if expert==-1:continue
        entry=result.setdefault(expert,dict(block_slots=0,real_rows=0,dummy_or_padding_slots=0))
        entry["block_slots"]+=block_m
        for assignment in sorted_ids[block*block_m:(block+1)*block_m]:
            if type(assignment) is not int or assignment<0:raise ValueError("bad assignment ID")
            real=assignment<len(real_mask)*top_k and real_mask[assignment//top_k]
            if real:
                if assignment in valid_assignment_ids:raise ValueError("duplicate live backend assignment")
                valid_assignment_ids.add(assignment);entry["real_rows"]+=1
            else:entry["dummy_or_padding_slots"]+=1
    return result


class CaptureState:
    def __init__(self, runner, config):
        import torch
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
        self.torch=torch;self.runner=runner;self.config=config
        self.dp=get_dp_group();self.ep=get_ep_group();self.tp=get_tp_group()
        if (self.tp.world_size,self.dp.world_size,self.ep.world_size)!=(1,8,8):
            raise ValueError("collector only admits TP1/DP8/EP8")
        if self.dp.rank_in_group!=self.ep.rank_in_group:raise ValueError("DP and EP ordering differs")
        self.rank=self.dp.rank_in_group;self.pending=[];self.current=None;self.layer=None
        self.ordinal=0;self.samplers={};self.writers={};self.failures=0
        self.out=Path(config["out"]);self.out.mkdir(parents=True,exist_ok=True)
        self.mode=config["mode"]
        if self.mode not in ("routes","trace"):raise ValueError("instrumentation forbidden in performance mode")
        pc=runner.parallel_config
        if pc.enable_dbo or pc.enable_eplb or runner.speculative_config is not None:
            raise ValueError("DBO/EPLB/speculative path outside collection contract")
        if self.mode=="routes" and (not runner.model_config.enforce_eager or runner.scheduler_config.async_scheduling):
            raise ValueError("route collection requires eager + synchronous scheduling")
        dev=torch.cuda.current_device();prop=torch.cuda.get_device_properties(dev)
        save(self.out/f"worker-{self.rank}.json",dict(rank=self.rank,pid=os.getpid(),device=dev,
            gpu_name=prop.name,gpu_uuid=str(getattr(prop,"uuid","unavailable")),
            dp_ranks=list(self.dp.ranks),ep_ranks=list(self.ep.ranks),mode=self.mode,
            diagnostic_timing_only=True,source_lock=config["source_lock_sha256"]))
        atexit.register(self.close)

    def armed(self):return Path(self.config["armed_file"]).is_file()

    def write(self,kind,**fields):
        if self.current is None:return
        block=self.current["block_id"]
        if block not in self.writers:
            self.writers[block]=JsonlWriter(self.out/f"{block}-rank{self.rank}.jsonl.gz")
        self.writers[block].write(dict(kind=kind,origin="real_router",run_id=self.config["run_id"],
            block_id=block,step=self.current["step"],rank=self.rank,**fields))

    def safe(self,fn):
        try:return fn()
        except Exception as exc:
            self.failures+=1
            # A capture failure must not strand other ranks in a collective.
            # Preserve it and let the original model call finish unchanged.
            try:
                if self.current is not None:self.write("failure",error=repr(exc),layer=self.layer.layer_id if self.layer else None)
            except Exception:
                pass  # The separate small failure receipt remains required.
            save(self.out/f"capture-failure-rank{self.rank}.json",dict(error=repr(exc),failures=self.failures))
            return None

    @contextmanager
    def nvtx(self,label):
        self.torch.cuda.nvtx.range_push(f"edmir/rank={self.rank}/{label}")
        try:yield
        finally:self.torch.cuda.nvtx.range_pop()

    @contextmanager
    def forward(self, context):
        if not self.armed():yield;return
        self.ordinal+=1;tokens=list(self.pending)
        if self.mode=="trace":
            with self.nvtx(f"forward={self.ordinal}/tokens={len(tokens)}"):
                yield
            return
        sizes=context.dp_metadata.num_tokens_across_dp_cpu.tolist()
        phases=Counter(t["phase"] for t in tokens)
        local=dict(rank=self.rank,block_id=route_block(tokens),prefill=phases["prefill"],
                   decode=phases["decode"],ordinal=self.ordinal,tokens=tokens)
        gathered=[None]*WORLD
        self.torch.distributed.all_gather_object(gathered,local,group=self.dp.cpu_group)
        if len({r["ordinal"] for r in gathered})!=1:raise ValueError("forward ordinals diverged")
        block_ids={r["block_id"] for r in gathered if r["block_id"] is not None}
        if len(block_ids)>1:raise ValueError("concurrent diagnostic workloads")
        if not block_ids:yield;return
        block=block_ids.pop()
        if any(t["request_id"].startswith("warmup-") for r in gathered for t in r["tokens"]):
            raise ValueError("warmup and measured diagnostic requests overlap")
        for r in gathered:r["block_id"]=block
        sampler=self.samplers.setdefault(block,StepSampler())
        category,selected=sampler.observe(self.ordinal,gathered)
        snapshot=None
        if self.config.get("snapshot_replay") and block=="diag-S32":
            snapshot={("pure_prefill",1):"GLM_PREFILL",("pure_decode",8):"GLM_DECODE"}.get(
                (category,sampler.steps[category]))
            selected=snapshot is not None
        old=self.current
        self.current=dict(step=self.ordinal,block_id=block,selected=selected,
                          sizes=sizes,all_tokens=[r["tokens"] for r in gathered],category=category,
                          snapshot=snapshot,phase_ordinal=sampler.steps[category])
        if selected:
            self.write("step",tokens=tokens,category=category,selected=True,
                       padded_sizes=sizes,local_dummy=not tokens)
        try:
            with self.nvtx(f"{block}/step={self.ordinal}/{category}"):yield
        except BaseException as exc:
            self.write("failure",error=repr(exc));raise
        finally:
            save(self.out/f"{block}-sampling-rank{self.rank}.json",dict(
                step_counts=dict(sampler.steps),token_counts=dict(sampler.tokens),
                capture_failures=self.failures,sampling_rule=(
                    "first prefill and eighth decode" if self.config.get("snapshot_replay")
                    else "first three steps of each global phase")))
            self.current=old

    def route(self,layer,weights,ids):
        if not self.current or not self.current["selected"] or self.mode!="routes":return
        # The naive AgRs path gathers activations + logits BEFORE select_experts.
        layout="gathered_dp" if layer.runner.do_naive_dispatch_combine else "local"
        sizes=self.current["sizes"];tokens=self.current["all_tokens"][self.rank]
        if ids.shape!=weights.shape:raise ValueError("router ID/weight shape mismatch")
        begin,end=route_slice_bounds(int(ids.shape[0]),rank=self.rank,sizes=sizes,real_count=len(tokens),layout=layout)
        if self.current["snapshot"] and layer.layer_id==23:
            if "snapshot_routes" in self.current:
                raise ValueError("replay expects one route per rank at layer 23")
            # Both layouts give this rank's padded source rows: offset 0 when routing is local
            # (observed on 2026-11-17), the rank's offset in the gathered rows for naive dispatch.
            padded_end=begin+sizes[self.rank]
            self.current["snapshot_routes"]=(ids[begin:padded_end].detach().cpu(),
                                            weights[begin:padded_end].detach().float().cpu(),layout)
        # Slice before D2H: do not copy all gathered routes on every rank.
        result=slice_routes(ids[begin:end].detach().cpu().tolist(),
            weights[begin:end].detach().float().cpu().tolist(),rank=self.rank,
            sizes=[len(t) for t in self.current["all_tokens"]],tokens=tokens,layout="local")
        result.update(routing_layout=layout,source_slice_offset=begin,
                      observed_router_rows=int(ids.shape[0]),padding_or_dummy_rows=sizes[self.rank]-len(tokens))
        self.write("route",layer=layer.layer_id,**result)

    def kernel(self,bound):
        if not self.current or not self.current["selected"] or self.mode!="routes" or self.layer is None:return
        a,b,c=bound["A"],bound["B"],bound["C"]
        self.write("kernel_observation",layer=self.layer.layer_id,input_shape=list(a.shape),
                   weight_shape=list(b.shape),output_shape=list(c.shape),
                   input_dtype=str(a.dtype),weight_dtype=str(b.dtype),
                   input_stride=list(a.stride()),weight_stride=list(b.stride()))
        dims=tuple(b.shape[-2:])
        projection={(3072,2048):"gate_up",(2048,1536):"down"}.get(dims)
        if projection is None:raise ValueError("unexpected physical projection geometry")
        if str(a.dtype)!="torch.bfloat16" or str(b.dtype)!="torch.bfloat16":raise ValueError("non-BF16 kernel")
        sizes=self.current["sizes"];alltokens=self.current["all_tokens"]
        mask=[]
        for size,tt in zip(sizes,alltokens):
            if len(tt)>size:raise ValueError("actual tokens exceed padded slice")
            mask += [True]*len(tt)+[False]*(size-len(tt))
        expected=len(mask)*(TOP_K if projection=="down" else 1)
        if a.shape[0]!=expected:raise ValueError("unsupported chunked/backend token layout")
        if projection=="gate_up" and self.current["snapshot"] and self.layer.layer_id==23:
            self.save_replay(a,b,c,bound,tokens=alltokens[self.rank])
        sorted_ids=bound["sorted_token_ids"]
        if sorted_ids is None:raise ValueError("naive assignment layout requires a separate collector")
        post=int(bound["num_tokens_post_padded"].item());cfg=bound["config"]
        bm=int(cfg["BLOCK_SIZE_M"])
        if post<0 or post>sorted_ids.numel() or bm<=0 or post%bm:raise ValueError("invalid observed padding extent")
        block_experts=bound["expert_ids"][:post//bm]
        local=block_experts>=0
        # Only this rank's live blocks are copied; retain global padded extent separately.
        local_ids=sorted_ids[:post].view(-1,bm)[local].reshape(-1).detach().cpu().tolist()
        local_experts=block_experts[local].detach().cpu().tolist()
        slots=padded_assignment(local_ids,local_experts,len(local_ids),bm,mask)
        emap=self.layer.expert_map
        self.write("kernel",layer=self.layer.layer_id,projection=projection,
            backend_source="vllm.fused_moe.invoke_fused_moe_triton_kernel",input_shape=list(a.shape),
            weight_shape=list(b.shape),output_shape=list(c.shape),input_stride=list(a.stride()),
            weight_stride=list(b.stride()),input_dtype=str(a.dtype),weight_dtype=str(b.dtype),
            output_dtype=str(c.dtype),accumulator_dtype="float32 (audited Triton source)",
            config={k:v for k,v in cfg.items() if type(v) in (int,float,str,bool)},
            num_tokens_post_padded=post,local_assignment_slots=len(local_ids),local_expert_assignment=slots,
            expert_map=emap.detach().cpu().tolist() if emap is not None else None,
            kernel_time_measured=False,physical_layout_verified=True)

    def save_replay(self,a,b,c,bound,tokens):
        snapshot=self.current["snapshot"]
        directory=self.out/"snapshots"/snapshot
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f"rank-{self.rank}.pt"
        if path.exists():raise ValueError("duplicate layer-23 gate/up snapshot")
        sizes=self.current["sizes"]
        begin=sum(sizes[:self.rank]);end=begin+sizes[self.rank]
        emap=self.layer.expert_map
        expected_map=[e%8 if e//8==self.rank else -1 for e in range(64)]
        if (emap is None or emap.detach().cpu().tolist()!=expected_map
                or tuple(b.shape)!=(8,3072,2048) or a.shape[1]!=2048
                or c.dtype!=self.torch.bfloat16 or bound["mul_routed_weight"] or bound["top_k"]!=4
                or bound.get("B_bias") is not None):
            raise ValueError("unexpected GLM gate/up contract")
        if (bound.get("sorted_token_ids") is None or tuple(c.shape)!=(a.shape[0],4,3072)
                or bound["compute_type"].name!="bf16"
                or any(bound.get(k) for k in ("use_fp8_w8a8","use_int8_w8a8","use_int8_w8a16",
                                               "use_int4_w4a16","per_channel_quant","block_shape"))
                or any(bound.get(k) is not None for k in ("A_scale","B_scale"))):
            raise ValueError("unsupported native gate/up argument")
        ids,weights,routing_layout=self.current.pop("snapshot_routes")
        cfg={k:v for k,v in bound["config"].items() if type(v) in (int,float,str,bool)}
        fm=importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        filename=fm.get_config_file_name(8,1536,None)
        folder=os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
        expected_sha=self.config.get("native_tune_sha256")
        installed=Path(folder)/filename if folder else None
        bundled=Path(fm.__file__).parent/"configs"/filename
        resolved=installed if installed is not None and installed.is_file() else bundled
        if resolved.is_file():
            digest=file_sha(resolved)
            configs={int(k):v for k,v in json.loads(resolved.read_text()).items() if k!="triton_version"}
            key=min(configs,key=lambda m:abs(m-int(a.shape[0])))
            if configs[key]!=cfg or (expected_sha is not None and digest!=expected_sha):
                raise ValueError("observed native launch differs from the frozen tune configuration")
            tune=dict(kind="tuned",sha256=digest,batch_size=key,source="user" if resolved==installed else "bundled")
        else:
            expected=fm.get_default_config(int(a.shape[0]),8,1536,2048,4,None,None)
            if expected!=cfg or expected_sha is not None:
                raise ValueError("native config source could not be verified")
            tune=dict(kind="default")
        tmp=path.with_suffix(".tmp")
        self.torch.save(dict(hidden=a[begin:end].detach().cpu().contiguous().clone(),
                             expert_weights=b.detach().cpu().contiguous(),ids=ids.contiguous(),
                             router_weights=weights.contiguous(),
                             **{key:bound[key].detach().cpu().contiguous()
                                for key in ("sorted_token_ids","expert_ids","num_tokens_post_padded")}),tmp)
        os.replace(tmp,path)
        save(path.with_suffix(".json"),dict(schema="edmir.glm-gate-up-snapshot.v2",origin="real_vllm",
            snapshot=snapshot,run_id=self.config["run_id"],model_revision=REVISION,rank=self.rank,
            layer=23,phase=self.current["category"],phase_ordinal=self.current["phase_ordinal"],
            step=self.current["step"],padded_sizes=sizes,tokens=tokens,snapshot_sha256=file_sha(path),
            source_lock_sha256=self.config["source_lock_sha256"],request_sha256=self.config["request_sha256"],
            native_config=cfg,native_tune=tune,native_strides=dict(A=list(a.stride()),B=list(b.stride()),C=list(c.stride())),
            routing_layout=routing_layout,
            expert_map=expected_map,gate_up_accumulator="float32"))

    def close(self):
        for writer in self.writers.values():writer.close()
        self.writers.clear()


def install(runner,config):
    import vllm
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker import gpu_model_runner as gm
    fm=importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    if importlib.metadata.version("vllm")!=ENGINE:raise ValueError("wrong vLLM version")
    lock=json.loads(Path(config["source_lock"]).read_text())
    if file_sha(config["source_lock"])!=config["source_lock_sha256"]:raise ValueError("source lock changed")
    verify_sources(Path(vllm.__file__).parent.parent,lock)
    state=CaptureState(runner,config)
    old_prepare=runner._prepare_inputs
    @functools.wraps(old_prepare)
    def prepare(scheduler_output,num_scheduled_tokens):
        batch=runner.input_batch
        state.pending=input_rows([dict(request_id=req,computed_tokens=int(batch.num_computed_tokens_cpu[i]),
            prompt_tokens=int(batch.num_prompt_tokens[i]),scheduled_tokens=int(num_scheduled_tokens[i]))
            for i,req in enumerate(batch.req_ids)])
        return old_prepare(scheduler_output,num_scheduled_tokens)
    runner._prepare_inputs=prepare
    old_dummy=runner._dummy_run
    @functools.wraps(old_dummy)
    def dummy(*args,**kwargs):
        previous=state.pending;state.pending=[]
        try:return old_dummy(*args,**kwargs)
        finally:state.pending=previous
    runner._dummy_run=dummy
    old_context=gm.set_forward_context
    @contextmanager
    @functools.wraps(old_context)
    def context(*args,**kwargs):
        with old_context(*args,**kwargs):
            with state.forward(get_forward_context()):yield
    gm.set_forward_context=context
    found=[]
    for _,layer in runner.model.named_modules():
        if not (hasattr(layer,"router") and hasattr(layer,"runner") and hasattr(layer,"w13_weight")):continue
        found.append(layer.layer_id)
        old_route=layer.router.select_experts
        def make_route(original,owner):
            @functools.wraps(original)
            def route(*args,**kwargs):
                with state.nvtx(f"layer={owner.layer_id}/router"):
                    result=original(*args,**kwargs)
                state.safe(lambda:state.route(owner,*result))
                return result
            return route
        layer.router.select_experts=make_route(old_route,layer)
        original=layer.runner._forward_impl
        def make_forward(original,owner):
            @functools.wraps(original)
            def forward(*args,**kwargs):
                previous=state.layer;state.layer=owner
                try:
                    with state.nvtx(f"layer={owner.layer_id}/moe"):return original(*args,**kwargs)
                finally:state.layer=previous
            return forward
        layer.runner._forward_impl=make_forward(original,layer)
    if sorted(found)!=list(LAYERS):raise ValueError("actual MoE layer set differs from GLM plan")
    original_kernel=fm.invoke_fused_moe_triton_kernel;signature=inspect.signature(original_kernel)
    @functools.wraps(original_kernel)
    def kernel(*args,**kwargs):
        if state.mode=="routes" and state.current and state.current["selected"]:
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();state.safe(lambda:state.kernel(bound.arguments))
        with state.nvtx(f"layer={state.layer.layer_id if state.layer else 'unknown'}/expert_gemm"):
            return original_kernel(*args,**kwargs)
    fm.invoke_fused_moe_triton_kernel=kernel
    for method in ("dispatch_router_logits","dispatch","combine"):
        original=getattr(AgRsAll2AllManager,method)
        def make_collective(original,name):
            @functools.wraps(original)
            def call(owner,*args,**kwargs):
                with state.nvtx(f"layer={state.layer.layer_id if state.layer else 'unknown'}/{name}"):
                    result=original(owner,*args,**kwargs)
                if state.current and state.current["selected"]:
                    state.safe(lambda:state.write("collective",layer=state.layer.layer_id if state.layer else None,
                        name="reduce_scatterv" if name=="combine" else "all_gatherv",wrapper=name,
                        tensors=[dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size())
                                 for t in args if hasattr(t,"numel")],
                        logical_tensor_bytes_not_link_traffic=True))
                return result
            return call
        setattr(AgRsAll2AllManager,method,make_collective(original,method))
    return state


def __getattr__(name):
    if name!="CaptureWorker":raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker
    class CaptureWorker(Worker):
        def load_model(self,*,load_dummy_weights=False):
            super().load_model(load_dummy_weights=load_dummy_weights)
            path=os.environ.get("EDMIR_REAL_MOE_CAPTURE_CONFIG")
            if not path:raise ValueError("explicit diagnostic config required")
            self.edmir_capture=install(self.model_runner,json.loads(Path(path).read_text()))
    return CaptureWorker
