#!/usr/bin/env python3
"""Offline-asset-only real MoE runner. Default: inspect local text/schedule only."""
import argparse
import asyncio
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.real_moe import (ENGINE, MODEL, PROFILE_ID, REVISION, collect_block, file_sha,
    model_vocab_size, save, schedule, validate_corpus, validate_token_packet, workload)
from edmir.benchmark.real_moe_config import engine_arguments, collection_files, validate_topology


def check_guard(guard, now, instance):
    if (guard.get("schema")!="edmir.real-moe-gpu-guard.v1" or
        guard.get("gpu_execution_authorized") is not True or
        guard.get("platform_auto_delete_verified") is not True or
        not instance or guard.get("instance_id")!=instance):
        raise ValueError("fresh authorized instance-bound platform deletion guard required")
    start=guard["billing_start_epoch"];stop=guard["platform_delete_epoch"]
    rate=guard["usd_per_hour"];budget=guard["approved_usd"]
    if any(type(x) not in (int,float) or not __import__('math').isfinite(x) for x in (start,stop,rate,budget)):
        raise ValueError("invalid billing values")
    if not start<=now<stop-120 or rate<=0 or budget<=0 or (stop-start)*rate/3600>budget:
        raise ValueError("expired or over-budget guard")
    return stop-120


def make_stat_logger(buffer):
    from vllm.v1.metrics.loggers import StatLoggerBase
    class RecordStats(StatLoggerBase):
        def __init__(self,vllm_config,engine_index=0):self.engine_index=engine_index
        def log_engine_initialized(self):pass
        def record(self,scheduler_stats,iteration_stats,mm_cache_stats=None,engine_idx=0):
            if len(buffer)>=262144:raise RuntimeError("scheduler metadata budget exceeded")
            row=dict(monotonic_ns=time.perf_counter_ns(),dp_engine=self.engine_index)
            for label,value in (("scheduler",scheduler_stats),("iteration",iteration_stats)):
                values=vars(value) if value is not None and hasattr(value,"__dict__") else {}
                row[label]={k:v for k,v in values.items() if type(v) in (int,float,bool,str)}
            buffer.append(row)
    return RecordStats


async def telemetry(out, stop):
    with (out/"gpu-telemetry.jsonl").open("x") as f:
        while not stop.is_set():
            proc=None
            try:
                proc=await asyncio.create_subprocess_exec("nvidia-smi",
                    "--query-gpu=timestamp,index,uuid,power.draw,clocks.sm,memory.used,utilization.gpu",
                    "--format=csv",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                stdout,stderr=await asyncio.wait_for(proc.communicate(),5)
                value=dict(monotonic_ns=time.perf_counter_ns(),stdout=stdout.decode(),stderr=stderr.decode(),exit_code=proc.returncode)
            except Exception as exc:
                if proc is not None and proc.returncode is None:proc.kill();await proc.wait()
                value=dict(error=repr(exc))
            f.write(json.dumps(value)+"\n");f.flush()
            try:await asyncio.wait_for(stop.wait(),5)
            except asyncio.TimeoutError:pass


async def execute(a,packet,receipt):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    import torch
    for name in ("vllm","torch","transformers","tokenizers"):
        if importlib.metadata.version(name)!=receipt["runtime"][name]:raise ValueError("package changed: "+name)
    if torch.version.cuda!=receipt["runtime"]["cuda_build"]:raise ValueError("CUDA build changed")
    if torch.cuda.device_count()!=8:raise ValueError("exactly eight GPUs required")
    devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                  memory=torch.cuda.get_device_properties(i).total_memory) for i in range(8)]
    if any("A100" not in d["name"] or "SXM" not in d["name"] or d["capability"]!=[8,0] or d["memory"]<75*1024**3 for d in devices):
        raise ValueError("this plan requires A100 80GB SM80")
    os.environ.pop("EDMIR_REAL_MOE_CAPTURE_CONFIG",None)
    args=engine_arguments(a.model_dir,a.mode)
    if a.mode!="baseline":
        config=dict(mode=a.mode,out=str((a.out/"workers").resolve()),
                    run_id=a.out.name,source_lock=str(a.source_lock.resolve()),
                    source_lock_sha256=file_sha(a.source_lock),armed_file=str((a.out/"capture.armed").resolve()),
                    snapshot_replay=a.snapshot_replay,request_sha256=file_sha(a.tokens),
                    native_tune_sha256=file_sha(a.native_tune) if a.native_tune else None)
        if a.native_tune:
            os.environ["VLLM_TUNED_CONFIG_FOLDER"]=str(a.native_tune.resolve().parent)
        elif a.snapshot_replay:
            os.environ.pop("VLLM_TUNED_CONFIG_FOLDER",None)
        save(a.out/"capture-config.json",config)
        os.environ["EDMIR_REAL_MOE_CAPTURE_CONFIG"]=str((a.out/"capture-config.json").resolve())
    save(a.out/"runtime-intent.json",dict(args=args,devices=devices,model=MODEL,revision=REVISION,
        timing_scope="native AsyncLLM; no HTTP transport",diagnostic_only=a.mode!="baseline",
        predictor_used=False,edmir_integrated=False))
    # Preserve topology evidence. Actual interconnect acceptance remains explicit.
    for label,command in (("topology",["nvidia-smi","topo","-m"]),("gpu-details",["nvidia-smi","-q"])):
        p=await asyncio.create_subprocess_exec(*command,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:stdout,stderr=await asyncio.wait_for(p.communicate(),20)
        except BaseException:
            p.kill();await p.wait();raise
        (a.out/(label+".txt")).write_bytes(stdout+stderr)
        if p.returncode:raise ValueError("GPU inventory failed")
        if label=="topology":validate_topology(stdout.decode())
    stats=[];engine=None;stop=asyncio.Event();monitor=asyncio.create_task(telemetry(a.out,stop))
    active_profile=False
    try:
        begin=time.perf_counter()
        engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**args),stat_loggers=[make_stat_logger(stats)])
        save(a.out/"load.json",dict(seconds=time.perf_counter()-begin))
        if a.mode!="baseline":(a.out/"capture.armed").write_text("diagnostic only\n")
        async def generate(request,request_id):
            finished=False
            try:
                async for result in engine.generate({"prompt_token_ids":request["prompt_token_ids"]},
                    SamplingParams(temperature=0,max_tokens=128,ignore_eos=True),request_id):
                    final=result.outputs[0]
                    yield dict(token_ids=list(final.token_ids),prompt_token_ids=list(result.prompt_token_ids),
                               finished=result.finished)
                    finished=result.finished
            finally:
                if not finished:await engine.abort(request_id)
        if a.mode=="baseline":
            functional=[]
            for r in packet["functional_requests"]:
                last=None
                async for v in engine.generate({"prompt_token_ids":r["prompt_token_ids"]},
                    SamplingParams(temperature=0,max_tokens=64),"functional-"+r["id"]):last=v
                ok=last is not None and last.finished and bool(last.outputs[0].token_ids)
                functional.append(dict(id=r["id"],success=ok,text=last.outputs[0].text if last else None,
                    output_token_ids=list(last.outputs[0].token_ids) if last else []))
                save(a.out/"functional.json",functional)
                if not ok:raise ValueError("functional generation failed")
        blocks=schedule() if a.mode=="baseline" else [dict(block_id="diag-"+n,**workload(n)) for n in ("S1","S32","L1","L32")]
        if a.snapshot_replay:
            blocks=[dict(block_id="diag-S32",**workload("S32"))]
            blocks[0]["requests"]=32
        for block in blocks:
            rows=[r for r in packet["requests"] if r["input_tokens"]==block["input_tokens"]]
            rows=sorted(rows,key=lambda r:r["source_id"])
            for j in range(2):
                warm=await collect_block(generate,rows[:block["concurrency"]],
                    block_id=f"warmup-{block['block_id']}-{j}",concurrency=block["concurrency"],
                    checkpoint=a.out/f"warmup-{block['block_id']}-{j}-partial.json")
                save(a.out/(f"warmup-{block['block_id']}-{j}.json"),warm)
                if warm["successes"]!=warm["expected_requests"]:raise ValueError("warmup failed")
            if a.mode=="trace":
                await engine.start_profile(block["block_id"]);active_profile=True
            n=block["concurrency"] if a.mode=="trace" else block["requests"]
            started=time.time()
            result=await collect_block(generate,rows[:n],block_id=block["block_id"],concurrency=block["concurrency"],
                                       checkpoint=a.out/(block["block_id"]+"-partial.json"))
            if active_profile:
                await engine.stop_profile();active_profile=False
            result.update(mode=a.mode,origin="real_vllm",start_epoch=started,diagnostic_only=a.mode!="baseline")
            save(a.out/(block["block_id"]+".json"),result)
            save(a.out/(block["block_id"]+"-engine-metrics.json"),stats);stats.clear()
            try:
                from prometheus_client import generate_latest
                (a.out/(block["block_id"]+"-prometheus.txt")).write_bytes(generate_latest())
            except Exception as exc:save(a.out/(block["block_id"]+"-metrics-failure.json"),dict(error=repr(exc)))
            if result["successes"]!=result["expected_requests"]:raise ValueError("failed requests retained; stop this run")
    finally:
        stop.set();await monitor
        if stats:save(a.out/"remaining-engine-metrics.json",stats)
        if engine is not None:
            if active_profile:await engine.stop_profile()
            engine.shutdown()
    if a.snapshot_replay:
        root=a.out/"workers"
        if list(root.glob("capture-failure-rank*.json")):
            raise ValueError("snapshot worker reported a capture failure")
        for snapshot in (PROFILE_ID+"_PREFILL",PROFILE_ID+"_DECODE"):
            for rank in range(8):
                stem=root/"snapshots"/snapshot/f"rank-{rank}"
                record=json.loads(stem.with_suffix(".json").read_text())
                if record["snapshot_sha256"]!=file_sha(stem.with_suffix(".pt")):
                    raise ValueError("snapshot file changed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("baseline","routes","trace"),default="baseline")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--native-tune",type=Path,help="frozen GLM vLLM tuning JSON")
    p.add_argument("--snapshot-replay",action="store_true",help="capture the two fixed S32 layer-23 steps")
    for key in ("tokens","cpu-receipt","manifest","model-dir","source-lock","guard"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.native_tune and not a.snapshot_replay:p.error("--native-tune requires --snapshot-replay")
    if a.snapshot_replay and a.mode!="routes":p.error("--snapshot-replay requires --mode routes")
    if PROFILE_ID!="GLM" and not a.snapshot_replay:p.error("only GLM has the full baseline and route plan; other models capture snapshots")
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        corpus=json.loads(a.corpus.read_text());check=validate_corpus(corpus)
        save(a.out/"local-inspection.json",dict(**check,blocks=schedule(),GPU_executed=False))
        if not a.execute:
            print(json.dumps(check));return
        required=("tokens","cpu_receipt","manifest","model_dir","source_lock")+(() if a.snapshot_replay else ("guard",))
        if any(getattr(a,k) is None for k in required):
            raise ValueError("execution requires all offline receipts"+("" if a.snapshot_replay else " and a fresh platform guard"))
        if a.guard is not None:
            guard=json.loads(a.guard.read_text());cutoff=check_guard(guard,time.time(),os.environ.get("EDMIR_INSTANCE_ID"))
            save(a.out/"guard.json",guard)
            def alarm(signum,frame):raise TimeoutError("collection cutoff: recover partial files before platform deletion")
            signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,cutoff-time.time())
        else:
            # The 2026-11-23 snapshot capture runs on an operator-managed lease without a platform guard.
            save(a.out/"guard.json",dict(schema="edmir.real-moe-gpu-guard.none",reason="operator-managed lease"))
        receipt=json.loads(a.cpu_receipt.read_text())
        if receipt.get("schema")!="edmir.real-moe-cpu-receipt.v1" or receipt.get("status")!="passed":
            raise ValueError("successful fixed-image CPU receipt required")
        if receipt.get("model",MODEL)!=MODEL:raise ValueError("CPU receipt belongs to another model")
        for f,key in ((a.tokens,"tokenized_file_sha256"),(a.corpus,"corpus_file_sha256"),
                      (a.manifest,"manifest_sha256"),(a.source_lock,"source_lock_sha256")):
            if file_sha(f)!=receipt[key]:raise ValueError("offline artifact changed: "+key)
        root=Path(__file__).resolve().parents[1]
        actual={str(f.relative_to(root)):file_sha(f) for f in collection_files(root)}
        if actual!=receipt["collection_source_hashes"]:raise ValueError("collector changed after CPU verification")
        from run_native_moe_smoke import inspect_assets
        manifest=json.loads(a.manifest.read_text())
        if not inspect_assets(manifest,a.model_dir,verify_hashes=True,model_id=MODEL,revision=REVISION)["ASSETS_VERIFIED"]:raise ValueError("assets incomplete")
        packet=json.loads(a.tokens.read_text())
        validate_token_packet(packet,corpus,model_vocab_size(a.model_dir))
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        asyncio.run(execute(a,packet,receipt))
        save(a.out/"completion.json",dict(status="collection_finished",mode=a.mode,
            acceptance_analysis_pending=True,predictor_modified=False,edmir_integrated=False))
    except BaseException as exc:
        save(a.out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        save(a.out/"files.json",{str(f.relative_to(a.out)):file_sha(f) for f in sorted(a.out.rglob("*")) if f.is_file() and f.name!="files.json"})


if __name__=="__main__":main()
