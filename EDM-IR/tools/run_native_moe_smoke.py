#!/usr/bin/env python3
"""Native vLLM readiness/first-inference packet, separate from EDM-IR timing.

Uses the documented 0.20.1 routed-experts output in diagnostic mode. This
request/token/layer record is NOT an EP-rank or scheduler-step shape trace.
Default operation is a CPU-only manifest review. No automatic downloads.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time

MODEL_ID="zai-org/GLM-4.7-Flash"
REVISION="7dd20894a642a0aa287e9827cb1a1f7f91386b67"
SCHEMA="edmir.native-moe-first-run.v1"


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_parallelism(config,tp=4,dp=2):
    if tp*dp!=8 or config['num_attention_heads']%tp or config['n_routed_experts']%(tp*dp):
        raise ValueError('attention heads / expert count do not divide declared TP / EP')
    return dict(tensor_parallel_size=tp,data_parallel_size=dp,data_parallel_size_local=dp,
                data_parallel_backend='mp',enable_expert_parallel=True)


def engine_arguments(model_dir, *, diagnostic=False):
    parallel=validate_parallelism(json.loads((model_dir/'config.json').read_text()))
    return dict(model=str(model_dir),tokenizer=str(model_dir),dtype='bfloat16',
        **parallel,
        enable_eplb=False,enable_prefix_caching=False,enable_return_routed_experts=diagnostic,
        trust_remote_code=False,enforce_eager=True,max_model_len=4096,max_num_seqs=8,
        max_num_batched_tokens=4096,kv_cache_memory_bytes=2*1024**3,
        moe_backend='triton',all2all_backend='allgather_reducescatter',
        attention_config={'backend':'TRITON_MLA','flash_attn_version':2,
                          'disable_flashinfer_prefill':True},
        enable_chunked_prefill=True,gpu_memory_utilization=.8,seed=20261116)


def inspect_assets(manifest,model_dir,*,verify_hashes=False,model_id=MODEL_ID,revision=REVISION):
    if manifest['id']!=model_id or manifest['revision']!=revision:
        raise ValueError('different model/revision needs its own plan')
    missing=[];verified=[]
    for row in manifest['weight_files']:
        name=row['rfilename'];path=model_dir/name
        if Path(name).name!=name:raise ValueError('only flat model shard paths supported')
        if not path.is_file():missing.append(name);continue
        if path.stat().st_size!=row['size']:raise ValueError('wrong shard size: '+name)
        if verify_hashes:
            if sha(path)!=row['lfs']['sha256']:raise ValueError('wrong shard digest: '+name)
            verified.append(name)
    for name,key in [('config.json','config.json_sha256'),('model.safetensors.index.json','model.safetensors.index.json_sha256')]:
        path=model_dir/name
        if not path.is_file():missing.append(name)
        elif sha(path)!=manifest[key]:raise ValueError('wrong model metadata: '+name)
    auxiliary=manifest.get('auxiliary_hashes',{})
    if not all(name in auxiliary for name in ('tokenizer.json','tokenizer_config.json')):
        raise ValueError('pinned tokenizer hashes are required')
    for name,expected in auxiliary.items():
        if Path(name).name!=name:raise ValueError('only flat model metadata paths supported')
        path=model_dir/name
        if not path.is_file():
            if name not in missing:missing.append(name)
        elif sha(path)!=expected:raise ValueError('wrong pinned auxiliary file: '+name)
    return dict(missing=missing,weights_present=not any(x.endswith('.safetensors') for x in missing),
                verified_weight_files=len(verified),weight_bytes=manifest['weight_file_bytes'],
                ASSETS_VERIFIED=not missing and len(verified)==len(manifest['weight_files']),
                GPU_runtime_verified=False)


def check_routes(routed,prompt_count,generated_count,*,experts=64,top_k=4):
    import numpy as np
    a=np.asarray(routed)
    expected=prompt_count+generated_count-1
    if a.ndim!=3 or a.shape!=(expected,47,top_k):
        raise ValueError('unrecognized routed-experts layout; preserve raw array for review')
    if not np.issubdtype(a.dtype,np.integer):raise ValueError('expert IDs must be integral')
    # vLLM 0.20.1 allocates all 47 layers with zeros; only FusedMoE routers
    # write their decoder layer IDs. GLM layer 0 is dense; layers 1..46 route.
    if np.any(a[:,0,:]!=0):raise ValueError('unexpected dense-layer placeholder')
    moe=a[:,1:,:]
    if np.any(moe<0) or np.any(moe>=experts):raise ValueError('expert IDs out of range')
    if np.any(np.diff(np.sort(moe,axis=-1),axis=-1)==0):raise ValueError('duplicate experts in token top-k')
    return dict(shape=list(a.shape),routed_assignments=int(moe.size),
                input_tokens_represented=expected,layer_axis_mapping=list(range(47)),
                dense_layers_excluded=[0],moe_layer_ids=list(range(1,47)),
                scheduler_step_and_source_rank_available=False,
                represents_padded_kernel_M=False)


async def execute(a,manifest,requests):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    import torch
    import numpy as np
    if importlib.metadata.version('vllm')!='0.20.1':raise ValueError('vLLM version differs from pinned image')
    receipt=json.loads(a.preflight_receipt.read_text())
    for name in ('transformers','tokenizers'):
        if importlib.metadata.version(name)!=receipt['runtime'][name]:
            raise ValueError(name+' differs from the CPU-tested image')
    if torch.__version__!=receipt['runtime']['torch'] or torch.version.cuda!=receipt['runtime']['cuda_build']:
        raise ValueError('PyTorch/CUDA differs from the CPU-tested image')
    if torch.cuda.device_count()!=8:raise ValueError('this first native packet declares eight GPUs')
    devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(8)]
    if any(d['capability']!=[8,0] or 'A100' not in d['name'] for d in devices):
        raise ValueError('this packet is for A100 SM80')
    diagnostic=a.mode=='routes'
    config=engine_arguments(a.model_dir,diagnostic=diagnostic)
    save(a.out/'runtime-intent.json',dict(model_id=MODEL_ID,revision=REVISION,
        engine_version=importlib.metadata.version('vllm'),torch_version=torch.__version__,cuda=torch.version.cuda,
        devices=devices,engine_args=config,mode=a.mode,diagnostic_only=diagnostic,
        performance_scope='native eager first-inference baseline; not tuned serving or EDM-IR speedup',
        tokenizer_files={name:sha(a.model_dir/name) for name in ('tokenizer.json','tokenizer_config.json')},
        rank_topology_verified=False))
    started=time.perf_counter();engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    save(a.out/'load.json',dict(load_seconds=time.perf_counter()-started))
    async def one(r,run_id):
        begin=time.perf_counter();first=None;final=None;last_count=0;updates=[]
        params=SamplingParams(temperature=0,max_tokens=64)
        try:
            async for o in engine.generate({'prompt_token_ids':r['prompt_token_ids']},params,run_id):
                now=time.perf_counter();final=o;count=len(o.outputs[0].token_ids)
                if count>last_count:
                    if first is None:first=now
                    updates.append(dict(seconds=now-begin,cumulative_tokens=count));last_count=count
            if final is None or last_count==0 or not final.finished:raise ValueError('empty or incomplete generation')
            if list(final.prompt_token_ids)!=r['prompt_token_ids']:
                raise ValueError('engine prompt tokens differ from the CPU-frozen chat template')
            end=time.perf_counter();completion=final.outputs[0]
            record=dict(request_id=run_id,origin='native_vllm',prompt=r['prompt'],
                input_tokens=len(final.prompt_token_ids),output_tokens=last_count,
                output_token_ids=list(completion.token_ids),text=completion.text,
                input_token_ids=list(final.prompt_token_ids),chat_template_applied=True,
                finished=final.finished,finish_reason=completion.finish_reason,success=True,
                output_updates=updates,diagnostic_only=diagnostic)
            if not diagnostic:
                record.update(wall_seconds=end-begin,observed_stream_ttft_seconds=first-begin,
                    observed_stream_tpot_seconds=(updates[-1]['seconds']-(first-begin))/(last_count-1) if last_count>1 else None)
            if diagnostic:
                raw=completion.routed_experts
                if raw is None:raise ValueError('engine returned no routed experts')
                # Preserve original evidence before validation, even if layout is unsupported.
                np.save(a.out/(run_id+'-routes.npy'),np.asarray(raw),allow_pickle=False)
                record['routes']=check_routes(raw,len(final.prompt_token_ids),last_count)
            return record
        except BaseException as exc:
            save(a.out/(run_id+'-failure.json'),dict(request_id=run_id,error=repr(exc),success=False,
                elapsed_seconds=time.perf_counter()-begin,output_updates=updates))
            raise
    try:
        # Exercise the complete concurrent workload before measuring. One
        # request can leave the second DP replica cold under internal routing.
        for warm_repeat in range(1 if diagnostic else 2):
            warmups=await asyncio.gather(*(one(r,f'warmup-{warm_repeat}-{i:03d}')
                                          for i,r in enumerate(requests)))
            for warmup in warmups:save(a.out/(warmup['request_id']+'.json'),warmup)
        results=[]
        for repeat in range(3 if not diagnostic else 1):
            begin=time.perf_counter()
            batch=await asyncio.gather(*(one(r,f'r{repeat}-{i:03d}') for i,r in enumerate(requests)))
            record=dict(repetition=repeat,requests=len(batch),output_tokens=sum(x['output_tokens'] for x in batch),diagnostic_only=diagnostic)
            if not diagnostic:
                record['batch_wall_seconds']=time.perf_counter()-begin
                record['effective_output_tokens_per_second']=record['output_tokens']/record['batch_wall_seconds']
            # Persist after the measured batch so disk writes are outside its boundary.
            for item in batch:save(a.out/(item['request_id']+'.json'),item)
            results.append(record);save(a.out/'batches.json',results)
    finally:engine.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model-dir','requests','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['baseline','routes'],default='baseline')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--verify-hashes',action='store_true')
    p.add_argument('--guard',type=Path)
    p.add_argument('--preflight-receipt',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());packet=json.loads(a.requests.read_text())
    if isinstance(packet,dict):
        if packet.get('schema')!='edmir.native-tokenized-requests.v1' or packet.get('model_revision')!=REVISION:
            raise ValueError('wrong tokenized request contract')
        if packet.get('engine_version')!='0.20.1' or packet.get('tokenizer_hashes')!=manifest['auxiliary_hashes']:
            raise ValueError('request tokenizer/engine differs from pinned assets')
        requests=packet['requests']
    else:
        requests=packet
        if a.execute:raise ValueError('CPU-verified tokenized request packet required before GPU execution')
    if not 1<=len(requests)<=8 or any(not isinstance(r.get('prompt'),str) or not r['prompt'] or len(r['prompt'])>8000 for r in requests):
        raise ValueError('1..8 bounded explicit requests required')
    if isinstance(packet,dict):
        vocab=json.loads((a.model_dir/'config.json').read_text())['vocab_size']
        for r in requests:
            ids=r.get('prompt_token_ids',[])
            if not 1<=len(ids)<=4096-64 or any(type(i) is not int or not 0<=i<vocab for i in ids):
                raise ValueError('invalid frozen prompt token IDs')
    try:
        if a.execute:
            if a.preflight_receipt is None:raise ValueError('successful offline CPU receipt required')
            receipt=json.loads(a.preflight_receipt.read_text())
            if receipt.get('status')!='passed' or receipt.get('tokenized_requests_sha256')!=sha(a.requests):
                raise ValueError('request packet differs from the successful CPU preflight')
            if receipt.get('native_runner_sha256')!=sha(Path(__file__)):
                raise ValueError('runner changed after the offline CPU preflight')
            if a.guard is None:raise ValueError('fresh platform stop/budget guard required for execution')
            from run_precision_followup import cutoff_from_guard
            guard=json.loads(a.guard.read_text())
            cutoff=cutoff_from_guard(guard,time.time(),schema=SCHEMA)
            save(a.out/'guard.json',guard)
            def timeout_handler(signum,frame):raise TimeoutError('collection deadline reached; recover partial outputs')
            signal.signal(signal.SIGALRM,timeout_handler)
            signal.setitimer(signal.ITIMER_REAL,min(cutoff-time.time(),1200))
        status=inspect_assets(manifest,a.model_dir,verify_hashes=a.verify_hashes or a.execute)
        status.update(mode=a.mode,GPU_executed=False,request_manifest_sha256=sha(a.requests))
        save(a.out/'readiness.json',status)
        if not a.execute:
            print(json.dumps(status));return
        if not status['ASSETS_VERIFIED']:raise ValueError('model assets incomplete; no online download during GPU execution')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        asyncio.run(execute(a,manifest,requests))
        signal.setitimer(signal.ITIMER_REAL,0)
        save(a.out/'completion.json',dict(state='native_run_complete',mode=a.mode,
            EDMIR_integrated=False,rank_step_shape_trace=False,independent_generalization=False))
    except BaseException as exc:
        save(a.out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':main()
