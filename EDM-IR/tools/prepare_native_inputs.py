#!/usr/bin/env python3
"""Run inside the pinned image with --network none and without --gpus."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import traceback

from run_native_moe_smoke import engine_arguments, REVISION, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','manifest','requests','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','GPU_used':False,'stages':{}}
    try:
        if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
            raise ValueError('offline environment required')
        if list(Path('/dev').glob('nvidia[0-9]*')):
            raise ValueError('GPU devices must not be exposed to preparation')
        versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
        save(a.out/'packages.json',dict(sorted(versions.items())))
        if importlib.metadata.version('vllm')!='0.20.1':raise ValueError('wrong engine version')
        import torch
        if torch.cuda.is_available():raise ValueError('CPU-only preparation required')
        report['runtime']={'machine':platform.machine(),'python':platform.python_version(),
            'torch':torch.__version__,'cuda_build':torch.version.cuda,
            'compiled_cuda_arches':getattr(torch._C,'_cuda_getArchFlags',lambda:None)(),
            'transformers':importlib.metadata.version('transformers'),
            'tokenizers':importlib.metadata.version('tokenizers')}
        report['native_runner_sha256']=sha(Path(__file__).with_name('run_native_moe_smoke.py'))
        manifest=json.loads(a.manifest.read_text())
        for name, expected in manifest['auxiliary_hashes'].items():
            if sha(a.model_dir/name)!=expected:raise ValueError('metadata hash mismatch: '+name)
        from transformers import AutoConfig, AutoTokenizer
        from tokenizers import Tokenizer
        cfg=AutoConfig.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
        if cfg.model_type!='glm4_moe_lite':raise ValueError('unexpected architecture')
        report['stages']['config']={'class':type(cfg).__name__,'dtype':str(cfg.dtype),
            'architecture':cfg.architectures,'layers':cfg.num_hidden_layers,'experts':cfg.n_routed_experts}
        raw=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
        hf=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
        from vllm.tokenizers.registry import get_tokenizer
        vt=get_tokenizer(str(a.model_dir),trust_remote_code=False,local_files_only=True)
        original=json.loads(raw.to_str())['pre_tokenizer']
        for tokenizer in (hf,vt):
            if json.loads(tokenizer.backend_tokenizer.to_str())['pre_tokenizer']!=original:
                raise ValueError('engine changed the pinned pre-tokenizer regex')
        texts=['Hello, world!', '中文分词：批处理与专家。', '  leading  and trailing  ',
               'a\nb\r\nc\tend', '1234567890 0.001 -1e-4', "I'm testing; we've tested.",
               'def f(x):\n    return x ** 2\n', 'e\u0301 é 👩🏽‍💻 🌏',
               'M=64, N=2048, K=4096', '', '[gMASK]<sop><|user|>你好<|assistant|></think>']
        tests=[]
        for s in texts:
            raw_ids=raw.encode(s,add_special_tokens=False).ids
            hf_ids=hf.encode(s,add_special_tokens=False)
            vllm_ids=vt.encode(s,add_special_tokens=False)
            if raw_ids!=hf_ids or raw_ids!=vllm_ids:raise ValueError('tokenizer implementations disagree')
            if hf.decode(hf_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=s:
                raise ValueError('tokenizer text round-trip changed content')
            tests.append({'text':s,'ids':raw_ids})
        report['stages']['tokenizer']={'class':type(hf).__name__,'vllm_class':type(vt).__name__,
            'pre_tokenizer_unchanged':True,'raw_hf_vllm_equal':True,
            'manual_mistral_regex_patch_applied':False,'tests':tests}
        requests=json.loads(a.requests.read_text());prepared=[]
        for row in requests:
            messages=[{'role':'user','content':row['prompt']}]
            rendered=hf.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=hf.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if not rendered.endswith('<|assistant|></think>'):raise ValueError('unexpected GLM generation prefix')
            if ids!=vt.encode(rendered,add_special_tokens=False) or ids!=raw.encode(rendered,add_special_tokens=False).ids:
                raise ValueError('chat template and tokenizer paths disagree')
            if not 1<=len(ids)<=4096-64:raise ValueError('request exceeds fixed context limit')
            prepared.append({**row,'chat_prompt':rendered,'prompt_token_ids':ids})
        from vllm.engine.arg_utils import AsyncEngineArgs
        fields={f.name for f in dataclasses.fields(AsyncEngineArgs)}
        args=engine_arguments(a.model_dir)
        if not args.keys()<=fields:raise ValueError('unsupported engine arguments: '+str(args.keys()-fields))
        engine_args=AsyncEngineArgs(**args)
        diagnostic=AsyncEngineArgs(**engine_arguments(a.model_dir,diagnostic=True))
        if not diagnostic.enable_return_routed_experts:raise ValueError('routes argument not accepted')
        from vllm.model_executor.models import ModelRegistry
        if 'Glm4MoeLiteForCausalLM' not in ModelRegistry.get_supported_archs():
            raise ValueError('model architecture not registered')
        report['stages']['engine_api']={'arguments':args,'constructor_passed':True,
            'GPU_model_or_distributed_processes_created':False,'architecture_registered':True}
        # Bind these source files to the actual image; used to audit dense-layer handling.
        import vllm
        source_root=Path(vllm.__file__).parent
        report['engine_source_hashes']={name:sha(source_root/name) for name in (
            'model_executor/models/glm4_moe_lite.py',
            'model_executor/layers/fused_moe/routed_experts_capturer.py',
            'v1/worker/gpu_model_runner.py')}
        packet={'schema':'edmir.native-tokenized-requests.v1','model_revision':REVISION,
            'engine_version':'0.20.1','tokenizer_hashes':manifest['auxiliary_hashes'],
            'source_requests_sha256':sha(a.requests),
            'chat_template_options':{'add_generation_prompt':True,'enable_thinking':False},
            'requests':prepared}
        save(a.out/'native-requests-tokenized.json',packet)
        report['tokenized_requests_sha256']=sha(a.out/'native-requests-tokenized.json')
        report['stages']['requests']={'count':len(prepared),'input_token_counts':[len(r['prompt_token_ids']) for r in prepared]}
        report['status']='passed'
        report['remaining_GPU_checks']=['NVIDIA driver compatibility','NCCL topology and TP/DP/EP startup',
            'SM80 attention/MoE kernel execution and JIT warmup','actual weight loading and generation',
            'actual routed-expert buffers and distributed correctness']
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        save(a.out/'cpu-engine-preflight.json',report)


if __name__=='__main__':main()
