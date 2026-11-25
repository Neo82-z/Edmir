#!/usr/bin/env python3
"""Offline preparation inside the pinned remote CPU image; no asset downloading."""
import argparse
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.real_moe import (ENGINE, MODEL, PROFILE, PROFILE_ID, REVISION, file_sha, model_vocab_size,
    save, tokenize_corpus, validate_corpus, validate_token_packet)
from edmir.benchmark.real_moe_config import engine_arguments, collection_files
from edmir.benchmark.vllm0201_capture import verify_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("model-dir","manifest","source-lock","corpus","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema="edmir.real-moe-cpu-receipt.v1",status="failed",GPU_used=False,
                model=MODEL,model_revision=REVISION,assets_ready=False)
    try:
        if platform.system()!="Linux" or platform.machine() not in ("x86_64","AMD64"):
            raise ValueError("run only in the pinned Linux x86_64 CPU environment")
        if os.environ.get("HF_HUB_OFFLINE")!="1" or os.environ.get("TRANSFORMERS_OFFLINE")!="1":
            raise ValueError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 required")
        if list(Path('/dev').glob('nvidia[0-9]*')):raise ValueError("GPU exposure forbidden for CPU preparation")
        from run_native_moe_smoke import inspect_assets
        manifest=json.loads(a.manifest.read_text())
        status=inspect_assets(manifest,a.model_dir,verify_hashes=True,model_id=MODEL,revision=REVISION)
        if not status['ASSETS_VERIFIED']:raise ValueError("missing/hash-invalid assets; no download fallback")
        import torch
        import vllm
        if torch.cuda.is_available():raise ValueError("GPU must not be available")
        if importlib.metadata.version("vllm")!=ENGINE:raise ValueError("wrong engine")
        nsys=shutil.which("nsys")
        if nsys is None:raise ValueError("Nsight Systems must be staged in the offline CPU environment first")
        report["nsys_version"]=subprocess.run([nsys,"--version"],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
        nsys_help=subprocess.run([nsys,"profile","--help"],check=True,capture_output=True,text=True,timeout=10).stdout
        if not all(s in nsys_help for s in ("capture-range","cuda-graph-trace","repeat")):
            raise ValueError("Nsight capture/repeat/graph options unavailable")
        arches=getattr(torch._C,"_cuda_getArchFlags",lambda:"")() or ""
        if "sm_80" not in arches:raise ValueError("image has no declared PyTorch SM80 binary")
        report["runtime"]={name:importlib.metadata.version(name) for name in ("vllm","torch","transformers","tokenizers")}
        report["runtime"].update(cuda_build=torch.version.cuda,compiled_arches=arches)
        lock=json.loads(a.source_lock.read_text())
        report["verified_engine_sources"]=verify_sources(Path(vllm.__file__).parent.parent,lock)
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
        from vllm.tokenizers.registry import get_tokenizer
        hf=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
        vt=get_tokenizer(str(a.model_dir),local_files_only=True,trust_remote_code=False)
        raw=Tokenizer.from_file(str(a.model_dir/"tokenizer.json"))
        original=json.loads(raw.to_str())["pre_tokenizer"]
        for tokenizer in (hf,vt):
            if json.loads(tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]!=original:
                raise ValueError("pinned pre-tokenizer was modified")
        tests=["中文，English 1234.5", "  spaces\nline\tend", "é e\u0301 👩🏽‍💻", "def f(x):\n    return x + 1"]
        for text in tests:
            if not raw.encode(text,add_special_tokens=False).ids==hf.encode(text,add_special_tokens=False)==vt.encode(text,add_special_tokens=False):
                raise ValueError("raw/HF/vLLM tokenizers disagree")
        corpus=json.loads(a.corpus.read_text());validate_corpus(corpus)
        for row in corpus["requests"]:
            messages=[dict(role="user",content=row["content"])]
            opts=dict(add_generation_prompt=True,enable_thinking=False,return_dict=False)
            text=hf.apply_chat_template(messages,tokenize=False,**opts)
            ids=hf.apply_chat_template(messages,tokenize=True,**opts)
            if not ids==raw.encode(text,add_special_tokens=False).ids==vt.encode(text,add_special_tokens=False):
                raise ValueError("full frozen request tokenization differs across tokenizer implementations")
        packet=tokenize_corpus(corpus,hf)
        vocab=model_vocab_size(a.model_dir)
        validate_token_packet(packet,corpus,vocab)
        functional=[]
        for r in corpus["requests"][:8]:
            text=r["content"].split("\n",1)[0]
            functional.append(dict(id=r["id"],prompt_token_ids=hf.apply_chat_template(
                [dict(role="user",content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)))
        packet["functional_requests"]=functional
        save(a.out/"requests-tokenized.json",packet)
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.model_executor.models import ModelRegistry
        from vllm.v1.engine.async_llm import AsyncLLM
        if PROFILE["architecture"] not in ModelRegistry.get_supported_archs():raise ValueError("architecture not registered")
        fields={f.name for f in dataclasses.fields(AsyncEngineArgs)}
        for mode in (("baseline","routes","trace") if PROFILE_ID=="GLM" else ("routes",)):
            args=engine_arguments(a.model_dir,mode)
            if not args.keys()<=fields:raise ValueError("unsupported engine fields: "+str(args.keys()-fields))
            AsyncEngineArgs(**args)
        if not all(hasattr(AsyncLLM,name) for name in ("generate","from_engine_args","start_profile","stop_profile","shutdown")):
            raise ValueError("AsyncLLM API changed")
        root=Path(__file__).resolve().parents[1]
        report.update(status="passed",assets_ready=True,manifest_sha256=file_sha(a.manifest),
            source_lock_sha256=file_sha(a.source_lock),corpus_file_sha256=file_sha(a.corpus),
            tokenized_file_sha256=file_sha(a.out/"requests-tokenized.json"),
            collection_source_hashes={str(f.relative_to(root)):file_sha(f) for f in collection_files(root)},
            tokenizer_raw_hf_vllm_agree=True,manual_regex_patch=False,
            remaining=["actual A100 driver/topology", "GPU JIT and Graph capture", "runtime routing and backend hooks"])
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:save(a.out/"cpu-receipt.json",report)


if __name__=="__main__":main()
