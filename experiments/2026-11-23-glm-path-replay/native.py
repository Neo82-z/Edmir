"""Pinned vLLM native replay and a bounded gate/up tuning job."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics


def module():
    if importlib.metadata.version("vllm") != "0.20.1":
        raise ValueError("native gate/up requires vLLM 0.20.1")
    return importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")


# Local experts, gate/up width N, hidden K, top-k and routed experts of each model's first projection.
SHAPES = {"GLM":(8,3072,2048,4,64),"QWEN35":(32,1024,2048,8,256),"DSV2L":(8,2816,2048,6,64)}


def config_name(fm, width):
    """vLLM names the tuned file by local experts and N/2 (the intermediate size)."""
    local,n,*_ = SHAPES[width]
    return fm.get_config_file_name(local,n//2,None)


def invoke(fm, a, b, c, alignment, config, top_k):
    import triton.language as tl

    ids,experts,post = alignment
    fm.invoke_fused_moe_triton_kernel(
        A=a,B=b,C=c,A_scale=None,B_scale=None,topk_weights=None,sorted_token_ids=ids,
        expert_ids=experts,num_tokens_post_padded=post,mul_routed_weight=False,top_k=top_k,
        config=config,compute_type=tl.bfloat16,use_fp8_w8a8=False,use_int8_w8a8=False,
        use_int8_w8a16=False,use_int4_w4a16=False,per_channel_quant=False,block_shape=None,B_bias=None)


def synthetic_align(ids, config, rank, experts):
    import torch
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    local = experts//8
    emap = torch.tensor([e%local if e//local==rank else -1 for e in range(experts)],
                        dtype=torch.int32,device=ids.device)
    return moe_align_block_size(ids,config["BLOCK_SIZE_M"],experts,emap,ignore_invalid_experts=False)


def tune(out, width):
    import torch
    import triton

    fm = module()
    if torch.cuda.device_count()!=8 or "A100-SXM4-80GB" not in torch.cuda.get_device_name(0):
        raise ValueError("tuning is for the registered eight-card SM80 host")
    local,n,k,top_k,experts = SHAPES[width]
    out.mkdir(parents=True,exist_ok=True)
    dest,record_path = out/config_name(fm,width),out/f"trials-{width}.json"
    if dest.exists() or record_path.exists():
        raise FileExistsError(f"{width} was already tuned in {out}")
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda",0)
    gen = torch.Generator(device=device).manual_seed(9123)
    w = (torch.randn((local,n,k),device=device,generator=gen)*.125).bfloat16()
    configs, trials = {},{}
    for m in (16,32,128,512,2048,8192,16384):
        a = (torch.randn((m,k),device=device,generator=gen)*.125).bfloat16()
        c = torch.empty((m,top_k,n),device=device,dtype=torch.bfloat16)
        # Synthetic untimed-tuning routes, with uniform expert coverage and unique top-k.
        ids = (torch.randint(experts,(m,1),device=device,generator=gen)
               +torch.arange(0,7*top_k,7,device=device)).remainder(experts).int()
        default = fm.get_default_config(m,local,n//2,k,top_k,None,None)
        candidates = [default]
        for bm,bn,bk,warps in ((16,64,64,4),(32,128,64,4),(64,128,32,4)):
            variant = dict(default,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,num_warps=warps,
                           GROUP_SIZE_M=4,num_stages=3)
            if variant not in candidates:
                candidates.append(variant)
        mine = (ids<local).nonzero()[:32]
        if not len(mine):
            raise ValueError("synthetic tuner produced no local assignments")
        rows,slots = mine.unbind(1)
        expected = torch.bmm(a[rows,None].float(),w[ids[rows,slots].long()].float().transpose(1,2)).squeeze(1)
        trials[m] = []
        for cfg in candidates:
            record = dict(config=cfg)
            try:
                alignment = synthetic_align(ids,cfg,0,experts)
                for _ in range(3):
                    invoke(fm,a,w,c,alignment,cfg,top_k)
                got = c[rows,slots].float()
                error = ((got-expected).abs()/expected.abs().clamp_min(1.)).max().item()
                if not torch.isfinite(got).all().item() or error>.02:
                    raise ValueError("tuning candidate failed numerical gate")
                stamps = []
                for _ in range(10):
                    start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    invoke(fm,a,w,c,alignment,cfg,top_k)
                    end.record()
                    end.synchronize()
                    stamps.append(start.elapsed_time(end))
                record.update(event_ms=statistics.median(stamps),error=error)
            except Exception as exc:
                record["failure"]=repr(exc)
            trials[m].append(record)
        eligible = [r for r in trials[m] if "event_ms" in r]
        if eligible:
            configs[m] = min(eligible,key=lambda r:r["event_ms"])["config"]
    record_path.write_text(json.dumps(trials,indent=2,allow_nan=False)+"\n")
    if configs.keys()!=trials.keys():
        raise RuntimeError("tuned config is incomplete; preserve trials and label default capture")
    dest.write_text(json.dumps({"triton_version":triton.__version__,**configs},indent=2)+"\n")
    print(dest)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True,help="one folder for all models; vLLM reads it as the tuned-config folder")
    p.add_argument("--width",choices=sorted(SHAPES),required=True)
    a = p.parse_args()
    tune(a.out,a.width)
