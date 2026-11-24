"""Single engine configuration shared by offline preflight and GPU runner."""
from pathlib import Path

from .real_moe import PROFILE


def engine_arguments(model_dir, mode):
    if mode not in ("baseline","routes","trace"):raise ValueError("unknown collection mode")
    args=dict(model=str(Path(model_dir).resolve()),tokenizer=str(Path(model_dir).resolve()),
        dtype="bfloat16",trust_remote_code=False,tensor_parallel_size=1,
        data_parallel_size=8,data_parallel_size_local=8,data_parallel_backend="mp",
        pipeline_parallel_size=1,enable_expert_parallel=True,enable_eplb=False,
        enable_dbo=False,enable_prefix_caching=False,enable_return_routed_experts=False,
        enforce_eager=mode=="routes",max_model_len=4608,max_num_seqs=32,
        max_num_batched_tokens=4096,kv_cache_memory_bytes=16*1024**3,
        moe_backend="triton",all2all_backend="allgather_reducescatter",
        enable_chunked_prefill=True,gpu_memory_utilization=.85,seed=20261117,
        disable_log_stats=False)
    # Per-model settings: the MLA attention backend for GLM and DeepSeek-V2-Lite, text-only for Qwen3.5.
    args.update(PROFILE["engine"])
    if mode=="routes":args["async_scheduling"]=False
    if mode in ("routes","trace"):
        args["worker_cls"]="edmir.benchmark.vllm0201_capture.CaptureWorker"
    if mode=="trace":args["profiler_config"]={"profiler":"cuda"}
    return args


def collection_files(root):
    root=Path(root)
    return [root/p for p in (
        "edmir/benchmark/real_moe.py","edmir/benchmark/real_moe_config.py",
        "edmir/benchmark/vllm0201_capture.py","tools/run_real_moe.py",
        "tools/prepare_real_moe.py","tools/analyze_real_moe.py","tools/run_native_moe_smoke.py")]


def validate_topology(text):
    rows={}
    for line in text.splitlines():
        values=line.split()
        if len(values)>=9 and values[0] in {f"GPU{i}" for i in range(8)}:
            rows[values[0]]=values[1:9]
    if set(rows)!={f"GPU{i}" for i in range(8)}:raise ValueError("eight GPU topology rows required")
    for i in range(8):
        for j in range(8):
            cell=rows[f"GPU{i}"][j]
            if i==j and cell!="X":raise ValueError("malformed topology diagonal")
            if i!=j and not (cell.startswith("NV") and cell[2:].isdigit() and int(cell[2:])>0):
                raise ValueError("all-pair NVLink topology required; PCIe fallback is not this plan")
    return rows
