from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir import EDMBuilder, MovementDesc, buffer_ref, detect_device, format_device_spec, param_ref  # noqa: E402
from ir.ops import COMPUTE_QKV, DATA_DEPENDENCY, EVENT_DEPENDENCY, KV_TRANSFER, STREAM_ORDER  # noqa: E402
from ir.printing import format_critical_path, format_summary, format_uops  # noqa: E402


def _load_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except Exception as exc:
        print(f"torch unavailable: {exc}")
        return None
    return torch


@contextmanager
def nvtx_range(torch: Any, name: str):
    try:
        torch.cuda.nvtx.range_push(name)
        yield
    finally:
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass


def _event_pair(torch: Any) -> tuple[Any, Any]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _elapsed_us(start: Any, end: Any) -> float:
    return float(start.elapsed_time(end)) * 1000.0


def run_trace(matrix: int, warmup: int, dtype_name: str) -> None:
    torch = _load_torch()
    if torch is None:
        return

    if not torch.cuda.is_available():
        print("CUDA unavailable; trace.py requires a CUDA device for real ops.")
        print(format_device_spec(detect_device(0)))
        return

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    spec = detect_device(0)
    device_label = "cuda:0"
    if spec is not None and spec.cuda_arch:
        device_label = f"{device_label}/{spec.cuda_arch}"

    print("== device ==")
    print(format_device_spec(spec))

    torch.cuda.set_device(0)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    a_cpu = torch.randn((matrix, matrix), dtype=dtype, pin_memory=True)
    a_gpu = torch.empty((matrix, matrix), dtype=dtype, device="cuda")
    b_gpu = torch.randn((matrix, matrix), dtype=dtype, device="cuda")

    for _ in range(warmup):
        a_gpu.copy_(a_cpu, non_blocking=True)
        _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    copy_start, copy_end = _event_pair(torch)
    wait_start, wait_end = _event_pair(torch)
    compute_start, compute_end = _event_pair(torch)
    copy_done = torch.cuda.Event()

    result = None
    with nvtx_range(torch, "edm.phase:trace"):
        with torch.cuda.stream(copy_stream):
            with nvtx_range(torch, "edm.kv:transfer.h2d"):
                copy_start.record(copy_stream)
                a_gpu.copy_(a_cpu, non_blocking=True)
                copy_done.record(copy_stream)
                copy_end.record(copy_stream)

        with torch.cuda.stream(compute_stream):
            with nvtx_range(torch, "edm.event:wait.copy_done"):
                wait_start.record(compute_stream)
                compute_stream.wait_event(copy_done)
                wait_end.record(compute_stream)

            with nvtx_range(torch, "edm.compute:qkv.matmul"):
                compute_start.record(compute_stream)
                result = torch.matmul(a_gpu, b_gpu)
                compute_end.record(compute_stream)

    compute_end.synchronize()
    if result is None:
        raise RuntimeError("trace workload did not produce a result")

    copy_us = _elapsed_us(copy_start, copy_end)
    wait_us = _elapsed_us(wait_start, wait_end)
    compute_us = _elapsed_us(compute_start, compute_end)
    bytes_moved = a_cpu.numel() * a_cpu.element_size()

    print("\n== raw CUDA timing ==")
    print(f"h2d_copy_us={copy_us:.3f}")
    print(f"event_wait_us={wait_us:.3f}")
    print(f"matmul_us={compute_us:.3f}")
    print(f"copy_bytes={bytes_moved}")

    graph = build_graph(
        copy_us=copy_us,
        compute_us=compute_us,
        bytes_moved=bytes_moved,
        device=device_label,
        matrix=matrix,
        dtype=dtype_name,
        wait_us=wait_us,
    )

    critical_path, _ = graph.critical_path()
    print("\n== EDM UOps ==")
    print(format_uops(graph, highlight=critical_path))
    print(format_critical_path(graph))
    print(format_summary(graph))


def build_graph(
    *,
    copy_us: float,
    compute_us: float,
    bytes_moved: int,
    device: str,
    matrix: int,
    dtype: str,
    wait_us: float,
):
    builder = EDMBuilder(default_phase="trace")
    t0 = 0.0
    t1 = copy_us
    t2 = copy_us + compute_us

    begin = builder.phase_begin("trace", t0)
    h2d = builder.movement(
        "h2d_activation_copy",
        t0,
        t1,
        op=KV_TRANSFER,
        stream="copy",
        device=device,
        movement=MovementDesc(kind="h2d_copy", backend="cuda", size_bytes=bytes_moved),
        reads=(buffer_ref("host_activation"),),
        writes=(buffer_ref("device_activation", "write", location=device),),
        attrs={"matrix": matrix, "dtype": dtype},
    )
    record = builder.record_event(
        "copy_done",
        t1,
        stream="copy",
        device=device,
    )
    wait = builder.wait_event(
        "wait_copy_done",
        t1,
        stream="compute",
        device=device,
        attrs={"measured_wait_us": wait_us},
    )
    matmul = builder.compute(
        "qkv_matmul",
        t1,
        t2,
        op=COMPUTE_QKV,
        stream="compute",
        device=device,
        reads=(buffer_ref("device_activation", location=device), param_ref("qkv_weight", location=device)),
        writes=(buffer_ref("qkv_output", "write", location=device),),
        attrs={"matrix": matrix, "dtype": dtype},
    )
    end = builder.phase_end("trace", t2)

    builder.sequence(begin, h2d, record, kind=STREAM_ORDER)
    builder.edge(record, wait, EVENT_DEPENDENCY, event="copy_done")
    builder.edge(h2d, matmul, DATA_DEPENDENCY, data_id="device_activation")
    builder.sequence(wait, matmul, end, kind=STREAM_ORDER)
    return builder.build()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real CUDA op and print EDM-IR UOps.")
    parser.add_argument("--matrix", type=int, default=2048, help="square matrix size for H2D copy and matmul")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations before measured trace")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="tensor dtype for the trace workload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_trace(matrix=args.matrix, warmup=args.warmup, dtype_name=args.dtype)


if __name__ == "__main__":
    main()
