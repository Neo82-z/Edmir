from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir import EDMBuilder, EDMGraph, MovementDesc, buffer_ref, const_ref, param_ref
from ir.ops import (
    COMM_ALL_GATHER,
    COMPUTE_ATTN,
    COMPUTE_MLP,
    COMPUTE_QKV,
    DATA_DEPENDENCY,
    EVENT_DEPENDENCY,
    KV_TRANSFER,
    PHASE_BOUNDARY,
    RUNTIME_LAUNCH_GAP,
    STREAM_ORDER,
)
from ir.printing import format_critical_path, format_summary, format_uops


def hidden_communication() -> EDMGraph:
    """Communication is present, but it is hidden behind longer compute."""

    b = EDMBuilder(default_phase="decode")
    begin = b.phase_begin("decode", 0.0)
    qkv = b.compute(
        "qkv",
        0.0,
        30.0,
        op=COMPUTE_QKV,
        stream="compute",
        reads=(buffer_ref("hidden"), param_ref("qkv_weight"), const_ref("position")),
        writes=(buffer_ref("qkv_partial", "write"),),
    )
    comm = b.comm(
        "tp_all_gather",
        5.0,
        25.0,
        op=COMM_ALL_GATHER,
        stream="comm",
        movement=MovementDesc(kind="collective", size_bytes=64 << 20, group="tp"),
        reads=(buffer_ref("qkv_partial"),),
        writes=(buffer_ref("qkv_gathered", "write"),),
    )
    attn = b.compute("attention", 30.0, 52.0, op=COMPUTE_ATTN, stream="compute")
    mlp = b.compute("mlp", 52.0, 92.0, op=COMPUTE_MLP, stream="compute")
    end = b.phase_end("decode", 92.0)

    b.sequence(begin, qkv, attn, mlp, end, kind=STREAM_ORDER)
    b.bind_phase(begin, end, comm)
    return b.build()


def exposed_communication() -> EDMGraph:
    """A communication result gates downstream attention work."""

    b = EDMBuilder(default_phase="decode")
    begin = b.phase_begin("decode", 0.0)
    qkv = b.compute(
        "qkv",
        0.0,
        30.0,
        op=COMPUTE_QKV,
        stream="compute",
        reads=(buffer_ref("hidden"), param_ref("qkv_weight"), const_ref("position")),
        writes=(buffer_ref("qkv_partial", "write"),),
    )
    comm = b.comm(
        "tp_all_gather",
        30.0,
        55.0,
        op=COMM_ALL_GATHER,
        stream="comm",
        movement=MovementDesc(kind="collective", size_bytes=64 << 20, group="tp"),
        reads=(buffer_ref("qkv_partial"),),
        writes=(buffer_ref("qkv_gathered", "write"),),
    )
    attn = b.compute("attention", 55.0, 77.0, op=COMPUTE_ATTN, stream="compute")
    mlp = b.compute("mlp", 77.0, 117.0, op=COMPUTE_MLP, stream="compute")
    end = b.phase_end("decode", 117.0)

    b.sequence(begin, qkv, kind=STREAM_ORDER)
    b.edge(qkv, comm, DATA_DEPENDENCY, data_id="qkv_partial")
    b.edge(comm, attn, DATA_DEPENDENCY, data_id="qkv_gathered")
    b.sequence(attn, mlp, end, kind=STREAM_ORDER)
    return b.build()


def harmful_overlap() -> EDMGraph:
    """Overlap adds event and KV-transfer work that still gates the request."""

    b = EDMBuilder(default_phase="decode")
    begin = b.phase_begin("decode", 0.0)
    qkv = b.compute(
        "qkv",
        0.0,
        30.0,
        op=COMPUTE_QKV,
        stream="compute",
        reads=(buffer_ref("hidden"), param_ref("qkv_weight"), const_ref("position")),
        writes=(buffer_ref("qkv_partial", "write"),),
    )
    record = b.record_event("record_comm_ready", 30.0, stream="compute")
    kv = b.movement(
        "remote_kv_fetch",
        30.0,
        48.0,
        op=KV_TRANSFER,
        stream="kv",
        reads=(buffer_ref("remote_kv"),),
        writes=(buffer_ref("local_kv", "write"),),
    )
    wait = b.wait_event("wait_kv_ready", 48.0, stream="compute")
    gap = b.uop(RUNTIME_LAUNCH_GAP, "handoff_gap", 48.0, 55.0, stream="runtime")
    attn = b.compute("attention", 55.0, 77.0, op=COMPUTE_ATTN, stream="compute")
    mlp = b.compute("mlp", 77.0, 117.0, op=COMPUTE_MLP, stream="compute")
    end = b.phase_end("decode", 117.0)

    b.sequence(begin, qkv, record, kind=STREAM_ORDER)
    b.edge(record, kv, EVENT_DEPENDENCY, event="comm_ready")
    b.edge(kv, wait, EVENT_DEPENDENCY, event="kv_ready")
    b.sequence(wait, gap, attn, mlp, end, kind=STREAM_ORDER)
    b.edge(begin, kv, PHASE_BOUNDARY)
    b.edge(kv, end, PHASE_BOUNDARY)
    return b.build()


def summarize(name: str, graph: EDMGraph) -> None:
    print(f"\n== {name} ==")
    critical_path, _ = graph.critical_path()
    print(format_uops(graph, highlight=critical_path))
    print(format_critical_path(graph))
    print(format_summary(graph))


def main() -> None:
    summarize("hidden communication", hidden_communication())
    summarize("exposed communication", exposed_communication())
    summarize("harmful overlap", harmful_overlap())


if __name__ == "__main__":
    main()
