# EDM-IR op and edge names.
#
# Keep concrete op names as namespace-style strings so dumps remain readable
# and passes can still aggregate by prefix, e.g. "comm.*" or "compute.*".

OpKind = str
EdgeKind = str

PHASE_BEGIN = "phase.begin"
PHASE_END = "phase.end"

REQUEST_ENQUEUE = "request.enqueue"
BATCH_FORM = "batch.form"
UBATCH_SPLIT = "ubatch.split"

METADATA_ATTN = "metadata.attn"
METADATA_KV = "metadata.kv"
METADATA_POSITION = "metadata.position"

KV_LOOKUP = "kv.lookup"
KV_TRANSFER = "kv.transfer"
KV_MATERIALIZE = "kv.materialize"

COMPUTE_QKV = "compute.qkv"
COMPUTE_ATTN = "compute.attn"
COMPUTE_MLP = "compute.mlp"
COMPUTE_GEMM = "compute.gemm"
COMPUTE_OTHER = "compute.other"

COMM_ALL_REDUCE = "comm.all_reduce"
COMM_ALL_GATHER = "comm.all_gather"
COMM_REDUCE_SCATTER = "comm.reduce_scatter"
COMM_ALL_TO_ALL = "comm.all_to_all"
COMM_P2P = "comm.p2p"
COMM_KV_TRANSFER = "comm.kv_transfer"

EVENT_RECORD = "event.record"
EVENT_WAIT = "event.wait"
STREAM_YIELD = "stream.yield"

RUNTIME_LAUNCH_GAP = "runtime.launch_gap"
CONTENTION_RESOURCE = "contention.resource"

STREAM_ORDER = "stream_order"
DATA_DEPENDENCY = "data_dependency"
EVENT_DEPENDENCY = "event_dependency"
COLLECTIVE_DEPENDENCY = "collective_dependency"
SCHEDULER_DEPENDENCY = "scheduler_dependency"
KV_READINESS = "kv_readiness"
PHASE_BOUNDARY = "phase_boundary"
RESOURCE_CONTENTION = "resource_contention"


ALL_OPS = frozenset({
    PHASE_BEGIN,
    PHASE_END,
    REQUEST_ENQUEUE,
    BATCH_FORM,
    UBATCH_SPLIT,
    METADATA_ATTN,
    METADATA_KV,
    METADATA_POSITION,
    KV_LOOKUP,
    KV_TRANSFER,
    KV_MATERIALIZE,
    COMPUTE_QKV,
    COMPUTE_ATTN,
    COMPUTE_MLP,
    COMPUTE_GEMM,
    COMPUTE_OTHER,
    COMM_ALL_REDUCE,
    COMM_ALL_GATHER,
    COMM_REDUCE_SCATTER,
    COMM_ALL_TO_ALL,
    COMM_P2P,
    COMM_KV_TRANSFER,
    EVENT_RECORD,
    EVENT_WAIT,
    STREAM_YIELD,
    RUNTIME_LAUNCH_GAP,
    CONTENTION_RESOURCE,
})

ALL_EDGE_KINDS = frozenset({
    STREAM_ORDER,
    DATA_DEPENDENCY,
    EVENT_DEPENDENCY,
    COLLECTIVE_DEPENDENCY,
    SCHEDULER_DEPENDENCY,
    KV_READINESS,
    PHASE_BOUNDARY,
    RESOURCE_CONTENTION,
})


def check_op(op: str) -> None:
    if op not in ALL_OPS:
        raise ValueError(f"unknown EDM op: {op}")


def check_edge_kind(kind: str) -> None:
    if kind not in ALL_EDGE_KINDS:
        raise ValueError(f"unknown EDM edge kind: {kind}")


def namespace(op: str) -> str:
    return op.split(".", 1)[0]


def is_namespace(op: str, ns: str) -> bool:
    return op.startswith(ns + ".")


def is_comm(op: str) -> bool:
    return is_namespace(op, "comm")


def is_data_movement(op: str) -> bool:
    return is_comm(op) or op == KV_TRANSFER


def is_compute(op: str) -> bool:
    return is_namespace(op, "compute")


def is_metadata(op: str) -> bool:
    return is_namespace(op, "metadata")


def is_kv(op: str) -> bool:
    return is_namespace(op, "kv")


def is_stream_overhead(op: str) -> bool:
    return is_namespace(op, "event") or op == STREAM_YIELD
