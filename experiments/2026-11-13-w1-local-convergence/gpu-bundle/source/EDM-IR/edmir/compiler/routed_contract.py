from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .model import DTYPE_BYTES, _check_schema


SCHEMA_ROUTED_TRACE = "edmir.routed-trace.v1"
SCHEMA_ROUTED_TRAFFIC = "edmir.routed-traffic-ledger.v1"


@dataclass(frozen=True, slots=True)
class RouteRecord:
    """One post-top-k routed row with an authoritative expert destination."""

    global_token_id: int
    source_rank: int
    destination_rank: int
    expert_id: int
    route_slot: int
    gate_weight: float = 1.0

    def input_key(self) -> tuple[int, int, int]:
        return (self.source_rank, self.global_token_id, self.route_slot)

    def output_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.destination_rank,
            self.expert_id,
            self.source_rank,
            self.global_token_id,
            self.route_slot,
        )


@dataclass(frozen=True, slots=True)
class RoutedTrace:
    """Frozen row-level routing contract for one benchmark problem.

    ``expert_placement[e]`` is the owning destination rank of expert ``e``.
    Routes are stored in canonical source-major input order.  Backends may use
    private layouts, but the benchmark compares outputs in ``output_key`` order.
    """

    trace_id: str
    world_size: int
    k: int
    n: int
    dtype: str
    phase: str
    expert_placement: tuple[int, ...]
    routes: tuple[RouteRecord, ...]

    def validate(self) -> None:
        if not self.trace_id:
            raise ValueError("routed trace_id must be non-empty")
        if any(type(v) is not int for v in (self.world_size, self.k, self.n, *self.expert_placement)):
            raise ValueError("dimensions and expert placement must be integers")
        if self.world_size < 2:
            raise ValueError("routed world_size must be at least two")
        if min(self.k, self.n) <= 0:
            raise ValueError("routed K and N must be positive")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported routed dtype: {self.dtype}")
        if self.phase not in {"prefill", "decode", "synthetic"}:
            raise ValueError(f"unsupported routed phase: {self.phase}")
        if not self.expert_placement:
            raise ValueError("routed trace must declare expert placement")
        if any(
            owner < 0 or owner >= self.world_size
            for owner in self.expert_placement
        ):
            raise ValueError("expert owner is outside the routed world")

        input_keys = tuple(route.input_key() for route in self.routes)
        if any(type(v) is not int for key in input_keys for v in key):
            raise ValueError("route identities and ranks must be integers")
        if input_keys != tuple(sorted(input_keys)):
            raise ValueError("routes must use canonical source-major input order")
        if len(set(input_keys)) != len(input_keys):
            raise ValueError("each (source, token, route slot) must be unique")

        token_experts: set[tuple[int, int]] = set()
        token_sources: dict[int, int] = {}
        token_slots: dict[int, list[int]] = {}
        for route in self.routes:
            if any(type(value) is not int for value in (
                route.global_token_id, route.source_rank, route.destination_rank,
                route.expert_id, route.route_slot,
            )):
                raise ValueError("route identities and ranks must be integers")
            if route.global_token_id < 0 or route.route_slot < 0:
                raise ValueError("token ids and route slots must be non-negative")
            if not math.isfinite(route.gate_weight):
                raise ValueError("gate weights must be finite")
            if route.source_rank < 0 or route.source_rank >= self.world_size:
                raise ValueError("route source is outside the routed world")
            if route.expert_id < 0 or route.expert_id >= len(self.expert_placement):
                raise ValueError("route expert is outside the placement table")
            if self.expert_placement[route.expert_id] != route.destination_rank:
                raise ValueError("route destination disagrees with expert placement")
            token_expert = (route.global_token_id, route.expert_id)
            if token_expert in token_experts:
                raise ValueError("one token cannot route twice to the same expert")
            token_experts.add(token_expert)
            source = token_sources.setdefault(route.global_token_id, route.source_rank)
            if source != route.source_rank:
                raise ValueError("a global token must have exactly one source rank")
            token_slots.setdefault(route.global_token_id, []).append(route.route_slot)
        for slots in token_slots.values():
            if sorted(slots) != list(range(len(slots))):
                raise ValueError("a token's route slots must be contiguous from zero")

    def canonical_output_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(
            sorted(range(len(self.routes)), key=lambda index: self.routes[index].output_key())
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": SCHEMA_ROUTED_TRACE,
            "trace_id": self.trace_id,
            "world_size": self.world_size,
            "k": self.k,
            "n": self.n,
            "dtype": self.dtype,
            "phase": self.phase,
            "expert_placement": list(self.expert_placement),
            "routes": [asdict(route) for route in self.routes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoutedTrace":
        _check_schema(value, SCHEMA_ROUTED_TRACE)
        trace = cls(
            trace_id=str(value["trace_id"]),
            world_size=value["world_size"],
            k=value["k"],
            n=value["n"],
            dtype=str(value["dtype"]),
            phase=str(value["phase"]),
            expert_placement=tuple(value["expert_placement"]),
            routes=tuple(RouteRecord(**route) for route in value["routes"]),
        )
        trace.validate()
        return trace


@dataclass(frozen=True, slots=True)
class RoutedTrafficLedger:
    trace_id: str
    replay: int
    unique_remote_tx_bytes: tuple[int, ...]
    unique_remote_rx_bytes: tuple[int, ...]
    replayed_remote_tx_bytes: tuple[int, ...]
    replayed_remote_rx_bytes: tuple[int, ...]
    max_rank_unique_remote_bytes: int
    max_rank_replayed_remote_bytes: int
    max_directional_cut_unique_bytes: int
    max_directional_cut_replayed_bytes: int
    nonempty_remote_buckets: int
    bucket_rows_p50: int
    bucket_rows_p95: int
    expert_rows: tuple[int, ...]
    cta_supply_by_rank: tuple[int, ...]
    routing_skew: float
    routing_entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_ROUTED_TRAFFIC, **asdict(self)}


def _nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _max_directional_cut(
    bucket_bytes: dict[tuple[int, int, int], int], world_size: int
) -> int:
    # Rank zero anchors one side so complementary partitions are visited once.
    # Both directions are evaluated explicitly below.
    masks = range(0, 1 << (world_size - 1))
    maximum = 0
    for tail in masks:
        left = {0}
        left.update(
            rank
            for rank in range(1, world_size)
            if tail & (1 << (rank - 1))
        )
        if len(left) == world_size:
            continue
        forward = sum(
            byte_count
            for (source, destination, _), byte_count in bucket_bytes.items()
            if source in left and destination not in left
        )
        reverse = sum(
            byte_count
            for (source, destination, _), byte_count in bucket_bytes.items()
            if source not in left and destination in left
        )
        maximum = max(maximum, forward, reverse)
    return maximum


def build_routed_traffic_ledger(
    trace: RoutedTrace,
    *,
    threadblock_m: int = 64,
    threadblock_n: int = 128,
    split_k_slices: int = 1,
    n_tiles_per_cta: int = 1,
) -> RoutedTrafficLedger:
    """Build D7's analytical ledger without claiming measured wire traffic."""

    trace.validate()
    if min(threadblock_m, threadblock_n, split_k_slices, n_tiles_per_cta) <= 0:
        raise ValueError("tile, split, and grouping parameters must be positive")

    row_bytes = trace.k * DTYPE_BYTES[trace.dtype]
    replay = math.ceil(math.ceil(trace.n / threadblock_n) / n_tiles_per_cta)
    buckets: dict[tuple[int, int, int], int] = {}
    expert_rows = [0] * len(trace.expert_placement)
    for route in trace.routes:
        expert_rows[route.expert_id] += 1
        if route.source_rank == route.destination_rank:
            continue
        key = (route.source_rank, route.destination_rank, route.expert_id)
        buckets[key] = buckets.get(key, 0) + 1

    unique_bucket_bytes = {key: rows * row_bytes for key, rows in buckets.items()}
    replayed_bucket_bytes = {
        key: byte_count * replay for key, byte_count in unique_bucket_bytes.items()
    }
    unique_tx = [0] * trace.world_size
    unique_rx = [0] * trace.world_size
    replayed_tx = [0] * trace.world_size
    replayed_rx = [0] * trace.world_size
    for (source, destination, _), byte_count in unique_bucket_bytes.items():
        unique_tx[source] += byte_count
        unique_rx[destination] += byte_count
        replayed_tx[source] += byte_count * replay
        replayed_rx[destination] += byte_count * replay

    cta_supply = [0] * trace.world_size
    n_groups = replay
    for expert_id, rows in enumerate(expert_rows):
        owner = trace.expert_placement[expert_id]
        cta_supply[owner] += (
            math.ceil(rows / threadblock_m) * n_groups * split_k_slices
            if rows
            else 0
        )

    total_rows = len(trace.routes)
    mean_rows = total_rows / len(expert_rows)
    routing_skew = max(expert_rows, default=0) / mean_rows if mean_rows else 0.0
    if total_rows and len(expert_rows) > 1:
        entropy = -sum(
            (rows / total_rows) * math.log(rows / total_rows)
            for rows in expert_rows
            if rows
        ) / math.log(len(expert_rows))
    else:
        entropy = 0.0

    bucket_rows = tuple(buckets.values())
    return RoutedTrafficLedger(
        trace_id=trace.trace_id,
        replay=replay,
        unique_remote_tx_bytes=tuple(unique_tx),
        unique_remote_rx_bytes=tuple(unique_rx),
        replayed_remote_tx_bytes=tuple(replayed_tx),
        replayed_remote_rx_bytes=tuple(replayed_rx),
        max_rank_unique_remote_bytes=max((*unique_tx, *unique_rx), default=0),
        max_rank_replayed_remote_bytes=max((*replayed_tx, *replayed_rx), default=0),
        max_directional_cut_unique_bytes=_max_directional_cut(
            unique_bucket_bytes, trace.world_size
        ),
        max_directional_cut_replayed_bytes=_max_directional_cut(
            replayed_bucket_bytes, trace.world_size
        ),
        nonempty_remote_buckets=len(buckets),
        bucket_rows_p50=_nearest_rank(bucket_rows, 0.50),
        bucket_rows_p95=_nearest_rank(bucket_rows, 0.95),
        expert_rows=tuple(expert_rows),
        cta_supply_by_rank=tuple(cta_supply),
        routing_skew=routing_skew,
        routing_entropy=entropy,
    )


def validate_fragmentation_sentinel(
    traces: Iterable[RoutedTrace],
    *,
    threadblock_m: int = 64,
    threadblock_n: int = 128,
    split_k_slices: int = 1,
    n_tiles_per_cta: int = 1,
) -> tuple[RoutedTrafficLedger, ...]:
    """Require D7 sentinel traces to differ only in bucket decomposition."""

    traces = tuple(traces)
    if len(traces) != 3:
        raise ValueError("fragmentation sentinel requires low, center, and high traces")
    ledgers = tuple(
        build_routed_traffic_ledger(
            trace,
            threadblock_m=threadblock_m,
            threadblock_n=threadblock_n,
            split_k_slices=split_k_slices,
            n_tiles_per_cta=n_tiles_per_cta,
        )
        for trace in traces
    )

    def row_margins(trace: RoutedTrace) -> tuple[tuple[int, ...], tuple[int, ...]]:
        source = [0] * trace.world_size
        destination = [0] * trace.world_size
        for route in trace.routes:
            source[route.source_rank] += 1
            destination[route.destination_rank] += 1
        return tuple(source), tuple(destination)

    first_trace = traces[0]
    first_ledger = ledgers[0]
    fixed = (
        first_trace.world_size,
        first_trace.k,
        first_trace.n,
        first_trace.dtype,
        first_trace.phase,
        first_trace.expert_placement,
        len(first_trace.routes),
        row_margins(first_trace),
        first_ledger.expert_rows,
        first_ledger.unique_remote_tx_bytes,
        first_ledger.unique_remote_rx_bytes,
        first_ledger.replayed_remote_tx_bytes,
        first_ledger.replayed_remote_rx_bytes,
        first_ledger.cta_supply_by_rank,
        first_ledger.max_directional_cut_unique_bytes,
        first_ledger.max_directional_cut_replayed_bytes,
    )
    for trace, ledger in zip(traces[1:], ledgers[1:]):
        candidate = (
            trace.world_size,
            trace.k,
            trace.n,
            trace.dtype,
            trace.phase,
            trace.expert_placement,
            len(trace.routes),
            row_margins(trace),
            ledger.expert_rows,
            ledger.unique_remote_tx_bytes,
            ledger.unique_remote_rx_bytes,
            ledger.replayed_remote_tx_bytes,
            ledger.replayed_remote_rx_bytes,
            ledger.cta_supply_by_rank,
            ledger.max_directional_cut_unique_bytes,
            ledger.max_directional_cut_replayed_bytes,
        )
        if candidate != fixed:
            raise ValueError(
                "fragmentation sentinel changed bytes, rank marginals, expert work, "
                "replay, or CTA supply"
            )
    bucket_counts = {ledger.nonempty_remote_buckets for ledger in ledgers}
    if len(bucket_counts) != len(ledgers):
        raise ValueError("fragmentation sentinel bucket counts must be distinct")
    return ledgers
