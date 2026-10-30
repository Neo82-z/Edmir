"""Small routed expert reference and private-layout checks; never a GPU timer."""
from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass

from edmir.compiler.routed_contract import RouteRecord, RoutedTrace


def fp32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def bf16(value: float) -> float:
    """Round finite FP32 to BF16, ties to even, retaining a Python container."""
    if not math.isfinite(value):
        raise ValueError("reference inputs must be finite")
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("!f", struct.pack("!I", bits))[0]


@dataclass(frozen=True)
class RoutedFixture:
    trace: RoutedTrace
    # Indexed by global token, so top-k routes share the same activation.
    activations: dict[int, tuple[float, ...]]
    weights: tuple[tuple[tuple[float, ...], ...], ...]
    seed: int

    def validate(self) -> None:
        self.trace.validate()
        if set(self.activations) != {r.global_token_id for r in self.trace.routes}:
            raise ValueError("activation coverage differs from global token IDs")
        if len(self.weights) != len(self.trace.expert_placement):
            raise ValueError("every expert, including empty experts, needs a weight")
        values = []
        for row in self.activations.values():
            if len(row) != self.trace.k:
                raise ValueError("activation K mismatch")
            values.extend(row)
        for weight in self.weights:
            if len(weight) != self.trace.k or any(len(row) != self.trace.n for row in weight):
                raise ValueError("expert weight K/N mismatch")
            values.extend(v for row in weight for v in row)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("reference inputs must be finite")


@dataclass(frozen=True)
class RoutedOutput:
    # Canonical expert-major outputs, plus both directions of the route map.
    values: tuple[tuple[float, ...], ...]
    output_to_input: tuple[int, ...]
    input_to_output: tuple[int, ...]
    expert_spans: tuple[tuple[int, int], ...]
    gate_weights: tuple[float, ...]


def make_fixture(trace: RoutedTrace, seed: int) -> RoutedFixture:
    trace.validate()
    if trace.dtype != "bfloat16":
        raise ValueError("the offline routed reference covers BF16 inputs only")
    rng = random.Random(seed)
    activations = {token: tuple(bf16(rng.uniform(-1, 1)) for _ in range(trace.k))
                   for token in sorted({r.global_token_id for r in trace.routes})}
    weights = tuple(tuple(tuple(bf16(rng.uniform(-1, 1)) for _ in range(trace.n))
                          for _ in range(trace.k)) for _ in trace.expert_placement)
    fixture = RoutedFixture(trace, activations, weights, seed)
    fixture.validate()
    return fixture


def canonicalize(trace: RoutedTrace, route_indices, values) -> RoutedOutput:
    """Check a backend's private order before canonicalizing its results."""
    trace.validate()
    route_indices, values = tuple(route_indices), tuple(tuple(row) for row in values)
    count = len(trace.routes)
    if any(type(i) is not int for i in route_indices) or sorted(route_indices) != list(range(count)):
        raise ValueError("output must cover every input route exactly once")
    if len(values) != count or any(len(row) != trace.n for row in values):
        raise ValueError("output route count or N mismatch")
    if any(not math.isfinite(v) for row in values for v in row):
        raise ValueError("output contains NaN/Inf")
    private = dict(zip(route_indices, values))
    order = trace.canonical_output_indices()
    inverse = [0]*count
    for out, inp in enumerate(order):
        inverse[inp] = out
    spans = []
    # Placement need not have contiguous owner ranks in expert-ID order.
    for expert in range(len(trace.expert_placement)):
        key = (trace.expert_placement[expert], expert)
        before = sum((r.destination_rank, r.expert_id) < key for r in trace.routes)
        length = sum(r.expert_id == expert for r in trace.routes)
        spans.append((before, before+length))
    return RoutedOutput(tuple(private[i] for i in order), order, tuple(inverse), tuple(spans),
                        tuple(trace.routes[i].gate_weight for i in order))


def execute_reference(fixture: RoutedFixture, *, split_k: int = 1,
                      group_n: int = 1, tile_n: int = 4, private_order=None) -> RoutedOutput:
    """Software FP32 accumulation on BF16-rounded inputs; NOT CUDA validation.

    Gate weights are metadata at this boundary; combine is intentionally not
    applied. Splitting changes reduction order, grouping only visitation order.
    """
    fixture.validate()
    trace = fixture.trace
    if not 1 <= split_k <= trace.k or group_n < 1 or tile_n < 1:
        raise ValueError("illegal reference split or grouping")
    order = tuple(range(len(trace.routes))) if private_order is None else tuple(private_order)
    if sorted(order) != list(range(len(trace.routes))):
        raise ValueError("private order must be a route permutation")
    outputs = []
    for index in order:
        route = trace.routes[index]
        a, w = fixture.activations[route.global_token_id], fixture.weights[route.expert_id]
        row = [0.0]*trace.n
        for start in range(0, trace.n, tile_n*group_n):
            for n in range(start, min(start+tile_n*group_n, trace.n)):
                partials = []
                for q in range(split_k):
                    acc = 0.0
                    for k in range(trace.k*q//split_k, trace.k*(q+1)//split_k):
                        acc = fp32(acc + a[k]*w[k][n])
                    partials.append(acc)
                acc = 0.0
                for partial in partials:
                    acc = fp32(acc + partial)
                row[n] = acc
        outputs.append(tuple(row))
    return canonicalize(trace, order, outputs)


def compare_outputs(expected: RoutedOutput, observed: RoutedOutput) -> dict:
    for name in ("output_to_input", "input_to_output", "expert_spans", "gate_weights"):
        if getattr(expected, name) != getattr(observed, name):
            raise ValueError(f"canonical metadata mismatch: {name}")
    if len(expected.values) != len(observed.values) or any(
        len(a) != len(b) for a, b in zip(expected.values, observed.values)
    ):
        raise ValueError("output shape mismatch")
    a = [v for row in expected.values for v in row]
    b = [v for row in observed.values for v in row]
    if not all(math.isfinite(v) for v in a+b):
        raise ValueError("numerical check contains NaN/Inf")
    errors = [abs(x-y) for x,y in zip(a,b)]
    # Explicit norms. These definitions need to be pinned in the real campaign
    # manifest; the offline smoke test does not silently freeze that choice.
    l2 = math.sqrt(sum(e*e for e in errors))/max(math.sqrt(sum(x*x for x in a)),1e-30)
    linf = max(errors,default=0)/max(max(map(abs,a),default=0),1e-30)
    return {"normalized_l2": l2, "normalized_max": linf,
            "checked_elements":len(a), "passed":l2 <= 1e-2 and linf <= 2e-2,
            "evidence_kind":"cpu_semantic_check_not_cuda"}


def toy_trace(world_size: int, pattern: str, *, top_k: int = 2) -> RoutedTrace:
    """Explicit synthetic routing, deliberately small and with uneven tails."""
    experts = 2*world_size
    if pattern not in {"balanced", "hotset", "empty"} or not 1 <= top_k <= experts:
        raise ValueError("unsupported toy pattern or top-k")
    placement = tuple(e//2 for e in range(experts))
    routes = []
    tokens = 0 if pattern == "empty" else 3*world_size+1
    for token in range(tokens):
        source = token % world_size
        for slot in range(top_k):
            expert = (token*top_k+slot)%experts if pattern == "balanced" else slot
            routes.append(RouteRecord(token, source, placement[expert], expert, slot,
                                      (slot+1)/(top_k*(top_k+1)/2)))
    trace = RoutedTrace(f"toy-p{world_size}-{pattern}",world_size,7,9,"bfloat16","synthetic",
                        placement,tuple(sorted(routes,key=lambda r:r.input_key())))
    trace.validate()
    return trace
