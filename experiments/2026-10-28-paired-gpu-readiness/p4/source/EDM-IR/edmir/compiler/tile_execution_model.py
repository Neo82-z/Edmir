"""Geometry and conditional dependency spans, without fitting or GPU claims.

The reviewed execution ledger owns kernel and host identities. Tile packing
rounds are descriptors, not achieved concurrency or a microsecond predictor.
"""
from __future__ import annotations

import heapq
import math

from .evidence_selection import assess_candidates
from .execution_ledger import execution_ledger

SCHEMA = "edmir.tile-execution-structure.v1"


def _integer(name, value, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _checked_ledger(ledger):
    # Reuse identity, graph, source, and comparison-context validation.
    assess_candidates({ledger["identity"]["family"]: ledger})


def packing_reference(ctas, sm_count, resident_ctas_per_sm=1):
    """Arithmetic capacity convention, including a full (not empty) final round."""
    _integer("CTA count", ctas, 2**40)
    _integer("SM count", sm_count, 1024)
    _integer("CTA residency bound", resident_ctas_per_sm, 32)
    capacity = sm_count * resident_ctas_per_sm
    full, remainder = divmod(ctas, capacity)
    return {"capacity_ctas": capacity, "rounds": full + bool(remainder),
            "last_round_ctas": remainder or capacity,
            "last_round_fraction": (remainder or capacity) / capacity,
            "total_slot_fraction": ctas / ((full + bool(remainder)) * capacity),
            "meaning": "reference packing only; ignores scheduling, stalls and serial progress"}


def describe_execution(shape, family, *, environment_id, sm_count=108,
                       residency_bound=None, **ledger_options):
    """No measured labels are accepted. Unknown library tiling stays unknown.

    residency_bound, when supplied, is a caller-declared static limit with a
    build source, not measured occupancy. No resource constants are guessed.
    """
    _integer("SM count", sm_count, 1024)
    ledger = execution_ledger(shape, family, environment_id=environment_id, **ledger_options)
    return structure_from_ledger(ledger, sm_count=sm_count, residency_bound=residency_bound)


def structure_from_ledger(ledger, *, sm_count=108, residency_bound=None):
    _checked_ledger(ledger)
    _integer("SM count", sm_count, 1024)
    counts = ledger["counting_identities"]
    facts = ledger["source_facts"]
    geo = None
    if facts["tile_mnk"] is not None:
        m, n, k = facts["gemm_mnk"]
        tm, tn, tk = facts["tile_mnk"]
        mt, nt = counts["m_tiles"]["value"], counts["n_tiles"]["value"]
        q = ledger["execution_graph"]["q"]
        output_tiles, ctas = mt * nt, counts["ctas"]["value"]
        geo = {
            "scope": "one executed GEMM launch", "gemm_mnk": [m, n, k],
            "tile_mnk": [tm, tn, tk], "m_tiles": mt, "n_tiles": nt,
            "output_tiles": output_tiles, "ctas": ctas, "q": q,
            "partition_k_iterations": counts["k_iterations_per_partition"]["value"],
            "valid_output_elements": m*n, "tile_output_slots": output_tiles*tm*tn,
            "valid_output_fraction": facts["valid_output_fraction"],
            "last_m_tile_rows": (m-1) % tm + 1,
            "last_n_tile_columns": (n-1) % tn + 1,
            "logical_flops": counts["logical_flops"]["value"],
            "padded_tile_work_flops": 2*output_tiles*tm*tn*k,
            "padded_work_is_measured_instruction_count": False,
            "one_cta_per_sm_reference": packing_reference(ctas, sm_count),
            "resource_bound_reference": None,
            "epilogue_order": {"output_chains": output_tiles, "stages_per_chain": q,
                               "ordered_edges": output_tiles*(q-1),
                               "chain_length_is_elapsed_time": False},
        }
    if residency_bound is not None:
        if geo is None:
            raise ValueError("unknown library geometry cannot receive CUTLASS residency bounds")
        if (not isinstance(residency_bound, dict)
                or set(residency_bound) != {"ctas_per_sm", "source", "identity_sha256"}
                or residency_bound["identity_sha256"] != ledger["identity_sha256"]
                or not isinstance(residency_bound["source"], str)
                or not residency_bound["source"].strip()):
            raise ValueError("residency bound needs the matching execution identity and build source")
        geo["resource_bound_reference"] = {
            **packing_reference(geo["ctas"], sm_count, residency_bound["ctas_per_sm"]),
            "source": residency_bound["source"], "observed_residency": False}
    host = ledger["identity"]["baseline_id"] is not None
    required = ["local_compute_service", "cpu_submission_and_completion"]
    required.append("staging_transfer_service" if host else "peer_access_service")
    if geo and geo["q"] > 1:
        required.append("serial_epilogue_and_waits")
    return {
        "schema": SCHEMA, "identity": ledger["identity"],
        "identity_sha256": ledger["identity_sha256"], "sm_count": sm_count,
        "geometry": geo,
        "program": {"gemm_launches": counts["gemm_launches"]["value"],
                    "gemms_per_destination": counts["gemms_per_destination"]["value"],
                    "dependency_policy": ledger["identity"]["dependency_policy"],
                    "execution_graph_digest": ledger["identity"]["execution_graph_digest"]},
        "service_evidence": {name: None for name in required},
        "logical_counts": counts,
        "missing_evidence": required + ["shared_resource_contention_and_overlap"],
        "predicted_latency_us": None, "compiler_admission": False,
    }


def conditional_dependency_span(ledger, node_durations_us, *, identity_sha256, source):
    """Evaluate a DAG with supplied durations, not an end-to-end wall-time model.

    Missing weights never default to zero. Durations may be hypothetical or
    separately measured; the caller must state their source. This calculation
    does not identify resource interference, CPU gaps, or internal kernel
    stages, and is not a lower/upper bound on the actual program without extra
    assumptions. Traversal is bounded and independent of node/edge order.
    """
    _checked_ledger(ledger)
    if identity_sha256 != ledger["identity_sha256"]:
        raise ValueError("duration context differs from execution identity")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("explicit duration source or hypothesis required")
    graph = ledger["execution_graph"]["execution"]
    nodes, edges = graph["ops"], graph["dependencies"]
    if not 1 <= len(nodes) <= 4096 or len(edges) > 16384:
        raise ValueError("dependency traversal budget exceeded")
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes) or any(not isinstance(n, str) or not n for n in ids):
        raise ValueError("unique nonempty operation IDs required")
    if not isinstance(node_durations_us, dict) or set(node_durations_us) - ids:
        raise ValueError("durations must belong to graph operation IDs")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in node_durations_us.values()):
        raise ValueError("finite nonnegative conditional durations required")
    parents, children = {n: set() for n in ids}, {n: set() for n in ids}
    for edge in edges:
        a, b = edge["src"], edge["dst"]
        if a not in ids or b not in ids or a == b:
            raise ValueError("invalid dependency endpoint")
        parents[b].add(a); children[a].add(b)
    degree = {n: len(parents[n]) for n in ids}
    ready = [n for n in ids if degree[n] == 0]; heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready); order.append(n)
        for child in sorted(children[n]):
            degree[child] -= 1
            if degree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(ids):
        raise ValueError("cyclic execution dependencies")
    missing = sorted(ids - set(node_durations_us))
    finish = {}
    if not missing:
        for n in order:
            finish[n] = max((finish[p] for p in parents[n]), default=0.) + node_durations_us[n]
            if not math.isfinite(finish[n]):
                raise ValueError("conditional duration overflow")
    return {"identity_sha256": identity_sha256, "duration_source": source,
            "status": "missing_node_durations" if missing else "conditional_dag_evaluation",
            "missing_nodes": missing, "dependency_span_us": max(finish.values()) if finish else None,
            "program_wall_time_us": None, "compiler_admission": False,
            "assumptions": ["supplied node durations are valid in this joint execution",
                            "dependencies represent all serialization relevant to this calculation"],
            "unmodeled": ["resource contention", "CPU submission gaps", "completion overhead",
                          "internal kernel stage overlap"]}
