"""Source-derived execution accounts, without fitting or latency prediction.

Counts describe valid logical elements, not physical HBM/NVLink transactions.
The supported implementation is the per-source SM80 CUTLASS runtime. Library
defaults and merged GEMMs must not silently inherit this implementation's account.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .dependence import derive_dependencies
from .host_baseline import PER_SOURCE_CUTLASS, WALL_TIMER, host_baseline
from .model import A2AGemmProblem
from .schedule import build_fused_program, build_sequential_program
from .two_stage_selection import parse_schedule_family

SCHEMA = "edmir.execution-ledger.v1"
FAMILIES = ("host", "direct", "splitk2", "splitk4", "splitk8", "splitk16",
            "direct_groupedn2", "splitk2_groupedn2", "splitk4_groupedn2",
            "splitk8_groupedn2", "splitk16_groupedn2")
CUTLASS_REVISION = "dcf215af68a2d08d305076c152a06f201728cd53"
SOURCE_PATHS = {
    "lowering": "EDM-IR/edmir/compiler/schedule.py",
    "dependencies": "EDM-IR/edmir/compiler/dependence.py",
    "kernel_type": "EDM-IR/runtime/src/cutlass_sm80_gemm.cuh",
    "fused_launcher": "EDM-IR/runtime/src/nccl_device_fused_cutlass.cu",
    "host_launcher": "EDM-IR/runtime/src/cutlass_gemm.cu",
    "runtime": "EDM-IR/runtime/src/cuda_backend.cpp",
    "prepared_runtime": "EDM-IR/runtime/src/prepared_execution.cpp",
    "timer": "EDM-IR/runtime/tools/benchmark_selector.cpp",
    "ledger": "EDM-IR/edmir/compiler/execution_ledger.py",
}
SOURCE_SYMBOLS = {
    "lowering": ["build_sequential_program", "build_fused_program"],
    "dependencies": ["derive_dependencies"],
    "kernel_type": ["Sm80Gemm", "Sm80Epilogue"],
    "fused_launcher": ["launch_lsa_cutlass_gemm_f16_sm80", "lsa_cutlass_workspace_size", "lsa_cutlass_gemm"],
    "host_launcher": ["launch_cutlass_gemm_f16_sm80"],
    "runtime": ["execute", "synchronize", "prepare_fused_cutlass_workspaces", "record_completion"],
    "prepared_runtime": ["prepare_execution"],
    "timer": ["measure"], "ledger": ["execution_ledger"],
}
# The formulas below were reviewed against this workspace snapshot. A new
# hash is provenance, not proof that an old formula still describes new code.
# Require explicit re-audit before accepting changes to the execution contract.
REVIEWED_SOURCE_HASHES = {
    "lowering": "c7dcbe3f7fd818c1f036afa2161449da8f18d56290db715bbd6331e55d8bfc97",
    "dependencies": "5f78141f7376b6b312592380743067343623b0248b27f0fee306951a12654450",
    "kernel_type": "3dfe30701dc6b0de406ea224aa49de5e571bba58e7a20fd6c875f7fd9f64f6e0",
    "fused_launcher": "0301fad59dfa9c8d2b7ddb8c9622cc75b6fccdaa9c6450a2d51fdbaf59a3f47a",
    "host_launcher": "5cd55bea876b7b7b6cc55fba13b94ac60f20011a59e6c5d3a9866f97417b7f98",
    "runtime": "7f3cb4935d5170096acbe9d23c02a917feb1029d40f4741951ffc39630fa14cb",
    "prepared_runtime": "491c6ceb9349fc919dced43cd36d5d963eb173616ad8b009cb9ad34f712ba35f",
    "timer": "1a9ff0f3cea4cfdf35d88b129793d6d24da8d90bafb637e3dccd26443cf8b26b",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_provenance(root: Path | None = None):
    root = root or Path(__file__).resolve().parents[3]
    sources = {}
    for name, relative in SOURCE_PATHS.items():
        path = root / relative
        sources[name] = {"path": relative, "symbols": SOURCE_SYMBOLS[name], "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)}
    return {"project": sources,
            "cutlass": {"revision": CUTLASS_REVISION,
                        "path": "include/cutlass/gemm/kernel/gemm.h",
                        "url": f"https://github.com/NVIDIA/cutlass/blob/{CUTLASS_REVISION}/include/cutlass/gemm/kernel/gemm.h",
                        "status": "pinned_source_semantics_not_a_verified_runtime_build"}}


def validate_shape(shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 4
            or any(type(x) is not int or x < 1 for x in shape)):
        raise ValueError("shape must be positive integers (P,M,K,N)")
    p, m, k, n = shape
    if not 2 <= p <= 8 or max(m, k, n) > 1_048_576:
        raise ValueError("bounded local account supports 2<=P<=8 and dimensions<=1048576")
    if k % 8 or n % 8:
        raise ValueError("the FP16 row-major CUTLASS account requires K,N divisible by 8")
    return p, m, k, n


def execution_ledger(shape, family, *, environment_id, metadata_mode="legacy",
                     baseline_id=PER_SOURCE_CUTLASS.baseline_id, provenance=None):
    p, m, k, n = validate_shape(shape)
    if family not in FAMILIES:
        raise ValueError("unsupported candidate implementation")
    if metadata_mode not in ("legacy", "prepared"):
        raise ValueError("metadata mode must be explicit: legacy or prepared")
    if not isinstance(environment_id, str) or not environment_id.strip():
        raise ValueError("environment identity required; it is not a hardware verification")
    baseline = host_baseline(baseline_id)
    if baseline != PER_SOURCE_CUTLASS:
        raise ValueError("this ledger does not describe cuBLAS or merged host implementations")
    q, g = parse_schedule_family(family)
    # Restrict the exact byte/partition ledger rather than pretending padded or
    # empty K partitions execute the same epilogue and useful load counts.
    if k % (32*q):
        raise ValueError("exact account requires aligned, nonempty 32*q K partitions")
    host = family == "host"
    launcher = "host_launcher" if host else "fused_launcher"
    tm, tn, tk = 64, 128*g, 32
    mt, nt = (m+tm-1)//tm, (n+tn-1)//tn
    tiles = mt*nt
    implementation = (baseline_id if host else f"direct_peer_cutlass_sm80_q{q}_g{g}")
    if metadata_mode == "prepared":
        implementation += "_prepared_metadata_v2"
    provenance = provenance if provenance is not None else source_provenance()
    for source, expected in REVIEWED_SOURCE_HASHES.items():
        if provenance['project'][source]['sha256'] != expected:
            raise ValueError(f"execution source changed or absent: {source}; re-audit the account before updating its reviewed hash")
    problem = A2AGemmProblem(p, m, k, n)
    program = (build_sequential_program(problem) if host
               else build_fused_program(problem, chunk_rows=m))
    dependency = derive_dependencies(program)
    dependency.require_valid()
    counts = {}

    def count(name, value, unit, scope, expression, source):
        counts[name] = dict(value=value, unit=unit, scope=scope,
                            expression=expression, source=source)

    pair = "one source-destination GEMM"
    whole = "all P ranks, one execute()"
    count("m_tiles", mt, "tiles", pair, "ceil(M/64)", "kernel_type")
    count("n_tiles", nt, "tiles", pair, "ceil(N/(128*g))", launcher)
    count("ctas", tiles*q, "CTAs", pair, "m_tiles*n_tiles*q", launcher)
    count("gemm_launches", p*p, "launches", whole, "P*P", "lowering")
    count("gemms_per_destination", p, "launches", "one destination rank", "P", "lowering")
    count("k_iterations_per_partition", k//(32*q), "iterations", "one CTA",
          "K/(32*q), under aligned partition precondition", "kernel_type")
    count("logical_flops", 2*m*k*n, "FLOPs", pair, "2*M*K*N", "kernel_type")
    count("unique_a_bytes", 2*m*k, "bytes", pair, "2*M*K", "kernel_type")
    count("unique_b_bytes", 2*k*n, "bytes", pair, "2*K*N", "kernel_type")
    count("partition_a_bytes", 2*m*k//q, "bytes", "one K partition, all output tiles",
          "2*M*K/q; working set, not total communication", "kernel_type")
    count("partition_b_bytes", 2*k*n//q, "bytes", "one K partition, all output tiles",
          "2*K*N/q", "kernel_type")
    count("logical_a_load_bytes", 2*m*k*nt, "bytes", pair,
          "2*M*K*n_tiles; q disjoint K partitions cover K once", launcher)
    count("logical_b_load_bytes", 2*k*n*mt, "bytes", pair,
          "2*K*N*m_tiles", launcher)
    count("output_write_bytes", 2*m*n*q, "bytes", pair, "2*M*N*q", "cutlass")
    count("output_read_bytes", 2*m*n*(q-1), "bytes", pair, "2*M*N*(q-1)", "cutlass")
    count("extra_output_bytes", 4*m*n*(q-1), "bytes", pair,
          "read+write minus the q=1 final write", "cutlass")
    count("epilogue_boundaries_per_tile", q-1, "dependencies", "one output tile",
          "q-1 ordered partition boundaries; not elapsed time", "cutlass")
    count("semaphore_slots", tiles if q > 1 else 0, "int32 slots", pair,
          "output_tiles if q>1 else 0", "fused_launcher")
    count("workspace_bytes", 4*tiles*p*p if q > 1 else 0, "bytes", whole,
          "4*output_tiles*P*P if q>1 else 0", "runtime")
    count("remote_unique_input_bytes", 2*p*(p-1)*m*k, "bytes", whole,
          "2*P*(P-1)*M*K", "lowering")
    count("remote_logical_a_load_bytes", 0 if host else 2*p*(p-1)*m*k*nt,
          "bytes", whole, "0 for staged host; 2*P*(P-1)*M*K*n_tiles for peer loads",
          "fused_launcher")
    count("explicit_transfer_payload_bytes", 2*p*(p-1)*m*k if host else 0,
          "bytes", whole, "host NCCL send payload; excludes protocol overhead", "runtime")
    count("local_copy_payload_bytes", 2*p*m*k if host else 0, "bytes", whole,
          "2*P*M*K for host diagonal copies, otherwise 0", "runtime")
    count("staging_allocation_bytes", sum(b.size_bytes for b in program.buffers if b.role == "staging"),
          "bytes", whole, "sum actual Program staging allocations", "lowering")
    count("final_stream_synchronizations", 3*p, "API calls", whole,
          "comm+compute+control per rank; wall timer includes synchronize()", "runtime")

    identity = dict(shape=[p, m, k, n], family=family, environment_id=environment_id,
                    baseline_id=baseline_id, implementation_id=implementation,
                    metadata_mode=metadata_mode, timing_scope=WALL_TIMER,
                    source_digest=digest(provenance))
    return {
        "schema": SCHEMA, "identity": identity, "identity_sha256": digest(identity),
        "semantics": {"M": "rows per source-destination pair, not request batch size",
                      "dtype": "FP16 operands/output, FP32 accumulator",
                      "split_k_rounding": "each serial epilogue writes FP16; not bitwise equivalent to q=1",
                      "correctness": "requires full-output validation under the declared numerical contract"},
        "source_facts": {
            "tile_mnk": [tm, tn, tk], "warp_mnk": [32, 64, 32],
            "threads_per_cta": 128*g, "mainloop_stages": 3, "split_k_serial": q > 1,
            "k_intervals": [[i*k//q, (i+1)*k//q] for i in range(q)],
            "valid_output_fraction": m*n/(tiles*tm*tn),
            "operation_counts": dict(sorted(Counter(op.kind for op in program.ops).items())),
            "program": program.to_dict(),
            "program_dependencies": [asdict(edge) for edge in dependency.dependencies],
            "runtime_contract": [
                "one compute stream per destination serializes its P GEMM launches",
                ("all destination staging inputs precede its GEMMs via a barrier" if host else
                 "GEMM A pointer comes from ncclGetLsaPointer; no separate NCCL send/recv for this path"),
                "dependency event waits and completion records remain in execute()",
                "split-K workspace allocation/zeroing is in prepare(), outside steady timing",
                "q>1: epilogues wait in K-partition order; final partition resets semaphore",
                ("access/device metadata is prepared before timing" if metadata_mode == "prepared" else
                 "legacy execute() still looks up accesses and constructs metadata"),
            ],
        },
        "counting_identities": counts,
        "assumptions": [
            "SM80 implementation and aligned row-major pointers pass CUTLASS can_implement",
            "runtime dependencies and CUTLASS revision match this source contract",
            "fused path has working NCCL Device API and peer-access topology",
            "inputs remain immutable during the measured execute/synchronize interval",
        ],
        "unknowns": [
            {"id": "physical_traffic", "reason": "cache reuse, sectors, protocol and fabric routing are not logical bytes"},
            {"id": "residency", "reason": "compiled registers/shared storage and actual resident CTAs are unmeasured"},
            {"id": "critical_path", "reason": "resource contention, actual overlap and semaphore stall durations are unmeasured"},
            {"id": "host_service_time", "reason": "CPU submission, metadata and synchronization cost are inside the wall timer"},
            {"id": "numerical_validity", "reason": "source legality does not establish device numerical correctness"},
        ],
        "latency_us": None,
        "cost_composition": "end-to-end makespan with dependencies and shared-resource contention; do not sum overlapping pipe counters or use max(ideal compute, ideal bandwidth) as a prediction",
        "provenance": provenance,
        "eligible_for_compiler_admission": False,
    }


def execution_ledgers(shape, families=FAMILIES, **kwargs):
    families = tuple(families)
    if not families or len(families) > len(FAMILIES) or len(set(families)) != len(families):
        raise ValueError("nonempty unique bounded candidate set required")
    provenance = source_provenance()
    return {name: execution_ledger(shape, name, provenance=provenance, **kwargs)
            for name in sorted(families)}
