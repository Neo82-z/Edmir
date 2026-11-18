"""Source-derived execution accounts, without fitting or latency prediction.

Counts describe valid logical elements, not physical HBM/NVLink transactions.
Each host implementation has its own partition and execution graph. Library
internals remain unknown; they never inherit CUTLASS tile/load estimates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .dependence import derive_dependencies
from .host_baseline import (PER_SOURCE_CUTLASS, PER_SOURCE_CUBLAS, MERGED_CUTLASS,
                            MERGED_CUBLASLT, WALL_TIMER, host_baseline)
from .execution_policy import LEGACY_POLICY, UNIFORM_PEER_POLICY, with_dependency_policy
from .model import A2AGemmProblem
from .schedule import build_fused_program, build_sequential_program
from .two_stage_selection import parse_schedule_family

SCHEMA = "edmir.execution-ledger.v2"
FAMILIES = ("host", "direct", "splitk2", "splitk4", "splitk8", "splitk16",
            "direct_groupedn2", "splitk2_groupedn2", "splitk4_groupedn2",
            "splitk8_groupedn2", "splitk16_groupedn2")
HOST_FAMILIES = {"host": PER_SOURCE_CUTLASS, "host_compiler_cublas": PER_SOURCE_CUBLAS,
                 "host_merged_cutlass": MERGED_CUTLASS, "host_merged_cublaslt": MERGED_CUBLASLT}
ALL_FAMILIES = FAMILIES + tuple(n for n in HOST_FAMILIES if n != "host")
CUTLASS_REVISION = "dcf215af68a2d08d305076c152a06f201728cd53"
SOURCE_PATHS = {
    "model": "EDM-IR/edmir/compiler/model.py",
    "policy": "EDM-IR/edmir/compiler/execution_policy.py",
    "publication": "EDM-IR/edmir/compiler/remote_publication.py",
    "target": "EDM-IR/edmir/compiler/target.py",
    "codegen": "EDM-IR/edmir/compiler/backends/cpp_backend.py",
    "program_types": "EDM-IR/runtime/include/edmir/runtime/program.hpp",
    "runtime_validation": "EDM-IR/runtime/src/validate.cpp",
    "uniform_adapter": "EDM-IR/runtime/src/uniform_workload.cpp",
    "merged_runtime": "EDM-IR/runtime/src/merged_host_backend.cpp",
    "host_registry": "EDM-IR/edmir/compiler/host_baseline.py",
    "runtime_header": "EDM-IR/runtime/include/edmir/runtime/cuda_backend.hpp",
    "prepared_header": "EDM-IR/runtime/include/edmir/runtime/prepared_execution.hpp",
    "merged_header": "EDM-IR/runtime/include/edmir/runtime/merged_host_backend.hpp",
    "uniform_header": "EDM-IR/runtime/include/edmir/runtime/uniform_workload.hpp",
    "family_parser": "EDM-IR/edmir/compiler/two_stage_selection.py",
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
    "model": ["Op.resources", "Program"], "policy": ["dependency_policy"],
    "publication": ["validate_remote_publication"], "target": ["Target"],
    "codegen": ["CppManifestBackend.emit"], "program_types": ["ProgramView", "DependencyPolicy"],
    "runtime_validation": ["validate_program"],
    "uniform_adapter": ["uniform_workload", "validate_uniform_peer_policy"],
    "merged_runtime": ["execute", "copy_from_host", "prepare_lt"],
    "host_registry": ["HOST_BASELINES"],
    "runtime_header": ["CudaBackendOptions", "CudaBackend"],
    "prepared_header": ["PreparedExecution"], "merged_header": ["MergedHostBackend"],
    "uniform_header": ["UniformWorkload"], "family_parser": ["parse_schedule_family"],
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
    "model": "6da0b9dfb88160c373c1e688eeb7c55b5efeddc4bf8325da71337a9f5a96b0e9",
    "policy": "18b2901a2d05907d7de831472160600e4258016d5fffc18f2299049559b8a9b3",
    "publication": "82fa8c42b951bf525a78269a538ab1fb0f0c618bf70fe5d49cdc51bc29d169cf",
    "target": "e611a335fc37c36f9fa4fcc5cd73abcf2856194b4569d0f84dbbc4d8cac205a4",
    "codegen": "b0e150bcc069d5fd39b27159f242cf720889f2f29c13040a88b810aad5d6eaa7",
    "program_types": "cc02aa0f96c8930861e46ece2555a31c1461f9f13cb4a910c2070020c60ecec6",
    "runtime_validation": "9dccf171bc695b57f548d16233f8ed26889a3fb1e21e4953ffb77f25c830d8c4",
    "uniform_adapter": "20b5e7d41ad10fd6b5d8dd364fc23505426f10e543333f4d8cd13d3c2625507d",
    "merged_runtime": "29c285572c07847b0ff6a83dca1003950fa34def6e60f25d8443fd7d0ecc48d4",
    "host_registry": "51bd1652d6b0bea770b2b09931e80b66466ba0bb127010a9da2029c9975ddf9d",
    "runtime_header": "9b9fa7f86706897fae126ac7b8dce6ea1992f6b154200d130e923201294b5fcb",
    "prepared_header": "f8849e43a0ddf7f2e96d2f56b7bb38a971e088c471af862e89aedf426e1278f0",
    "merged_header": "4cfac502042b97f1868a5a7ae5034cb6b3a06836b910ec1c7345498028c452b4",
    "uniform_header": "ea373addde9c7b225a997f5542389c4163f50ab10aa31557d838e060adbe9a32",
    "family_parser": "4d875a82ca57fbd07ce538a4fecf4db0d306c963945c010bcbe6cc05a2e8a8d2",
    "lowering": "c7dcbe3f7fd818c1f036afa2161449da8f18d56290db715bbd6331e55d8bfc97",
    "dependencies": "ac05505b8202ab55cc4a1b0a196d63ece8760ed7043226c5f3f403a0144677cd",
    "kernel_type": "3dfe30701dc6b0de406ea224aa49de5e571bba58e7a20fd6c875f7fd9f64f6e0",
    "fused_launcher": "0301fad59dfa9c8d2b7ddb8c9622cc75b6fccdaa9c6450a2d51fdbaf59a3f47a",
    "host_launcher": "5cd55bea876b7b7b6cc55fba13b94ac60f20011a59e6c5d3a9866f97417b7f98",
    "runtime": "93459b8be9717f0983f6633790353d2aeede02044fff514ea5eedcbcb26e0a43",
    "prepared_runtime": "491c6ceb9349fc919dced43cd36d5d963eb173616ad8b009cb9ad34f712ba35f",
    "timer": "134f0d39a7966bb8c36a5252a76c729e3db0beeec3ba0b7bc268617f6d7754fc"
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
                     baseline_id=None, provenance=None,
                     dependency_policy=LEGACY_POLICY, algorithm_id=None):
    p, m, k, n = validate_shape(shape)
    if family not in ALL_FAMILIES:
        raise ValueError("unsupported candidate implementation")
    host = family in HOST_FAMILIES
    baseline = HOST_FAMILIES.get(family, PER_SOURCE_CUTLASS)
    if baseline_id is not None and host_baseline(baseline_id) != baseline:
        raise ValueError("baseline identity does not match candidate family; use its explicit host family")
    baseline_id = baseline.baseline_id if host else None
    merged = baseline in (MERGED_CUTLASS, MERGED_CUBLASLT) and host
    library = baseline in (PER_SOURCE_CUBLAS, MERGED_CUBLASLT) and host
    if merged:
        if metadata_mode not in ("legacy", "merged"):
            raise ValueError("merged host has its own metadata implementation")
        metadata_mode = "merged"
    elif metadata_mode not in ("legacy", "prepared"):
        raise ValueError("metadata mode must be explicit: legacy or prepared")
    if not isinstance(environment_id, str) or not environment_id.strip():
        raise ValueError("environment identity required; it is not a hardware verification")
    if host and dependency_policy != LEGACY_POLICY:
        raise ValueError("uniform peer dependency policy is fused-only")
    if algorithm_id is not None and (not isinstance(algorithm_id, str) or not algorithm_id.strip()):
        raise ValueError("algorithm identity must be nonempty when supplied")
    q, g = (1, 1) if host else parse_schedule_family(family)
    if baseline == MERGED_CUBLASLT and host:
        algorithm = algorithm_id  # None remains unmeasured and ineligible for TimingEvidence.
    else:
        algorithm = ("library_default" if library else
                     "sm80_64x128x32" + (f"_splitk{q}" if q > 1 else ""))
        if algorithm_id is not None and algorithm_id != algorithm:
            raise ValueError("algorithm identity does not match the source implementation")
    # Restrict the exact byte/partition ledger rather than pretending padded or
    # empty K partitions execute the same epilogue and useful load counts.
    if not library and k % (32*q):
        raise ValueError("exact account requires aligned, nonempty 32*q K partitions")
    launcher = "host_launcher" if host else "fused_launcher"
    tm, tn, tk = 64, 128*g, 32
    gemm_m = p*m if merged else m
    mt, nt = (gemm_m+tm-1)//tm, (n+tn-1)//tn
    tiles = mt*nt
    implementation = (baseline_id if host else f"direct_peer_cutlass_sm80_q{q}_g{g}")
    if metadata_mode == "prepared":
        implementation += "_prepared_metadata_v2"
    if dependency_policy != LEGACY_POLICY:
        implementation += "_" + dependency_policy
    provenance = provenance if provenance is not None else source_provenance()
    for source, expected in REVIEWED_SOURCE_HASHES.items():
        if provenance.get('project', {}).get(source, {}).get('sha256') != expected:
            raise ValueError(f"execution source changed or absent: {source}; re-audit the account before updating its reviewed hash")
    problem = A2AGemmProblem(p, m, k, n)
    program = (build_sequential_program(problem) if host
               else build_fused_program(problem, chunk_rows=m))
    program = with_dependency_policy(program, dependency_policy)
    dependency = derive_dependencies(program)
    dependency.require_valid()
    counts = {}

    def count(name, value, unit, scope, expression, source):
        counts[name] = dict(value=value, unit=unit, scope=scope,
                            expression=expression, source=source)

    pair = "one merged destination GEMM" if merged else "one source-destination GEMM"
    whole = "all P ranks, one execute()"
    count("m_tiles", mt, "tiles", pair, "ceil(GEMM_M/64); GEMM_M=P*M for merged, otherwise M", "kernel_type")
    count("n_tiles", nt, "tiles", pair, "ceil(N/(128*g))", launcher)
    count("ctas", tiles*q, "CTAs", pair, "m_tiles*n_tiles*q", launcher)
    count("gemm_launches", p if merged else p*p, "launches", whole, "P if merged else P*P", "merged_runtime" if merged else "lowering")
    count("gemms_per_destination", 1 if merged else p, "launches", "one destination rank",
          "1 if merged else P", "merged_runtime" if merged else "lowering")
    count("k_iterations_per_partition", k//(32*q), "iterations", "one CTA",
          "K/(32*q), under aligned partition precondition", "kernel_type")
    count("logical_flops", 2*gemm_m*k*n, "FLOPs", pair, "2*GEMM_M*K*N", "kernel_type")
    count("unique_a_bytes", 2*gemm_m*k, "bytes", pair, "2*GEMM_M*K", "kernel_type")
    count("unique_b_bytes", 2*k*n, "bytes", pair, "2*K*N", "kernel_type")
    count("partition_a_bytes", 2*gemm_m*k//q, "bytes", "one K partition, all output tiles",
          "2*GEMM_M*K/q; working set, not total communication", "kernel_type")
    count("partition_b_bytes", 2*k*n//q, "bytes", "one K partition, all output tiles",
          "2*K*N/q", "kernel_type")
    count("logical_a_load_bytes", 2*gemm_m*k*nt, "bytes", pair,
          "2*GEMM_M*K*n_tiles; q disjoint K partitions cover K once", launcher)
    count("logical_b_load_bytes", 2*k*n*mt, "bytes", pair,
          "2*K*N*m_tiles", launcher)
    count("output_write_bytes", 2*gemm_m*n*q, "bytes", pair, "2*GEMM_M*N*q", "cutlass")
    count("output_read_bytes", 2*gemm_m*n*(q-1), "bytes", pair, "2*GEMM_M*N*(q-1)", "cutlass")
    count("extra_output_bytes", 4*gemm_m*n*(q-1), "bytes", pair,
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
    count("staging_allocation_bytes", 2*p*p*m*k if merged else sum(b.size_bytes for b in program.buffers if b.role == "staging"),
          "bytes", whole, "merged P destination matrices or actual Program staging allocations", "merged_runtime" if merged else "lowering")
    count("final_stream_synchronizations", (2 if merged else 3)*p, "API calls", whole,
          "2 streams per merged rank, otherwise 3; wall timer includes synchronize()", "merged_runtime" if merged else "runtime")

    if library:
        for name in ("m_tiles", "n_tiles", "ctas", "k_iterations_per_partition", "partition_a_bytes",
                     "partition_b_bytes", "logical_a_load_bytes", "logical_b_load_bytes",
                     "output_write_bytes", "output_read_bytes", "extra_output_bytes",
                     "epilogue_boundaries_per_tile", "semaphore_slots", "workspace_bytes"):
            counts.pop(name, None)
    if merged:
        count("workspace_allocation_bytes", p*32*1024*1024 if library else 0, "bytes", whole,
              "32 MiB per rank for cuBLASLt; zero for merged CUTLASS q1", "merged_runtime")
    graph = {"backend": "merged_host" if merged else "cuda",
             "dependency_policy": dependency_policy,
             "input_manifest": program.to_dict(),
             "manifest_dependencies": [asdict(e) for e in dependency.dependencies],
             "manifest_order": list(dependency.topological_order),
             "launch_mnk": [gemm_m, n, k], "q": None if library else q, "g": None if library else g,
             "metadata_mode": metadata_mode, "algorithm_id": algorithm}
    if merged:
        # This backend executes its own graph, not the input manifest's P^2 GEMMs.
        graph["execution"] = merged_execution_graph(p, m, k, n)
    else:
        graph["execution"] = {"ops": [asdict(op) for op in program.ops],
                              "dependencies": [asdict(e) for e in dependency.dependencies]}
    comparison_context = {"shape": [p, m, k, n], "environment_id": environment_id,
                          "workload": "uniform_shared_destination_weights_v1",
                          "dtype": "FP16_operands_and_outputs_FP32_accumulator",
                          "timing_scope": WALL_TIMER, "input_readiness": "completed_upload_v1",
                          "timer_source_sha256": provenance["project"]["timer"]["sha256"]}
    identity = dict(shape=[p, m, k, n], family=family, environment_id=environment_id,
                    baseline_id=baseline_id, implementation_id=implementation,
                    metadata_mode=metadata_mode, timing_scope=WALL_TIMER,
                    dependency_policy=dependency_policy, algorithm_id=algorithm,
                    comparison_context=comparison_context,
                    execution_graph_digest=digest(graph), source_digest=digest(provenance))
    return {
        "schema": SCHEMA, "identity": identity, "identity_sha256": digest(identity),
        "execution_graph": graph,
        "semantics": {"M": "rows per source-destination pair, not request batch size",
                      "GEMM_M": "P*M for merged host, otherwise M",
                      "dtype": "FP16 operands/output, FP32 accumulator",
                      "split_k_rounding": ("library internals unspecified" if library else
                                           "each serial epilogue writes FP16; not bitwise equivalent to q=1"),
                      "correctness": "requires full-output validation under the declared numerical contract"},
        "source_facts": {
            "gemm_mnk": [gemm_m, n, k], "tile_mnk": None if library else [tm, tn, tk],
            "warp_mnk": None if library else [32, 64, 32],
            "threads_per_cta": None if library else 128*g, "mainloop_stages": None if library else 3,
            "split_k_serial": None if library else q > 1,
            "k_intervals": None if library else [[i*k//q, (i+1)*k//q] for i in range(q)],
            "valid_output_fraction": None if library else gemm_m*n/(tiles*tm*tn),
            "operation_counts": ({"transfer": p*p, "gemm": p} if merged else
                                 dict(sorted(Counter(op.kind for op in program.ops).items()))),
            "program": program.to_dict(), "program_is_input_manifest": merged,
            "program_dependencies": [asdict(edge) for edge in dependency.dependencies],
            "runtime_contract": ([
                "one P*M GEMM per destination after all destination receives complete",
                "2 streams per rank; send/recv and local copies are inside execute()",
                "cuBLASLt heuristic and workspace allocation are outside timing",
            ] if merged else [
                "one cuBLAS GEMM per source on the destination compute stream",
                "cublasGemmEx requests CUBLAS_GEMM_DEFAULT_TENSOR_OP; actual kernel internals are unknown",
                "transfers, dependency events and completion remain inside execute/synchronize",
            ] if library else [
                "one compute stream per destination serializes its P GEMM launches",
                "library internals are not CUTLASS counts" if library else "CUTLASS source-specific account",
                ("all destination staging inputs precede its GEMMs via a barrier" if host else
                 "GEMM A pointer comes from ncclGetLsaPointer; no separate NCCL send/recv for this path"),
                "dependency event waits and completion records remain in execute()",
                "split-K workspace allocation/zeroing is in prepare(), outside steady timing",
                "q>1: epilogues wait in K-partition order; final partition resets semaphore",
                ("access/device metadata is prepared before timing" if metadata_mode == "prepared" else
                 "legacy execute() still looks up accesses and constructs metadata"),
            ]) + ["uploads complete before execute; immutable inputs and completed iterations required",
                   "dependency policy: " + dependency_policy],
        },
        "counting_identities": counts,
        "assumptions": [
            "library implementation accepts the declared row-major matrices" if library else
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


def execution_ledgers(shape, families=FAMILIES, *, algorithm_ids=None, **kwargs):
    families = tuple(families)
    if not families or len(families) > len(ALL_FAMILIES) or len(set(families)) != len(families):
        raise ValueError("nonempty unique bounded candidate set required")
    provenance = source_provenance()
    algorithm_ids = {} if algorithm_ids is None else algorithm_ids
    if set(algorithm_ids) - set(families):
        raise ValueError("algorithm identities must belong to requested candidates")
    result = {}
    for name in sorted(families):
        options = dict(kwargs)
        if name in algorithm_ids:
            options["algorithm_id"] = algorithm_ids[name]
        if name in ("host_merged_cutlass", "host_merged_cublaslt"):
            options["metadata_mode"] = "merged"
        if name in HOST_FAMILIES:
            options["dependency_policy"] = LEGACY_POLICY
        result[name] = execution_ledger(shape, name, provenance=provenance, **options)
    return result


def merged_execution_graph(p, m, k, n):
    """Backend graph, including shared communication stream order, not library internals."""
    nodes, edges, last = [], [], {}
    for step in range(p):
        for d in range(p):
            src = (d+step) % p
            name = f"transfer.s{src}.d{d}"
            nodes.append({"id": name, "kind": "local_copy" if src == d else "send_recv",
                          "source": src, "destination": d, "payload_bytes": 2*m*k})
            for rank in dict.fromkeys((src, d)):
                if rank in last:
                    edges.append({"src": last[rank], "dst": name, "kind": "comm_queue", "rank": rank})
                last[rank] = name
    for d in range(p):
        name = f"gemm.d{d}"
        nodes.append({"id": name, "kind": "gemm", "destination": d, "mnk": [p*m, n, k]})
        edges.append({"src": last[d], "dst": name, "kind": "ready_event", "rank": d})
    return {"ops": nodes, "dependencies": edges, "completion": "compute_and_communication_streams_all_ranks"}
