#!/usr/bin/env python3
"""Prepare static programs and reproducible CPU references without a GPU."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler import (A2AGemmProblem, CppManifestBackend, Target, TopologySnapshot,
                            FusedScheduleContract, build_fused_program, build_sequential_program)
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.two_stage_selection import parse_schedule_family
from independent_selector_validation import (ALL_FAMILIES, BASELINES, WALL_TIMER, digest, verify_plan)

REPO = Path(__file__).resolve().parents[2]
PILOT_MKN = ((64, 1024, 512), (64, 4096, 2048), (64, 8192, 512), (256, 8192, 512))
SEED = 20261020


def program_artifact(shape, family):
    p, m, k, n = shape
    problem = A2AGemmProblem(p, m, k, n)
    topology = TopologySnapshot(p, ("planned-A100",)*p, tuple((True,)*p for _ in range(p)))
    host = family in BASELINES
    provider = "cublas" if family == "host_compiler_cublas" else "cutlass"
    q, g = (1, 1) if host else parse_schedule_family(family)
    kernel = ("library_default" if provider == "cublas" else
              "sm80_64x128x32"+(f"_splitk{q}" if q > 1 else ""))
    target = Target("selector-experiment", topology, arch="sm_80", gemm_provider=provider,
                    gemm_kernel=kernel, communication_api="host_nccl" if host else "nccl_device",
                    fused_schedule=FusedScheduleContract(g))
    program = build_sequential_program(problem) if host else build_fused_program(problem, chunk_rows=m)
    return CppManifestBackend().emit(program, target)


def matrix(rows, columns, stream):
    """Integer avalanche mixing; exact FP16 dyadic values, no RNG-library state."""
    import numpy as np
    x = np.arange(rows*columns, dtype=np.uint32)
    x ^= np.uint32((SEED+stream*0x9e3779b9) & 0xffffffff)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7feb352d)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846ca68b)
    x ^= x >> np.uint32(16)
    values = ((x & 31).astype(np.int32)-16).astype(np.float32)/32
    return values.astype("<f2").reshape(rows, columns)


def write_inputs(shape, root):
    """Canonical A[source,destination], then W[destination]; C uses the same order."""
    import numpy as np
    p, m, k, n = shape
    root.mkdir(parents=True, exist_ok=False)
    weights = [matrix(k, n, p*p+d) for d in range(p)]
    input_path, reference_path = root/"inputs.bin", root/"reference.bin"
    with input_path.open("xb") as inputs, reference_path.open("xb") as outputs:
        for source in range(p):
            for destination in range(p):
                a = matrix(m, k, source*p+destination)
                inputs.write(a.tobytes(order="C"))
                c = a.astype(np.float32) @ weights[destination].astype(np.float32)
                if c.dtype != np.float32 or not np.isfinite(c).all():
                    raise ValueError("CPU FP32 reference failed")
                outputs.write(c.astype("<f4").tobytes(order="C"))
        for w in weights:
            inputs.write(w.tobytes(order="C"))
    return {"shape": list(shape), "input_seed": SEED, "input_sha256": sha256(input_path),
            "reference_sha256": sha256(reference_path),
            "input_bytes": input_path.stat().st_size, "reference_bytes": reference_path.stat().st_size,
            "generator": "uint32 avalanche v1; (low5bits-16)/32 exactly represented in FP16",
            "reference": "CPU_FP32_on_identical_FP16_inputs", "numpy_version": np.__version__}


def source_manifest():
    paths = [*sorted((REPO/"EDM-IR/edmir").rglob("*.py")),
             *sorted((REPO/"EDM-IR/runtime").rglob("*.cpp")),
             *sorted((REPO/"EDM-IR/runtime").rglob("*.cu")),
             *sorted((REPO/"EDM-IR/runtime").rglob("*.cuh")),
             *sorted((REPO/"EDM-IR/runtime/include").rglob("*.hpp")),
             *sorted((REPO/"EDM-IR/tools").glob("*.py")),
             REPO/"EDM-IR/runtime/CMakeLists.txt"]
    # Ignore build trees if the repository has local CMake output.
    paths = [p for p in paths if "build" not in p.relative_to(REPO).parts]
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def prepare(out, *, plan_path=None, with_inputs=False):
    if out.exists():
        raise ValueError("choose a new package directory")
    if plan_path:
        envelope = json.loads(plan_path.read_text())
        plan = verify_plan(envelope)
        mode = "independent"
        works = plan["workloads"]
        repetitions = plan["measurement"]["repetitions"]
    else:
        mode, repetitions = "pilot_development", 3
        rng = random.Random(SEED)
        works = []
        for m, k, n in PILOT_MKN:
            orders = []
            for _ in range(repetitions):
                order = list(ALL_FAMILIES)
                rng.shuffle(order)
                orders.append(order)
            works.append({"shape": [4, m, k, n], "trial_orders": orders})
        envelope = None
    out.mkdir(parents=True)
    if envelope:
        (out/"independent_plan.json").write_text(json.dumps(envelope, indent=2)+"\n")
    commands, programs, jobs, workloads = [], {}, [], []
    for wi, work in enumerate(works):
        shape = work["shape"]
        wi_name = f"w{wi:03d}"
        workloads.append({"id": wi_name, **work})
        for family in ALL_FAMILIES:
            compiled_family = "host" if family in ("host_merged_cutlass", "host_merged_cublaslt") else family
            target = f"selector_{wi_name}_{compiled_family}"
            if target in programs:
                continue
            directory = out/"generated"/target
            program_artifact(shape, compiled_family).write_to(directory)
            programs[target] = {"shape": shape, "compiled_family": compiled_family,
                                "header": str((directory/"edmir_program.hpp").relative_to(out))}
            commands += [
                f'add_executable({target}_layout "${{CMAKE_CURRENT_SOURCE_DIR}}/tools/check_selector_layout.cpp")',
                f'target_include_directories({target}_layout PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/generated/{target}")',
                f'target_link_libraries({target}_layout PRIVATE edmir_runtime)',
                f'add_test(NAME {target}_layout COMMAND {target}_layout)',
                'if(EDMIR_ENABLE_CUDA)',
                f'  add_executable({target} "${{CMAKE_CURRENT_SOURCE_DIR}}/tools/benchmark_selector.cpp")',
                f'  target_include_directories({target} PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/generated/{target}")',
                f'  target_link_libraries({target} PRIVATE edmir_cuda_runtime)',
                'endif()',
            ]
        for repetition, order in enumerate(work["trial_orders"]):
            for position, family in enumerate(order):
                compiled = "host" if family in ("host_merged_cutlass", "host_merged_cublaslt") else family
                jobs.append({"workload_id": wi_name, "shape": shape, "family": family,
                             "repetition": repetition, "trial_order": position,
                             "executable": f"selector_{wi_name}_{compiled}"})
    (out/"generated_targets.cmake").write_text("\n".join(commands)+"\n")
    sources = source_manifest()
    payload = {"schema": "edmir.selector-gpu-package.v1", "mode": mode,
               "status": "prepared_locally_not_gpu_validated",
               "timing_scope": WALL_TIMER, "baseline_contracts": {n: b.to_dict() for n, b in BASELINES.items()},
               "warmup": 20, "iterations": 100, "repetitions": repetitions,
               "workloads": workloads, "programs": programs, "jobs": jobs,
               "sources": sources, "collector_sha256": digest(sources),
               "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
               "plan_sha256": envelope["plan_sha256"] if envelope else None}
    (out/"package.json").write_text(json.dumps({"package_sha256": digest(payload), "package": payload}, indent=2)+"\n")
    if with_inputs:
        prepare_inputs(out)
    return payload


def verify_package(root):
    envelope = json.loads((root/"package.json").read_text())
    p = envelope["package"]
    if digest(p) != envelope["package_sha256"]:
        raise ValueError("package manifest hash mismatch")
    for path, expected in p["sources"].items():
        if sha256(REPO/path) != expected:
            raise ValueError(f"source changed after preparing package: {path}")
    for path, expected in p["files"].items():
        if sha256(root/path) != expected:
            raise ValueError(f"generated package file changed: {path}")
    if p["mode"] == "independent":
        verify_plan(json.loads((root/"independent_plan.json").read_text()))
    return p


def prepare_inputs(root):
    p = verify_package(root)
    output = root/"inputs.json"
    if output.exists():
        raise ValueError("input manifest already exists")
    manifest = {}
    for work in p["workloads"]:
        manifest[work["id"]] = write_inputs(work["shape"], root/"data"/work["id"])
    output.write_text(json.dumps({"package_sha256": digest(p), "workloads": manifest}, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--plan", type=Path)
    make.add_argument("--with-inputs", action="store_true")
    inputs = sub.add_parser("inputs")
    inputs.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out, plan_path=args.plan, with_inputs=args.with_inputs)
        print(json.dumps({"programs": len(result["programs"]), "process_runs": len(result["jobs"]),
                          "mode": result["mode"]}))
    else:
        result = prepare_inputs(args.package)
        print(f"Prepared {len(result)} input/reference sets")
