#!/usr/bin/env python3
"""Prepare the four-shape W1 registry and its fixed schedule without a device."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil

from prepare_selector_gpu import REPO, SEED, program_artifact
from prepare_p8_gpu import PRIMARY, IMMUTABLE, LEGACY, CUTLASS_REVISION, NCCL_REVISION
from edmir.compiler import A2AGemmProblem, CppManifestBackend, Target, TopologySnapshot
from edmir.compiler.schedule import build_pipelined_program
from edmir.compiler.measurement_dataset import sha256

SHAPES = {"grouped": (8, 64, 8192, 1024), "m512": (8, 512, 8192, 512),
          "large_n": (8, 64, 4096, 2048), "m256": (8, 256, 8192, 512)}
PAIR = ("pipeline", "pipeline_serial")
SEEDS = (SEED, 20261101, 20261102)


def baseline(shape, serial):
    p, m, k, n = shape
    topology = TopologySnapshot(p, ("planned-A100",)*p, ((True,)*p,)*p)
    target = Target("W1-staging", topology, arch="sm_80", gemm_provider="cutlass",
                    gemm_kernel="sm80_64x128x32", communication_api="host_nccl")
    program = build_pipelined_program(A2AGemmProblem(p, m, k, n), chunk_rows=m,
                                      buffer_count=2, serialize_on_rank=serial)
    return CppManifestBackend().emit(program, target)


def registry(directory, shape, families):
    includes, entries, written = [], [], set()
    for family in families:
        compiled = "host" if family.startswith("host") else family
        if compiled not in written:
            if compiled in PAIR:
                artifact = baseline(shape, compiled == "pipeline_serial")
            else:
                artifact = program_artifact(shape, compiled, dependency_policy=LEGACY if compiled == "host" else IMMUTABLE)
            artifact.write_to(directory/compiled)
            includes.extend([f"#define edmir_generated program_{compiled}",
                             f'#include "{compiled}/edmir_program.hpp"', "#undef edmir_generated"])
            written.add(compiled)
        entries.append(f'  Candidate{{"{family}", program_{compiled}::program}},')
    header = ["#pragma once", "#include <array>", *includes, "namespace edmir_batch {",
              "struct Candidate { std::string_view name; edmir::runtime::ProgramView (*program)(); };",
              "inline constexpr std::array candidates{", *entries, "};", "}"]
    (directory/"edmir_batch.hpp").write_text("\n".join(header)+"\n")


def predictions():
    # This model was fitted for the previous execution identity. Carry the
    # historical held-shape predictions as a named comparator, with omissions.
    source = REPO/"experiments/2026-10-28-precision-scoring/run-01"
    by_shape, provenance = {}, {}
    for f in range(4):
        path = source/f"p8-adamw-ce1-fold{f}.json"
        provenance[path.name] = sha256(path)
        for row in json.loads(path.read_text())["evaluation"]["decisions"]:
            by_shape[tuple(row["shape"])] = {k.split("@")[0]:math.exp(v) for k,v in row["predicted_log_us"].items()}
    rows = []
    for tag, shape in SHAPES.items():
        old = by_shape[shape]
        fams = list(PRIMARY)+(list(PAIR) if tag in ("m512","large_n") else [])
        for family in fams:
            rows.append({"shape":shape,"family":family,"historical_predicted_us":old.get(family),
                         "historical_rank":sorted(old,key=old.get).index(family)+1 if family in old else None,
                         "w1_calibrated_prediction_us":None,
                         "support":"previous_identity_development_only" if family in old else "unscored"})
    return {"source":"P8 AdamW CE1 leave-shape-out development comparator","sources":provenance,"rows":rows}


def prepare(out, inputs):
    out.mkdir(parents=True,exist_ok=False)
    rng = random.Random(20261101)
    schedule, cmake = [], []
    for tag, shape in SHAPES.items():
        has_pair = tag in ("m512","large_n")
        families = list(PRIMARY)+(list(PAIR) if has_pair else [])
        registry(out/tag,shape,families)
        for seed in SEEDS:
            schedule.append(dict(shape=tag,phase="check",seed=seed,repetition=0,order=families))
        order=list(PRIMARY)
        rng.shuffle(order)
        for rep in range(3):
            shift=rep*4
            schedule.append(dict(shape=tag,phase="measure",seed=SEED,repetition=rep,order=order[shift:]+order[:shift]))
        if has_pair:
            for rep in range(3):
                schedule.append(dict(shape=tag,phase="pair",seed=SEED,repetition=rep,
                                     order=list(PAIR if rep%2==0 else reversed(PAIR))))
        for mode,src,lib in [("layout","check_w1_registry.cpp","edmir_runtime"),("batch","benchmark_selector.cpp","edmir_cuda_runtime")]:
            target=f"w1_{tag}_{mode}"
            cmake += ["if(EDMIR_ENABLE_CUDA)" if mode=="batch" else "if(TRUE)",
                      f'add_executable({target} "${{CMAKE_CURRENT_SOURCE_DIR}}/tools/{src}")',
                      f'target_include_directories({target} PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/{tag}")',
                      f'target_link_libraries({target} PRIVATE {lib})',
                      f'target_compile_definitions({target} PRIVATE EDMIR_SELECTOR_BATCH=1)' if mode=="batch" else f'add_test(NAME {target} COMMAND {target})',
                      "endif()"]
    cmake.extend(["if(EDMIR_CUDA_API_TRACE_TESTS)",
                  'add_executable(w1_group_lifetime "${CMAKE_CURRENT_SOURCE_DIR}/tests/w1_group_lifetime_test.cpp")',
                  'target_include_directories(w1_group_lifetime PRIVATE "${EDMIR_EXPERIMENT_DIR}/large_n" ${EDMIR_CUDA_API_HEADER_DIRS})',
                  'target_link_libraries(w1_group_lifetime PRIVATE edmir_cuda_api_trace_runtime)',
                  'add_test(NAME w1_group_lifetime COMMAND w1_group_lifetime)', "endif()"])
    # Run paired controls after the primary matrix, before optional profiling.
    schedule = [j for j in schedule if j["phase"]!="pair"]+[j for j in schedule if j["phase"]=="pair"]
    (out/"generated_targets.cmake").write_text("\n".join(cmake)+"\n")
    (out/"plan.json").write_text(json.dumps({"schema":"edmir.w1.fp16.v1","shapes":SHAPES,"jobs":schedule,
        "seeds":SEEDS,"cutlass":CUTLASS_REVISION,"nccl":NCCL_REVISION,"primary_blocks":156,"paired_blocks":12,
        "hardware":"8x A100-SXM4-80GB full NV12 P2P","warmup":20,"iterations":100},indent=2)+"\n")
    (out/"predictions.json").write_text(json.dumps(predictions(),indent=2)+"\n")
    src=out/"source"/"EDM-IR"
    shutil.copytree(REPO/"EDM-IR/runtime",src/"runtime",ignore=shutil.ignore_patterns("build","__pycache__"))
    for sub in ("edmir","tools"):
        for path in (REPO/"EDM-IR"/sub).rglob("*.py"):
            dest=src/path.relative_to(REPO/"EDM-IR");dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    prior=REPO/"experiments/2026-10-25-runpod-hour-01/recovered/work/build/selector_build.json"
    receipt=json.loads(prior.read_text())
    lock={Path(name).name:value for name,value in receipt["dependencies"].items()}
    (out/"nccl-artifacts.json").write_text(json.dumps(lock,indent=2)+"\n")
    shutil.copyfile(inputs/"inputs.json",out/"inputs.json")
    files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out/"files.json").write_text(json.dumps(files,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.out.resolve(),args.inputs.resolve())
