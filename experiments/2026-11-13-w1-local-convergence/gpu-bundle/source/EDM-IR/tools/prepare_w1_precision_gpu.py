#!/usr/bin/env python3
"""Seal the existing four W1 shapes plus FP32 reduction candidates, locally.

No downloads, GPU access, new shapes or timing labels. Uses the audited W1
input manifest and pinned dependencies; the copied runner has a hard deadline.
"""
import argparse
import json
from pathlib import Path
import random
import shutil

from prepare_w1_gpu import registry, SHAPES, SEEDS, PRIMARY, REPO, CUTLASS_REVISION, NCCL_REVISION
from edmir.compiler.measurement_dataset import sha256


def prepare(out, archived):
    out.mkdir(parents=True,exist_ok=False)
    old=json.loads((archived/"plan.json").read_text())
    if old["shapes"]!={k:list(v) for k,v in SHAPES.items()} or old["seeds"]!=list(SEEDS):
        raise ValueError("only the original four development shapes and three seeds are allowed")
    new=[f"splitk{q}"+("_groupedn2" if g==2 else "")+"_fp32reduce" for q in (8,16) for g in (1,2)]
    families=list(PRIMARY)+new
    jobs=[];cmake=[];rng=random.Random(20261113)
    # Same collection/timing policy; no pair-phase labels enter primary scores.
    for tag in ("m512","large_n","grouped","m256"):
        registry(out/tag,SHAPES[tag],families)
        for seed in SEEDS:jobs.append(dict(shape=tag,phase="check",seed=seed,repetition=0,order=families))
        order=list(families);rng.shuffle(order)
        for rep in range(3):
            shift=rep*5
            jobs.append(dict(shape=tag,phase="measure",seed=SEEDS[0],repetition=rep,order=order[shift:]+order[:shift]))
        for mode,src,lib in [("layout","check_w1_registry.cpp","edmir_runtime"),("batch","benchmark_selector.cpp","edmir_cuda_runtime")]:
            target=f"w1_{tag}_{mode}"
            cmake += ["if(EDMIR_ENABLE_CUDA)" if mode=="batch" else "if(TRUE)",
                f'add_executable({target} "${{CMAKE_CURRENT_SOURCE_DIR}}/tools/{src}")',
                f'target_include_directories({target} PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/{tag}")',
                f'target_link_libraries({target} PRIVATE {lib})',
                f'target_compile_definitions({target} PRIVATE EDMIR_SELECTOR_BATCH=1)' if mode=="batch" else f'add_test(NAME {target} COMMAND {target})',"endif()"]
    (out/"generated_targets.cmake").write_text("\n".join(cmake)+"\n")
    plan=dict(schema="edmir.w1.precision.v2",shapes=SHAPES,seeds=SEEDS,jobs=jobs,
              cutlass=CUTLASS_REVISION,nccl=NCCL_REVISION,warmup=20,iterations=100,
              primary_blocks=len(families)*4*3,paired_blocks=0,
              hardware="8x A100-SXM4-80GB full NV12 P2P",novel_candidates=new,
              admission="All three full-output correctness checks pass before timing; unknown/failed has no score",
              current_status="local descriptors validated; CUDA compilation and GPU execution pending")
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    for name in ("inputs.json","nccl-artifacts.json"):
        shutil.copyfile(archived/name,out/name)
    src=out/"source/EDM-IR"
    shutil.copytree(REPO/"EDM-IR/runtime",src/"runtime",ignore=shutil.ignore_patterns("build","__pycache__"))
    for sub in ("edmir","tools"):
        for path in (REPO/"EDM-IR"/sub).rglob("*.py"):
            dest=src/path.relative_to(REPO/"EDM-IR");dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    (out/"files.json").write_text(json.dumps({str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},indent=2)+"\n")
    print(json.dumps(dict(shape_count=4,candidates_per_shape=len(families),jobs=len(jobs),GPU_started=False)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--archived",type=Path,default=REPO/"experiments/2026-11-11-shadeform-w1/protocol")
    args=parser.parse_args();prepare(args.out.resolve(),args.archived.resolve())
