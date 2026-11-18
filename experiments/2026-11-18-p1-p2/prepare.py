#!/usr/bin/env python3
"""Seal P1/P2 inputs, predictions and descriptors without executing a GPU."""
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'EDM-IR/tools')]
import numpy as np
import prepare_w1_gpu as w1
from prepare_selector_gpu import write_inputs, implementation_id, MERGED_FAMILIES
from edmir.compiler import w1_candidate_selection as score

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def main():
    protected=json.loads((ROOT/'artifacts/2026-11-18-repository-sync/protected-files.json').read_text())
    assert all(sha(f)==h for f,h in protected.items())
    save(HERE/'protection-before.json', protected)
    protocol_path=ROOT/'experiments/2026-10-28-shape-evaluation/frozen/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    primary=protocol['candidates']
    assert primary==list(w1.PRIMARY) and len(primary)==13
    shapes={f'b{i+1}':s for i,s in enumerate(protocol['frozen_boundary_shapes'])}
    assert len(shapes)==6
    shapes['m512']=[8,512,8192,512]
    p1=['pipeline_serial','pipeline','splitk4','splitk4_groupedn2']
    model_path=ROOT/'experiments/2026-11-17-final-operator/frozen-predictions/coupled_geometry-large_n.json'
    model=json.loads(model_path.read_text())['model']
    predictions={}
    for tag,shape in shapes.items():
        if tag=='m512':continue
        scores={}; support={}; queries=[]
        for family in primary:
            row=dict(shape=shape,candidate_id=family,execution_context=model['execution_context'],
                     source_id=model['source_id'],phase='measure',precision=score.precision(family),
                     error_contract=score.ERROR_CONTRACT,sm_count=108,
                     implementation_id=implementation_id(family,'merged' if family in MERGED_FAMILIES else 'prepared',
                         dependency_policy='uniform_immutable_peer_v1' if family.startswith(('direct','splitk')) else 'resource_queues_v1'))
            # This evaluates the fixed numerical head; it is not evidence admission.
            x=(np.asarray(score.features(row,model['variant']))-model['means'])/model['scales']
            log_us=float(x@np.asarray(model['theta'][:-1])+model['theta'][-1])
            if not math.isfinite(log_us) or not -20<log_us<30:raise ValueError('nonfinite frozen extrapolation')
            scores[family]=math.exp(log_us)
            support[family]=score.support_key(row) in model['support']
            queries.append(row)
        key=lambda f:(scores[f],0 if f in score.HOSTS else 1,f)
        rank=sorted((f for f in primary if support[f]),key=key)
        assert rank
        predictions[tag]=dict(shape=shape,scores_us=scores,training_identity_supported=support,
             frozen_ranking=rank,queries_for_geometry_only=queries,
             extrapolated_axes={a:dict(value=shape[i],training_range=model['shape_axis_ranges'][a])
                 for i,a in enumerate(('P','M','K','N')) if not model['shape_axis_ranges'][a][0]<=shape[i]<=model['shape_axis_ranges'][a][1]},
             compiler_admission=False,interpretation='prelabel frozen-head transfer diagnostic; strict new-context refusal reported separately')
    save(HERE/'preregistered-predictions.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
         primary_model=str(model_path.relative_to(ROOT)),primary_model_sha256=sha(model_path),
         reason='Same coupled large-N model already frozen for the final replication; no model selection using six-shape results',
         model_fitted=False,predictions=predictions,tie_rule='host first, then candidate_id',
         new_context_policy='No rebinding, no bypass of production admission; descriptive ranking evaluated separately'))
    shutil.copyfile(model_path,HERE/'frozen-primary-model.json')
    bundle=HERE/'bundle';bundle.mkdir(exist_ok=False)
    cmake=[]
    for tag,shape in shapes.items():
        w1.registry(bundle/tag,shape,p1 if tag=='m512' else primary)
        for mode,src,lib in [('layout','check_w1_registry.cpp','edmir_runtime'),('batch','benchmark_selector.cpp','edmir_cuda_runtime')]:
            target=f'w1_{tag}_{mode}'
            cmake += ['if(EDMIR_ENABLE_CUDA)' if mode=='batch' else 'if(TRUE)',
                f'add_executable({target} "${{CMAKE_CURRENT_SOURCE_DIR}}/tools/{src}")',
                f'target_include_directories({target} PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/{tag}")',
                f'target_link_libraries({target} PRIVATE {lib})',
                f'target_compile_definitions({target} PRIVATE EDMIR_SELECTOR_BATCH=1)' if mode=='batch' else f'add_test(NAME {target} COMMAND {target})','endif()']
    (bundle/'generated_targets.cmake').write_text('\n'.join(cmake)+'\n')
    src=bundle/'source/EDM-IR'
    shutil.copytree(ROOT/'EDM-IR/runtime',src/'runtime',ignore=shutil.ignore_patterns('build','__pycache__'))
    for sub in ('edmir','tools'):
        for p in (ROOT/'EDM-IR'/sub).rglob('*.py'):
            dest=src/p.relative_to(ROOT/'EDM-IR');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    inputs={}
    for tag,shape in shapes.items():
        inputs[tag]={}
        for seed in w1.SEEDS:
            inputs[tag][str(seed)]=write_inputs(shape,HERE/'inputs'/tag/str(seed),seed=seed)
            print('CPU inputs/reference ready',tag,seed,flush=True)
    save(bundle/'inputs.json',inputs)
    shutil.copyfile(ROOT/'experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2/nccl-artifacts.json',bundle/'nccl-artifacts.json')
    jobs=[]
    # Check all seeds before any timed block for a given surface.
    for tag in shapes:
        families=p1 if tag=='m512' else primary
        for seed in w1.SEEDS:jobs.append(dict(shape=tag,phase='check',seed=seed,repetition=0,order=families))
    # Six paired orders at one existing point; equal work, only dependency serialization differs.
    for rep in range(6):
        order=p1[rep%4:]+p1[:rep%4]
        if rep%2:order=list(reversed(order))
        jobs.append(dict(shape='m512',phase='pair',seed=w1.SEEDS[rep%3],repetition=rep,order=order))
    # Each measured selection block is separate from all oracle verification blocks.
    # Three verification blocks per frozen shape. Never retune after these labels.
    rng=random.Random(20261118)
    for tag in predictions:
        order=list(primary);rng.shuffle(order)
        for rep in range(3):
            shift=4*rep
            jobs.append(dict(shape=tag,phase='verify',seed=w1.SEEDS[0],repetition=rep,order=order[shift:]+order[:shift]))
        jobs.append(dict(shape=tag,phase='select',seed=w1.SEEDS[0],repetition=0,order=[],
            derive_order='union of first three numerically admitted frozen ranks and fixed controls; deterministic candidate-id order'))
    plan=dict(schema='edmir.p1-p2.closeout.v1',shapes=shapes,seeds=list(w1.SEEDS),primary=primary,p1_candidates=p1,
         jobs=jobs,cutlass=w1.CUTLASS_REVISION,nccl=w1.NCCL_REVISION,warmup=20,iterations=100,
         frozen_protocol_sha256=sha(protocol_path),predictions_sha256=sha(HERE/'preregistered-predictions.json'),
         model_sha256=sha(model_path),fixed_order=['host_merged_cublaslt','splitk4_groupedn2','host_merged_cutlass'],
         selection_policy='First three admitted model-supported frozen ranks; separate selection block chooses fastest for k=1/2/3. Verification never chooses shortlist.',
         p1_trace_cases={f'm512-{f}':['m512',f] for f in p1},
         p1_question='At fixed work and precision, does relaxing pipeline dependencies overlap communication and GEMM; what changes jointly under q4 g1->g2?',
         p1_limits='No attribution of physical traffic, occupancy or causal hidden-communication percentage from timeline alone',
         max_correctness_rows=246,max_verification_rows=234,max_selection_rows=36,p1_timing_rows=24,
         budget_usd=25,compute_cap_usd=24,recovery_reserve_seconds=600,no_training=True)
    save(bundle/'plan.json',plan)
    save(bundle/'files.json',{str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()})
    assert all(sha(f)==h for f,h in protected.items())
    print('P1/P2 sealed:',len(shapes),'shapes;',len(jobs),'process groups',flush=True)

if __name__=='__main__':main()
