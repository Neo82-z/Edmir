"""Source and arithmetic evidence; no assertion of measured cache behavior."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import shape_transfer_selection as model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=HERE/'source-audit';out.mkdir(exist_ok=False)
    rows,_,_=diagnose.load_inputs()
    files={}
    for rel in ('EDM-IR/runtime/src/cuda_backend.cpp','EDM-IR/runtime/src/cutlass_sm80_gemm.cuh',
                'EDM-IR/runtime/src/nccl_device_fused_cutlass.cu'):
        text=subprocess.run(['git','show','f584d7d:'+rel],cwd=ROOT,text=True,capture_output=True,check=True,timeout=5).stdout
        p=out/Path(rel).name;p.write_text(text);files[rel]={'revision':'f584d7d','snapshot':str(p.relative_to(ROOT)),'sha256':sha(p)}
    header=(out/'cutlass_sm80_gemm.cuh').read_text();backend=(out/'cuda_backend.cpp').read_text()
    if 'GemmIdentityThreadblockSwizzle<>' not in header:raise AssertionError('swizzle changed')
    begin=backend.index('void prepare_fused_cutlass_workspaces()');end=backend.index('void release_fused_cutlass_workspaces()',begin)
    if 'cudaMemset(' not in backend[begin:end]:raise AssertionError('workspace initialization changed')
    if backend.count('cudaMemset(')!=backend[begin:end].count('cudaMemset('):raise AssertionError('other memset path requires review')
    cutlass=ROOT/'experiments/2026-10-28-paired-gpu-readiness/dependencies/cutlass/include/cutlass'
    for rel in ('gemm/threadblock/threadblock_swizzle.h','gemm/kernel/gemm.h'):
        p=cutlass/rel;files[str(p.relative_to(ROOT))]={'sha256':sha(p),'reviewed_dependency':'dcf215af68a2d08d305076c152a06f201728cd53','archived_binary_dependency_verified':False}
    swizzle=(cutlass/'gemm/threadblock/threadblock_swizzle.h').read_text()
    if 'template <int N = 1>' not in swizzle:raise AssertionError('swizzle default changed')
    kernel=(cutlass/'gemm/kernel/gemm.h').read_text()
    if 'semaphore.release(lock)' not in kernel or 'resets the semaphore for subsequent grids' not in kernel:raise AssertionError('semaphore lifecycle changed')
    def lookup(shape,family):return next(r for r in rows if r['shape']==list(shape) and r['candidate_id']==family)
    collision_shapes=((4,64,4096,2048),(4,256,4096,512))
    collision=[]
    for shape in collision_shapes:
        rr={q:lookup(shape,f'splitk{q}') for q in (2,4)}
        collision.append({'shape':shape,'q2_us':rr[2]['latency_us'],'q4_us':rr[4]['latency_us'],
            'q4_minus_q2_us':rr[4]['latency_us']-rr[2]['latency_us'],
            'q2_geometry':model.geometry(rr[2]),'q4_geometry':model.geometry(rr[4])})
    for q in (2,4):
        if model.work_key(lookup(collision_shapes[0],f'splitk{q}'))!=model.work_key(lookup(collision_shapes[1],f'splitk{q}')):
            raise AssertionError('count collision changed')
    gaps=[c['q4_minus_q2_us'] for c in collision]
    if gaps[0]*gaps[1]>=0:raise AssertionError('expected opposite empirical directions')
    observed_bound=abs(gaps[1]-gaps[0])/2
    result=json.loads((ROOT/'experiments/2026-10-28-local-simulation-revision/verified/results.json').read_text())
    before={tuple(d['shape']):d for d in result['decisions']['retained']};changes=[]
    for d in result['decisions']['M_work_extension']:
        b=before[tuple(d['shape'])]
        for n,t in d['observed_us'].items():
            old=math.exp(b['prediction']['predicted_log_us'][n]);new=math.exp(d['prediction']['predicted_log_us'][n])
            if old==new:continue
            detail=d['prediction']['axis_details'][n]
            shapes=[d['shape'],*[a['shape'] for x in detail['transfers'] for a in x['anchors']]]
            changes.append({'shape':d['shape'],'candidate':n,'old_us':old,'new_us':new,'observed_us':t,
                'shape_mean_mape_contribution_change':(abs(new/t-1)-abs(old/t-1))/len(d['observed_us'])/41,
                'M_tile_counts':[(s[1]+63)//64 for s in shapes],
                'M_tail_fractions':[(s[1]%64)/64 if s[1]%64 else 1. for s in shapes],
                'extrapolation_weights':detail['transfers'][0]['weights']})
    accounted=math.fsum(d['shape_mean_mape_contribution_change'] for d in changes)
    delta=result['metrics']['M_work_extension']['shape_mean_mape']-result['metrics']['retained']['shape_mean_mape']
    if not math.isclose(accounted,delta,abs_tol=1e-12):raise AssertionError('MAPE contribution account failed')
    payload={'sources':files,'source_findings':{'default_swizzle_N':1,'reviewed_default_swizzle_N_boundary_switch':False,
        'short_K_q2_partition_iterations':8,'short_K_q1_partition_iterations':16,
        'short_K_partition_padding':False,'workspace_memset_location':'construction/setup, not each measured iteration',
        'last_partition_releases_zero_for_next_grid':True,
        'source_is_proof_of_actual_archived_binary':False},
        'work_collision':collision,'minimum_max_gap_error_for_any_equal_work_only_function_us':observed_bound,
        'bound_scope':'one shared function restricted to this count-only representation; not a CV error lower bound and not a hardware-cause proof',
        'M_extension_changes':sorted(changes,key=lambda d:d['shape_mean_mape_contribution_change'],reverse=True),
        'total_shape_mean_mape_change':accounted,'GPU_started':False,'new_training':False,'frozen_six_parsed_or_scored':False}
    (out/'audit.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'source_findings':payload['source_findings'],'opposite_q4_minus_q2_gaps_us':gaps,
                     'equal_work_only_gap_error_bound_us':observed_bound,'MAPE_change_percentage_points':accounted*100},indent=2))


if __name__=='__main__':main()
