"""Fixed operand reuse scenarios and unchanged scoring controls. No fitting."""
from itertools import product
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import operand_reuse_model as reuse
from edmir.compiler import axis_residual_selection as axis
from edmir.compiler.dependency_wave_model import KernelCase
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def cases():
    specs=[(64,512,n,q,1) for n in (128,256,512,1024,2048) for q in (1,2)]
    specs += [(m,4096,n,q,1) for m,n in ((64,2048),(256,512)) for q in (2,4)]
    specs += [(64,8192,1024,q,g) for q,g in product((4,8),(1,2))]
    specs += [(512,4096,512,q,1) for q in (2,4)]
    return [KernelCase('conditional-uniform-P4-FP16',4,m,k,n,q,g) for m,k,n,q,g in specs]


def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design=json.loads((HERE/'design-before.json').read_text());guard={**design['source_hashes'],**design['protected_hashes']}
    for p,h in guard.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior source/evidence changed: '+p)
    (out/'execution-before.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'EDM-IR/edmir/compiler/operand_reuse_model.py')},indent=2)+'\n')
    results=[]
    for cap,order,remote in product(design['trace_scenarios']['cache_capacity_bytes'],('m_fast','n_fast'),(False,True)):
        scenario=reuse.ReuseScenario(cap,order,remote,True,2)
        for c in cases():
            r=reuse.replay(c,scenario);results.append(r)
            if time.monotonic()-start>60:raise TimeoutError('review deadline')
    def select(m,k,n,q,g,cap,order,remote):
        return next(r for r in results if r['shape_P_M_K_N']==[4,m,k,n] and r['q']==q and r['group_n']==g
            and r['scenario']['capacity_bytes']==cap and r['scenario']['tile_order']==order and r['scenario']['remote_A_cached']==remote)
    differences=[]
    for cap,order,remote in product(design['trace_scenarios']['cache_capacity_bytes'],('m_fast','n_fast'),(False,True)):
        for n in (128,256,512,1024,2048):
            a=select(64,512,n,1,1,cap,order,remote)['passes'][1]['operands']
            b=select(64,512,n,2,1,cap,order,remote)['passes'][1]['operands']
            differences.append({'N':n,'capacity':cap,'order':order,'remote_cached':remote,
                'warm_q2_minus_q1_read_miss_bytes':{x:b[x]['read_miss_bytes']-a[x]['read_miss_bytes'] for x in a},
                'warm_q2_minus_q1_write_bytes':b['C']['write_bytes']-a['C']['write_bytes']})
    reference=json.loads((ROOT/'experiments/2026-10-28-local-simulation-revision/verified/results.json').read_text())
    with diagnose.forbid_training() as calls:
        rows,folds,models=diagnose.load_inputs();new={}
        for fold,base in models.items():new[fold]=axis.assemble(base,[r for r in rows if folds[tuple(r['shape'])]!=fold],'M_work_extension')
        for name,predict,mm in (('retained',axis.base.predict,models),('M_work_extension',axis.predict,new)):
            for old in reference['decisions'][name]:
                shape=tuple(old['shape']);rr=[r for r in rows if tuple(r['shape'])==shape]
                p=predict(mm[folds[shape]],rr)
                if p['predicted_log_us']!=old['prediction']['predicted_log_us']:raise AssertionError('score changed')
    full=[r for r in results if r['scenario']['capacity_bytes']==41943040]
    summary={'simulations':len(results),'total_references':sum(r['references'] for r in results),
        'max_single_simulation_seconds':max(r['cpu_seconds'] for r in results),
        '40MiB_warm_B_misses_zero_count':sum(r['passes'][1]['operands']['B']['read_miss_bytes']==0 for r in full),
        '40MiB_scenarios':len(full),'maximum_conditional_footprint_bytes':max(r['footprint']['conditional_cacheable_bytes'] for r in results),
        'smallest_cache_short_K_miss_benefit_cases':sum(any(v<0 for v in d['warm_q2_minus_q1_read_miss_bytes'].values()) for d in differences if d['capacity']==65536),
        'score_changes':0,'retained_wrong':reference['metrics']['retained']['wrong'],
        'M_work_experiment_wrong':reference['metrics']['M_work_extension']['wrong'],
        'training_calls':calls,'GPU_started':False,'frozen_six_parsed_or_scored':False,'new_measured_or_training_labels':0,
        'cpu_seconds':time.monotonic()-start}
    for p,h in guard.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior source/evidence changed: '+p)
    (out/'results.json').write_text(json.dumps({'summary':summary,'scenarios':results,'short_K_pair_differences':differences},indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
