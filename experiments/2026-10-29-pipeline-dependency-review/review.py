"""Fixed synthetic dependency scenarios, no fit and no microsecond labels."""
from itertools import product
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import pipeline_dependency_model as pipeline
from edmir.compiler import dependency_wave_model as wave
from edmir.compiler import axis_residual_selection as axis
from audit import audit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def cases():
    specs=[(64,512,n,q,1) for n in (128,256,512,1024,2048) for q in (1,2)]
    specs += [(128,512,512,q,1) for q in (1,2)]
    specs += [(64,8192,1024,q,g) for q,g in product((4,8),(1,2))]
    specs += [(512,4096,512,q,1) for q in (2,4)]
    return [wave.KernelCase('conditional-P4-FP16',4,m,k,n,q,g) for m,k,n,q,g in specs]


def main(out):
    started=time.monotonic(); out.mkdir(parents=True,exist_ok=False)
    design=json.loads((HERE/'design-before.json').read_text())
    guard={**design['source_hashes'],**design['protected_hashes']}
    for p,h in guard.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior evidence changed: '+p)
    source=audit();(out/'source-audit.json').write_text(json.dumps(source,indent=2)+'\n')
    files=[Path(__file__),HERE/'audit.py',ROOT/'EDM-IR/edmir/compiler/pipeline_dependency_model.py']
    (out/'execution-before.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2)+'\n')
    grid=design['scenario_grid']; runs=[]; cache={}; rows,folds,bases=diagnose.load_inputs()
    def observed(m,k,n,q,g):
        family='direct' if q==1 else f'splitk{q}'
        if g==2:family+='_groupedn2'
        return next(r['latency_us'] for r in rows if r['shape']==[4,m,k,n] and r['candidate_id']==family)
    actual_short=[observed(64,512,n,2,1)-observed(64,512,n,1,1) for n in (128,256,512,1024,2048)]
    sign=lambda v:1 if v>0 else (-1 if v<0 else 0)
    for la,lb,barrier,width,slots,epi in product(grid['latency_A'],grid['latency_B'],grid['barrier'],grid['width_service'],grid['slot_counts'],grid['epilogue_units']):
        times={};details=[]
        for c in cases():
            key=(c.depth,c.group_n,la,lb,barrier,width)
            if key not in cache:
                p=pipeline.profile(c);scale=1 if width=='warp_parallel' else p['warps']
                s=pipeline.PipelineService(p['A_copy_loop_iterations_per_thread_per_stage']*scale,
                    p['B_copy_loop_iterations_per_thread_per_stage']*scale,la,lb,scale,barrier,1.,
                    'synthetic_units','fixed per-CTA services; ideal parallel warps or serialized warp issue; no resource calibration')
                cache[key]=pipeline.simulate_cta(c,s)
            cta=cache[key]
            # Output traffic increases with width; this timing ratio is an
            # explicit scenario, not a measurement of epilogue bandwidth.
            e=epi*c.group_n
            service=wave.ServiceScenario(cta['scenario_cta_time'],e,2*e,'synthetic_units','constant CTA services and fixed slot admission; first output write vs read+write fixture')
            r=wave.simulate(c,service,slots=slots)
            times[(c.m,c.k,c.n,c.q,c.group_n)]=r['scenario_makespan']
            details.append({'shape':[c.world_size,c.m,c.k,c.n],'q':c.q,'group_n':c.group_n,
                            'scenario_time':r['scenario_makespan'],'resident_wait_slot_time':r['resident_wait_slot_time'],
                            'all_fit':r['all_fit']})
        gaps=[times[64,512,n,2,1]-times[64,512,n,1,1] for n in (128,256,512,1024,2048)]
        if slots==108 and len(set(gaps))!=1:raise AssertionError('constant-service all-fit invariance failed')
        gg=[times[64,8192,1024,8,g]-times[64,8192,1024,4,g] for g in (1,2)]
        runs.append({'scenario':{'latency_A':la,'latency_B':lb,'barrier':barrier,'width_service':width,'slots':slots,'first_epilogue_per_g':epi},
                     'short_K_gaps':gaps,'short_K_signs':[sign(x) for x in gaps],
                     'grouped_q8_minus_q4_gaps_g1_g2':gg,'grouped_interaction':gg[1]-gg[0],
                     'cases':details})
        if time.monotonic()-started>60:raise TimeoutError('local review deadline')
    reference=json.loads((ROOT/'experiments/2026-10-28-local-simulation-revision/verified/results.json').read_text())
    with diagnose.forbid_training() as calls:
        alt={f:axis.assemble(base,[r for r in rows if folds[tuple(r['shape'])]!=f],'M_work_extension') for f,base in bases.items()}
        for name,predict,models in (('retained',axis.base.predict,bases),('M_work_extension',axis.predict,alt)):
            for old in reference['decisions'][name]:
                shape=tuple(old['shape']); surface=[r for r in rows if tuple(r['shape'])==shape]
                p=predict(models[folds[shape]],surface)
                if p['predicted_log_us']!=old['prediction']['predicted_log_us']:raise AssertionError('score changed')
    # Arbitrary constant all-fit services give one N-independent gap. Best
    # possible maximum absolute error is half the observed range, regardless
    # of how many pipeline subcomponents the single-CTA service contains.
    invariant_bound=(max(actual_short)-min(actual_short))/2
    grouped_observed=[observed(64,8192,1024,8,g)-observed(64,8192,1024,4,g) for g in (1,2)]
    summary={'scenarios':len(runs),'conditional_GEMM_evaluations':sum(len(r['cases']) for r in runs),
             'unique_CTA_graphs':len(cache),'max_CTA_graph_seconds':max(r['cpu_seconds'] for r in cache.values()),
             'all_fit_short_K_invariance_checks':sum(r['scenario']['slots']==108 for r in runs),
             'observed_short_K_q2_minus_direct_us':actual_short,
             'single_shared_all_fit_constant_gap_minimax_error_us':invariant_bound,
             'bound_scope':'one shared constant-gap function only; not a CV or hardware error bound',
             'short_K_sign_patterns':sorted({tuple(r['short_K_signs']) for r in runs}),
             'short_K_full_pattern_matches':sum(r['short_K_signs']==[sign(x) for x in actual_short] for r in runs),
             'observed_grouped_q8_minus_q4_gaps_us':grouped_observed,
             'grouped_two_direction_matches':sum([sign(x) for x in r['grouped_q8_minus_q4_gaps_g1_g2']]==[sign(x) for x in grouped_observed] for r in runs),
             'score_changes':0,'retained_wrong':reference['metrics']['retained']['wrong'],
             'M_work_experiment_wrong':reference['metrics']['M_work_extension']['wrong'],
             'training_calls':calls,'GPU_started':False,'frozen_six_parsed_or_scored':False,
             'new_training_labels':0,'compiler_admission':False,'cpu_seconds':time.monotonic()-started}
    for p,h in guard.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior evidence changed: '+p)
    payload={'summary':summary,'profiles':[{'shape':[c.world_size,c.m,c.k,c.n],'q':c.q,'g':c.group_n,**pipeline.profile(c)} for c in cases()],
             'CTA_graphs':[{'key':list(key),'result':value} for key,value in cache.items()],'runs':runs}
    (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
