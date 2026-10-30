"""Existing stress partitions, all regressions and full traversal cost."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import parallel_depth_selection as model
from review_calibrated_structure import aggregate, DATA, PROTECTED


def main():
    out=HERE/'audit';out.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    variants=(('depth','latency'),('depth_parallel_group','latency'),('depth_parallel_group','relative_pair'))
    cuts=('all_M512','all_N2048','all_host_winners')
    paths=[DATA,Path(__file__),HERE/'run-06/results.json',ROOT/'EDM-IR/edmir/compiler/parallel_depth_selection.py',
        ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py']
    paths.extend(ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{n}.json' for n in cuts)
    design={'cuts':cuts,'variants':variants,'frozen_six_used':False,'max_seconds':60,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    rows=json.loads(DATA.read_text())['p4']['rows'];results={};started=time.monotonic()
    for cut in cuts:
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{cut}.json').read_text())['model']['prior']
        train_shapes={tuple(s) for s in prior['training_shapes']}
        train=[r for r in rows if tuple(r['shape']) in train_shapes]
        test=[{**r,'split':'evaluation'} for r in rows if tuple(r['shape']) not in train_shapes]
        results[cut]={}
        for v,o in variants:
            if time.monotonic()-started>60:raise TimeoutError('bounded audit')
            fit=model.fit(train,prior,v,objective=o);dd=model.evaluate(fit,test);a=aggregate(dd)
            a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
            key=v if o=='latency' else v+'__'+o
            (out/f'{cut}-{key}.json').write_text(json.dumps({'model':fit,'evaluation':a},indent=2)+'\n')
            results[cut][key]={k:v for k,v in a.items() if k!='decisions'}
    full=json.loads((HERE/'run-06/results.json').read_text())['variants']
    baseline={tuple(d['shape']):d for d in full['depth']['decisions']}
    changes={};target_details={}
    for key in ('depth','depth_parallel_group','depth_parallel_group__relative_pair'):
        changes[key]=[]
        for d in full[key]['decisions']:
            b=baseline[tuple(d['shape'])]
            if d['provisional_choice']!=b['provisional_choice']:
                changes[key].append({'shape':d['shape'],'old_choice':b['provisional_choice'],
                    'new_choice':d['provisional_choice'],'old_regret':b['provisional_regret'],
                    'new_regret':d['provisional_regret'],'measured_us':d['measured_us']})
            if d['shape']==[4,64,8192,1024]:
                target_details[key]={'predicted_gap_group8_minus_q4_us':math.exp(d['predicted_log_us']['splitk8_groupedn2'])-math.exp(d['predicted_log_us']['splitk4']),
                    'measured_gap_us':d['measured_us']['splitk8_groupedn2']-d['measured_us']['splitk4'],
                    'prediction_us':{n:math.exp(d['predicted_log_us'][n]) for n in ('splitk4','splitk8_groupedn2')},
                    'local_K_correction_applied':any(d['curve_details'][n]['curve_applied'] for n in ('splitk4','splitk8_groupedn2'))}
    # Reload all nine variants: serialization/current-source reproduction, no refit.
    max_difference=0.
    for key,a in full.items():
        for f in range(4):
            fit=json.loads((HERE/f'run-06/{key}-fold{f}.json').read_text())
            for d in fit['evaluation']['decisions']:
                rr=[r for r in rows if r['shape']==d['shape']]
                replay=model.predict(fit['model'],rr)
                if replay['provisional_choice']!=d['provisional_choice']:raise ValueError('saved score replay changed')
                max_difference=max(max_difference,max(abs(v-d['predicted_log_us'][n]) for n,v in replay['predicted_log_us'].items()))
    runtime=[];work=[]
    key='depth_parallel_group__relative_pair'
    for f in range(4):
        fitted=json.loads((HERE/f'run-06/{key}-fold{f}.json').read_text())
        for d in fitted['evaluation']['decisions']:
            work.append((fitted['model'],[r for r in rows if r['shape']==d['shape']]))
    for repeat in range(21):
        if time.monotonic()-started>60:raise TimeoutError('bounded traversal audit')
        t=time.perf_counter()
        for fit,rr in work:model.predict(fit,rr)
        if repeat:runtime.append((time.perf_counter()-t)*1000)
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during audit: '+p)
    result={'stress':results,'changed_choices':changes,'grouped_target':target_details,
        'full_traversal_41_shapes_346_candidates_ms':{'median':statistics.median(runtime),'max':max(runtime),'repetitions':20},
        'saved_scores_max_difference':max_difference,'seconds':time.monotonic()-started,
        'protected_unchanged':True,'frozen_six_used':False,'admitted_new_default':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_choices',)},indent=2))


if __name__=='__main__':main()
