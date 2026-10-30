"""Audit all failures, existing group holdouts and bounded local inference."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import split_stage_selection as model
from review_calibrated_structure import aggregate, DATA, PROTECTED


def main():
    out=HERE/'audit';out.mkdir(exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    run=HERE/'final-validation'
    results=json.loads((run/'results.json').read_text())['variants']
    cuts=('all_M512','all_N2048','all_host_winners')
    sources=[DATA,Path(__file__),run/'results.json',ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py',
        ROOT/'EDM-IR/edmir/compiler/conditional_curve_selection.py']
    sources.extend(ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{name}.json' for name in cuts)
    design={'cuts':cuts,'variants':['previous','depth'],'frozen_six_used':False,
        'new_split_or_hyperparameter_search':False,'max_seconds':60,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    started=time.monotonic();data=json.loads(DATA.read_text())['p4']['rows']
    stress={}
    for name in cuts:
        old=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{name}.json').read_text())
        prior=old['model']['prior'];shapes={tuple(s) for s in prior['training_shapes']}
        train=[r for r in data if tuple(r['shape']) in shapes]
        test=[{**r,'split':'evaluation'} for r in data if tuple(r['shape']) not in shapes]
        stress[name]={}
        for variant in ('previous','depth'):
            if time.monotonic()-started>60:raise TimeoutError('bounded audit')
            fit=(model.curves.fit(train,prior,mode='host_tiles_fused_residual') if variant=='previous'
                 else model.fit(train,prior,'depth'))
            dd=model.evaluate(fit,test);a=aggregate(dd)
            a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
            (out/f'{name}-{variant}.json').write_text(json.dumps({'model':fit,'evaluation':a},indent=2)+'\n')
            stress[name][variant]={k:v for k,v in a.items() if k!='decisions'}
    before={tuple(d['shape']):d for d in results['previous']['decisions']}
    after={tuple(d['shape']):d for d in results['depth']['decisions']}
    changed=[];short=[]
    for s,d in after.items():
        b=before[s]
        if b['provisional_choice']!=d['provisional_choice']:
            changed.append({'shape':s,'old_choice':b['provisional_choice'],'new_choice':d['provisional_choice'],
                'measured_best':min(d['measured_us'],key=d['measured_us'].get),
                'old_regret':b['provisional_regret'],'new_regret':d['provisional_regret'],
                'old_predicted_us':{n:math.exp(v) for n,v in b['predicted_log_us'].items()},
                'new_predicted_us':{n:math.exp(v) for n,v in d['predicted_log_us'].items()},
                'measured_us':d['measured_us']})
        if s[2]==512:
            delta=lambda dd:math.exp(dd['predicted_log_us']['splitk2'])-math.exp(dd['predicted_log_us']['direct'])
            short.append({'shape':s,'previous_q2_minus_direct_us':delta(b),'new_q2_minus_direct_us':delta(d),
                'measured_q2_minus_direct_us':d['measured_us']['splitk2']-d['measured_us']['direct'],
                'choice':d['provisional_choice'],'supported':not d['requires_validation']})
    coefficients=[];fold_work=[]
    for fold in range(4):
        ff=json.loads((run/f'depth-fold{fold}.json').read_text())
        names=ff['model']['stage_prior']['fit']['names'];coef=ff['model']['stage_prior']['fit']['coefficients']
        activation=json.loads((run/f'depth_activation-fold{fold}.json').read_text())['model']['stage_prior']['fit']
        coefficients.append({'fold':fold,'depth_fit':dict(zip(names,coef)),
            'kkt':ff['model']['stage_prior']['fit']['kkt_max_violation'],
            'activation_when_depth_present':dict(zip(activation['names'],activation['coefficients']))['split_serial_path_enabled']})
        shapes={tuple(d['shape']) for d in ff['evaluation']['decisions']}
        fold_work.append((ff['model'],[[r for r in data if tuple(r['shape'])==s] for s in sorted(shapes)]))
    # Twenty repetitions, same full 41 shapes; no candidate pruning. Warm cache.
    runtimes=[]
    for repeat in range(21):
        if time.monotonic()-started>60:raise TimeoutError('bounded audit')
        t=time.perf_counter()
        for fitted,surfaces in fold_work:
            for surface in surfaces:model.predict(fitted,surface)
        if repeat:runtimes.append((time.perf_counter()-t)*1000)
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during audit: '+p)
    result={'changed_decisions':changed,'short_K_deltas':short,'coefficients':coefficients,'stress':stress,
        'inference_41_shapes_346_candidates_ms':{'median':statistics.median(runtimes),'max':max(runtimes),'repetitions':20},
        'protected_unchanged':True,'seconds':time.monotonic()-started,'compiler_admission':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('changed_decisions','coefficients')},indent=2))


if __name__=='__main__':main()
