"""Fixed stress cuts, correction margins and bounded score traversal."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import curvature_transfer_selection as model
from review_calibrated_structure import aggregate,DATA,PROTECTED


def main():
    out=HERE/'audit';out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    cuts=('all_M512','all_N2048','all_host_winners')
    sources=[DATA,Path(__file__),HERE/'final-validation/results.json',ROOT/'EDM-IR/edmir/compiler/curvature_transfer_selection.py',
        ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py']
    sources.extend(ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{cut}.json' for cut in cuts)
    design={'cuts':cuts,'variants':['depth','fixed','intercept'],'max_seconds':60,'frozen_six_used':False,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    rows=json.loads(DATA.read_text())['p4']['rows'];stress={};started=time.monotonic()
    for cut in cuts:
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/stress/{cut}.json').read_text())['model']['prior']
        shapes={tuple(s) for s in prior['training_shapes']}
        train=[r for r in rows if tuple(r['shape']) in shapes]
        test=[{**r,'split':'evaluation'} for r in rows if tuple(r['shape']) not in shapes]
        stress[cut]={}
        for variant in ('depth','fixed','intercept'):
            if time.monotonic()-started>60:raise TimeoutError('bounded audit')
            fitted=model.stage.fit(train,prior) if variant=='depth' else model.fit(train,prior,variant)
            dd=model.evaluate(fitted,test);a=aggregate(dd);a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
            (out/f'{cut}-{variant}.json').write_text(json.dumps({'model':fitted,'evaluation':a},indent=2)+'\n')
            stress[cut][variant]={k:v for k,v in a.items() if k!='decisions'}
    final=json.loads((HERE/'final-validation/results.json').read_text())['variants']
    before={tuple(d['shape']):d for d in final['depth']['decisions']};changes=[]
    for d in final['intercept']['decisions']:
        b=before[tuple(d['shape'])]
        if b['provisional_choice']!=d['provisional_choice']:
            a=b['provisional_choice'];c=d['provisional_choice']
            changes.append({'shape':d['shape'],'old_choice':a,'new_choice':c,'old_regret':b['provisional_regret'],'new_regret':d['provisional_regret'],
                'predicted_winning_margin_us':math.exp(d['predicted_log_us'][a])-math.exp(d['predicted_log_us'][c]),
                'measured_winning_margin_us':d['measured_us'][a]-d['measured_us'][c],
                'requires_validation':d['requires_validation']})
    work=[];intercepts=[]
    for f in range(4):
        saved=json.loads((HERE/f'final-validation/intercept-fold{f}.json').read_text())
        state=saved['model']['stage_prior'];intercepts.append({'fold':f,**state['curvature_transfer']})
        for d in saved['evaluation']['decisions']:
            work.append((saved['model'],[r for r in rows if r['shape']==d['shape']],d))
    max_delta=0.;runtime=[]
    for repeat in range(21):
        if time.monotonic()-started>60:raise TimeoutError('bounded traversal')
        t=time.perf_counter()
        for fitted,rr,d in work:
            p=model.predict(fitted,rr)
            if not repeat:
                assert p['provisional_choice']==d['provisional_choice']
                max_delta=max(max_delta,max(abs(v-d['predicted_log_us'][n]) for n,v in p['predicted_log_us'].items()))
        if repeat:runtime.append((time.perf_counter()-t)*1000)
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during audit')
    result={'stress':stress,'changes_from_five_error_model':changes,'intercept_fits':intercepts,
        'full_traversal_41_shapes_ms':{'median':statistics.median(runtime),'max':max(runtime),'repetitions':20},
        'saved_score_max_delta':max_delta,'seconds':time.monotonic()-started,'protected_unchanged':True,
        'frozen_six_used':False,'compiler_admission':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='intercept_fits'},indent=2))


if __name__=='__main__':main()
