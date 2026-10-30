"""Test the attributed successful change with all other prior terms frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import curvature_transfer_selection as model
from review_calibrated_structure import aggregate,DATA,PROTECTED


def main(out):
    out.mkdir(exist_ok=False,parents=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected changed: '+p)
    data=json.loads(DATA.read_text())['p4'];folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    sources=[DATA,Path(__file__),HERE/'attribution.json',ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py',
        ROOT/'EDM-IR/edmir/compiler/curvature_transfer_selection.py']
    jobs=[(v,f) for v in ('previous','depth','fixed','intercept') for f in range(4)]
    design={'jobs':jobs,'folds':data['folds'],'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'hypothesis':'the depth curve supplies the three successful rank changes; other-term drift is unnecessary and worsens the new grouped-N error',
        'controls':'previous seven-error model, five-error depth model, frozen other terms, frozen terms with one training-only intercept',
        'loss_and_capacity':'original same-fold NNLS, ridge, capacity; no pairwise loss or new feature axes',
        'frozen_six_used':False,'gpu_cost':0,'max_seconds':60,'independent_validation':False}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    start=time.monotonic();variants={}
    for variant,fold in jobs:
        if time.monotonic()-start>60:raise TimeoutError('bounded cause review')
        train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
        test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
        if variant=='previous':fit=model.stage.curves.fit(train,prior,mode='host_tiles_fused_residual')
        elif variant=='depth':fit=model.stage.fit(train,prior)
        else:fit=model.fit(train,prior,variant)
        dd=model.evaluate(fit,test);variants.setdefault(variant,[]).extend(dd)
        (out/f'{variant}-fold{fold}.json').write_text(json.dumps({'model':fit,'evaluation':aggregate(dd)},indent=2)+'\n')
    original=json.loads((ROOT/'experiments/2026-10-28-split-stage-review/final-validation/results.json').read_text())['variants']
    for v in ('previous','depth'):
        if variants[v]!=original[v]['decisions']:raise ValueError('control predictions changed')
    summaries={}
    for v,dd in variants.items():
        a=aggregate(dd);a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
        a['wrong_shapes']=[{'shape':d['shape'],'choice':d['provisional_choice'],'best':min(d['measured_us'],key=d['measured_us'].get),
            'regret':d['provisional_regret']} for d in dd if d['provisional_regret']>0]
        summaries[v]=a
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during fit')
    result={'variants':summaries,'seconds':time.monotonic()-start,'protected_unchanged':True,'compiler_admission':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{k:x for k,x in a.items() if k!='decisions'} for v,a in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
