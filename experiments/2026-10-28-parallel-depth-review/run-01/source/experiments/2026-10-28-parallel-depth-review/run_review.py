"""Bounded matched controls for concurrency x partition-depth interactions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import parallel_depth_selection as model
from review_calibrated_structure import aggregate, DATA, PROTECTED


def main(out):
    out.mkdir(exist_ok=False,parents=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    data=json.loads(DATA.read_text())['p4'];folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    sources=[DATA,Path(__file__),ROOT/'EDM-IR/edmir/compiler/conditional_curve_selection.py',
        ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py',ROOT/'EDM-IR/edmir/compiler/parallel_depth_selection.py',
        ROOT/'EDM-IR/edmir/compiler/additive_latency_selection.py']
    jobs=[(v,f) for v in model.VARIANTS for f in range(4)]
    design={'jobs':jobs,'folds':data['folds'],'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'concurrency_basis':'u=C/(C+B); B=ceil(M/64)*ceil(N/(128g))*q; C is unchanged empirical prior capacity',
        'coupling':'u times shared depth ramps, plus u-only control',
        'group_control':'replace group*K/q by nonnegative group*shared_depth_ramps; group_fixed retains floor',
        'fit':'unchanged fold ridge, capacity, shape-balanced NNLS; all feature coefficients nonnegative',
        'evaluation':'all candidates including unsupported cases; no tolerance band or forced direct choice',
        'frozen_six_used':False,'gpu_cost':0,'max_seconds':60,'compiler_admission':False,
        'evidence_scope':'development replay; neither independent validation nor physical time-component identification'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    started=time.monotonic();variants={}
    for variant,fold in jobs:
        if time.monotonic()-started>60:raise TimeoutError('bounded parallel-depth review')
        train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
        test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
        fit=model.fit(train,prior,variant);dd=model.evaluate(fit,test);variants.setdefault(variant,[]).extend(dd)
        (out/f'{variant}-fold{fold}.json').write_text(json.dumps({'model':fit,'evaluation':aggregate(dd)},indent=2)+'\n')
    old=json.loads((ROOT/'experiments/2026-10-28-split-stage-review/final-validation/results.json').read_text())['variants']['depth']['decisions']
    for a,b in zip(old,variants['depth']):
        if a!=b:raise ValueError('baseline depth predictions changed')
    summaries={}
    for variant,dd in variants.items():
        a=aggregate(dd);a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
        a['wrong_shapes']=[{'shape':d['shape'],'choice':d['provisional_choice'],
            'best':min(d['measured_us'],key=d['measured_us'].get),'regret':d['provisional_regret']} for d in dd if d['provisional_regret']>0]
        a['short_K_wrong']=sum(d['shape'][2]==512 and d['provisional_regret']>0 for d in dd)
        summaries[variant]=a
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during review')
    result={'variants':summaries,'seconds':time.monotonic()-started,'protected_unchanged':True,
            'previous_control_reproduced':True,'compiler_admission':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{k:x for k,x in a.items() if k!='decisions'} for v,a in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
