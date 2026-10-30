"""Predeclared 2x2 split activation/depth controls on the unchanged four folds."""
import argparse
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


def main(out):
    out.mkdir(exist_ok=False,parents=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    data=json.loads(DATA.read_text())['p4'];folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    sources=[DATA,Path(__file__),ROOT/'EDM-IR/edmir/compiler/conditional_curve_selection.py',
             ROOT/'EDM-IR/edmir/compiler/split_stage_selection.py',ROOT/'EDM-IR/edmir/compiler/additive_latency_selection.py']
    jobs=[(v,f) for v in ('previous',*model.VARIANTS) for f in range(4)]
    design={'jobs':jobs,'same_folds':data['folds'],'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'fit':'shape-balanced nonnegative squared latency error; same training-fold ridge/capacity as existing prior',
        'factorial':'one source-motivated split-path activation feature x shared monotone partition-depth curve',
        'depth_knots':'distinct K/(32*q) in training rows only; no hand-selected short-K threshold',
        'host':'previous supported tile/interval model unchanged',
        'fused_curve':'same-M/N K residual anchors recomputed against each fitted prior; no unseen labels',
        'primary':'whole-candidate misselections and bidirectional host/fused errors; all abstentions retained',
        'max_seconds':60,'frozen_six_used':False,'gpu_cost':0,'retries_or_optimizer_search':False,
        'evidence_scope':'retrospective development, not independent validation; no isolated latency attribution'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    started=time.monotonic();variants={};models=[]
    for variant,fold in jobs:
        if time.monotonic()-started>60:raise TimeoutError('bounded split stage review')
        train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
        test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
        fitted=model.curves.fit(train,prior,mode='host_tiles_fused_residual') if variant=='previous' else model.fit(train,prior,variant)
        decisions=model.evaluate(fitted,test);models.append((variant,fitted,test))
        variants.setdefault(variant,[]).extend(decisions)
        (out/f'{variant}-fold{fold}.json').write_text(json.dumps({'model':fitted,'evaluation':aggregate(decisions)},indent=2)+'\n')
    summaries={}
    for variant,dd in variants.items():
        a=aggregate(dd);a['wrong_choices']=sum(d['provisional_regret']>0 for d in dd)
        a['wrong_shapes']=[{'shape':d['shape'],'choice':d['provisional_choice'],
            'best':min(d['measured_us'],key=d['measured_us'].get),'regret':d['provisional_regret']} for d in dd if d['provisional_regret']>0]
        a['short_K_wrong']=sum(d['shape'][2]==512 and d['provisional_regret']>0 for d in dd)
        summaries[variant]=a
    # Control: refitting the unchanged basis must reproduce previous scores.
    for a,b in zip(variants['previous'],variants['linear']):
        if a['shape']!=b['shape'] or max(abs(a['predicted_log_us'][n]-b['predicted_log_us'][n]) for n in a['predicted_log_us'])>1e-10:
            raise ValueError('linear control does not reproduce existing predictions')
    for p,h in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=h:raise ValueError('input changed during review')
    result={'variants':summaries,'seconds':time.monotonic()-started,'protected_unchanged':True,
            'linear_control_reproduced':True,'compiler_admission':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{k:x for k,x in a.items() if k not in ('decisions','wrong_shapes')} for v,a in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
