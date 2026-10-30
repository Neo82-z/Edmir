#!/usr/bin/env python3
"""A single fixed K-curve control, with unchanged folds and all failures saved."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(Path(__file__).parent)]
from edmir.compiler import conditional_curve_selection as model
from review_calibrated_structure import aggregate, DATA, PROTECTED


def main(out, modes):
    out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protected=json.loads(PROTECTED.read_text())
    for p,d in protected.items():
        if sha(ROOT/p)!=d:raise ValueError('protected file changed: '+p)
    data=json.loads(DATA.read_text())['p4']
    folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    jobs=[(mode,fold) for mode in modes for fold in range(4)]
    sources=[Path(__file__),ROOT/'EDM-IR/edmir/compiler/conditional_curve_selection.py',DATA,
             ROOT/'EDM-IR/edmir/compiler/calibrated_candidate_selection.py']
    sources += [ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json' for fold in range(4)]
    design={'jobs':jobs,'rule':'same identity/precision/candidate/M/N; linear K interpolation only between nearest measured training anchors',
            'modes':{'prior':'same-fold prior only','latency':'direct end-to-end time interpolation where bracketed',
                     'prior_residual':'interpolate measured-minus-prior residual in microseconds where bracketed',
                     'host_interval':'host-only nonnegative K interval slopes x [1,log2(ceil(M/64)),N/512], training K knots; fused unchanged',
                     'fused_residual':'original host; same-candidate K residual only for fused',
                     'host_interval_fused_residual':'host interval model + fused K residual; 2x2 factorial with prior control',
                     'host_tiles':'same interval model replacing log M by ceil(M/64)-1, unchanged coefficient count',
                     'host_tiles_fused_residual':'host tiles interval model + fused K residual'},
            'outside_bracket':'retain explicit prior prediction; no synthetic labels or zero-regret abstention',
            'optimizer_search':False,'retry_on_validation_failure':False,'total_seconds_cap':60,
            'frozen_six_used':False,'new_gpu_cost':0,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'protected_files':protected,'primary':'complete candidate-set misselection, bidirectional host/fused errors, mean/max regret',
            'limitation':'retrospective development; curve interpolation is not causal communication/compute decomposition'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    started=time.monotonic(); results={}; models=[]
    for mode,fold in jobs:
        if time.monotonic()-started>60:raise TimeoutError('bounded curve review')
        train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
        test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
        fitted=model.fit(train,prior,mode=mode); decisions=model.evaluate(fitted,test)
        models.append((mode,fitted,test))
        artifact={'model':fitted,'evaluation':aggregate(decisions)}
        (out/f'{mode}-fold{fold}.json').write_text(json.dumps(artifact,indent=2)+'\n')
        results.setdefault(mode,[]).extend(decisions)
    summary={}
    for mode,decisions in results.items():
        a=aggregate(decisions)
        a['wrong_choices']=sum(d['provisional_regret']>0 for d in decisions)
        a['total_shapes']=len(decisions)
        a['curved_candidates']=sum(v['curve_applied'] for d in decisions for v in d['curve_details'].values())
        a['candidate_count']=sum(len(d['measured_us']) for d in decisions)
        a['host_interval_candidates']=sum(v['host_interval_applied'] for d in decisions for v in d['curve_details'].values())
        a['wrong_shapes']=[{'shape':d['shape'],'choice':d['provisional_choice'],
            'best':min(d['measured_us'],key=d['measured_us'].get),'regret':d['provisional_regret']} for d in decisions if d['provisional_regret']>0]
        summary[mode]=a
    # Include unmodified previous models, without retuning or relabelling them.
    previous_path=ROOT/'experiments/2026-10-28-calibrated-structure/run-01/results.json'
    previous=json.loads(previous_path.read_text())
    comparisons={name:{**{k:v for k,v in a.items() if k!='decisions'},
                 'wrong_choices':sum(d['provisional_regret']>0 for d in a['decisions'])}
                 for name,a in previous['variants'].items() if name.startswith('p4')}
    samples=[]
    for _ in range(10):
        begin=time.perf_counter()
        for mode,fitted,test in models:
            if mode!=modes[-1]:continue
            for shape in sorted({tuple(r['shape']) for r in test}):
                model.predict(fitted,[r for r in test if tuple(r['shape'])==shape])
        samples.append((time.perf_counter()-begin)*1000)
    for p,d in {**protected,**design['source_hashes']}.items():
        if sha(ROOT/p)!=d:raise ValueError('input changed during review: '+p)
    output={'variants':summary,'previous_results_sha256':sha(previous_path),'previous':comparisons,
            'seconds':time.monotonic()-started,'inference_all_41_shapes_ms':{'median':statistics.median(samples),'max':max(samples)},
            'protected_unchanged':True,'compiler_admission':False,'frozen_six_used':False}
    (out/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ('decisions','wrong_shapes')} for k,v in summary.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=model.MODES,default=['prior','latency','prior_residual'])
    args=p.parse_args();main(args.out,args.modes)
