#!/usr/bin/env python3
"""Change host calibration coverage only, with all target-shape labels excluded."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_gpu_evidence import REPO, read, sha, load_evidence
from evaluate_mechanism_prediction import prediction
from evaluate_post_gpu_revision import write, score_scopes, choices
from edmir.compiler.mechanism_prediction import HOSTS, MechanismModel, basis
from edmir.compiler.prediction_support import linear_target_support

BASE=REPO/'experiments/2026-10-25-post-gpu-revision'


def unique_extension(primary, extension):
    known={(tuple(r['shape']),r['candidate_id']) for r in primary}
    added=[r for r in extension if (tuple(r['shape']),r['candidate_id']) not in known]
    if any(r['family'] not in ('host_merged_cutlass','host_merged_cublaslt') for r in added):
        raise ValueError('only missing merged-host candidates may be added')
    return added


def host_fold(rows, shape):
    train=[r for r in rows if r['family'] in HOSTS and r['shape']!=list(shape) and r['eligible_for_calibration']]
    model=MechanismModel.fit(train,variant='cutlass_depth')
    if any(list(s)==list(shape) for s in model.training_shapes):raise ValueError('target-shape leakage')
    return model,train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    start=time.monotonic();a.out.mkdir(parents=True)
    data,provenance=load_evidence();run=BASE/'run-02';seal=read(run/'analysis-seal.json')
    for name in ('heldout-predictions.json','metrics.json'):
        if sha(run/name)!=seal['outputs'][name]:raise ValueError('reference result hash mismatch')
    before=read(run/'heldout-predictions.json');added=unique_extension(data['primary'],data['extension'])
    expanded=data['primary']+added
    if len(added)!=8:raise ValueError('expected exactly eight previously unmeasured merged-host labels')
    forecasts={};folds=[];supports=[]
    for shape in sorted({tuple(r['shape']) for r in expanded}):
        model,train=host_fold(expanded,shape)
        folds.append({'shape':shape,'model':model.to_dict(),'training_config_ids':[r['config_id'] for r in train]})
        for label,targets in (('primary',data['primary']),('repeat',data['repeat']),('expanded',expanded)):
            previous={r['config_id']:r for r in before[f"{'primary' if label=='expanded' else label}/shape/regime_max"]}
            for row in targets:
                if tuple(row['shape'])!=shape:continue
                if row['family'] in HOSTS:
                    output=prediction(model,row,'development_host_coverage_whole_shape_leave_out')
                    vectors=[basis(r['shape'],r['family'],'cutlass_depth')[1] for r in train if r['family']==row['family']]
                    support=linear_target_support(vectors,basis(row['shape'],row['family'],'cutlass_depth')[1])
                    supports.append({'evaluation':label,'shape':shape,'candidate_id':row['candidate_id'],**support})
                else:
                    output=previous[row['config_id']]
                forecasts.setdefault(label,[]).append(output)
    acceptance=read(BASE/'design.json')['acceptance']
    results={label:score_scopes(rows,acceptance) for label,rows in forecasts.items()}
    selection={label:choices(rows) for label,rows in forecasts.items()}
    paired={label:{'before':read(run/'metrics.json')[f'{label}/shape/regime_max'], 'after':results[label]}
            for label in ('primary','repeat')}
    for label in ('primary','repeat'):
        ref={r['config_id']:r for r in before[f'{label}/shape/regime_max']}
        if set(ref)!={r['config_id'] for r in forecasts[label]}:raise ValueError('paired support changed')
        for row in forecasts[label]:
            if row['family'] not in HOSTS and row != ref[row['config_id']]:raise ValueError('fused predictions changed')
    for name,value in {'design':read(BASE/'host-coverage-design.json'),'provenance':provenance,'configurations':expanded,
        'added-configurations':added,'folds':folds,'heldout-predictions':forecasts,'metrics':results,
        'paired-comparison':paired,'selection':selection,'support':supports}.items():write(a.out/(name+'.json'),value)
    write(a.out/'decision.json',{'host_coverage_changed':True,'cost_expression_changed':False,'added_labels':8,
        'fused_predictions_unchanged':True,'independent_validation':False,'compiler_admission':False,
        'reason':'Coverage correction is useful development evidence, but does not resolve fused expression failures or validate automatic selection.'})
    source=[Path(__file__),BASE/'host-coverage-design.json',run/'analysis-seal.json',
            Path(__file__).with_name('post_gpu_evidence.py'),Path(__file__).with_name('evaluate_post_gpu_revision.py'),
            Path(__file__).with_name('evaluate_mechanism_prediction.py'),
            *[REPO/'EDM-IR/edmir/compiler'/f for f in ('mechanism_prediction.py','prediction_support.py','candidate_budget.py')]]
    write(a.out/'analysis-seal.json',{'sources':{str(f.relative_to(REPO)):sha(f) for f in source},
        'outputs':{f.name:sha(f) for f in a.out.iterdir() if f.is_file()},'elapsed_seconds':time.monotonic()-start})
    print(json.dumps({'added':len(added),'paired_primary_mape':results['primary']['stable']['macro_shape_mean_absolute_percentage_error'],
                      'paired_primary_max_ape':results['primary']['stable']['max_absolute_percentage_error'],
                      'elapsed_seconds':time.monotonic()-start},ensure_ascii=False))


if __name__=='__main__':main()
