#!/usr/bin/env python3
"""Historical expression comparison, NOT transfer of the current P4 model.

Keep the old protocol and candidate mask explicit. Numerical solvers evaluate
the expressions after refitting within each legacy sweep fold; production
context/warmup/candidate admission checks are not changed or bypassed for use.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics as st
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import load_selector_dataset
from edmir.compiler.two_stage_selection import TwoStageSelector
from edmir.compiler.mechanism_prediction import basis, nonnegative_fit, geometry
from edmir.compiler.regime_prediction import regime_basis, max_branch_fit
from edmir.compiler.metric_evidence import metric_evidence
from post_gpu_evidence import REPO, read, sha
from evaluate_post_gpu_revision import write

BASE=REPO/'experiments/2026-10-25-initial-data-replay'
DATA=REPO/'experiments/2026-10-20-offline-predictor/data'


def refit_expression(rows, rho):
    models={}
    for host in (True,False):
        rs=[r for r in rows if r.is_host==host]
        counts=Counter(r.shape for r in rs)
        vectors=[basis(r.shape,'host','cutlass_depth')[1] if host else regime_basis(r.shape,r.family)[1] for r in rs]
        weights=[1/counts[r.shape] for r in rs]
        fit=(nonnegative_fit(vectors,[1000*r.measured_ms for r in rs],weights) if host else
             max_branch_fit(vectors,[1000*r.measured_ms for r in rs],weights,retained_fraction=rho))
        models['host' if host else 'fused']=fit
    return models


def expression_predict(model,row,rho):
    vector=basis(row.shape,'host','cutlass_depth')[1] if row.is_host else regime_basis(row.shape,row.family)[1]
    terms=[v*c for v,c in zip(vector,model['host' if row.is_host else 'fused']['coefficients_us'])]
    return math.fsum(terms) if row.is_host else math.fsum(terms[:-2])+max(terms[-2:])+rho*min(terms[-2:])


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r['shape'])].append(r)
    decisions=[];wrong=0;pairs=0
    for shape,group in sorted(groups.items()):
        ranked=sorted(group,key=lambda r:(r['predicted_us'],r['family']))
        best=min(group,key=lambda r:(r['observed_us'],r['family']))
        for k in (1,2,3):
            chosen=min(ranked[:k],key=lambda r:(r['observed_us'],r['family']))
            decisions.append({'shape':shape,'budget':k,'candidates':len(group),'shortlist':[r['family'] for r in ranked[:k]],
                              'winner':chosen['family'],'measured_subset_best':best['family'],
                              'loss':chosen['observed_us']/best['observed_us']-1})
        for a,b in combinations(group,2):
            pairs+=1
            wrong+=min((a,b),key=lambda r:(r['predicted_us'],r['family']))['family']!=min((a,b),key=lambda r:(r['observed_us'],r['family']))['family']
    return {'configurations':len(rows),'shapes':len(groups),
        'macro_shape_MAPE':st.mean(st.mean(abs(r['predicted_us']/r['observed_us']-1) for r in g) for g in groups.values()),
        'max_APE':max(abs(r['predicted_us']/r['observed_us']-1) for r in rows),
        'wrong_pairs':wrong,'pairs':pairs,
        'selection':{str(k):{'mean_loss':st.mean(d['loss'] for d in decisions if d['budget']==k),
                            'max_loss':max(d['loss'] for d in decisions if d['budget']==k),
                            'exact_winners':sum(d['loss']==0 for d in decisions if d['budget']==k)} for k in (1,2,3)},
        'decisions':decisions}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    start=time.monotonic();design=read(BASE/'design.json')
    p4,cal,p8,manifest=load_selector_dataset(DATA)
    allowed=set(design['candidate_support']);common=[r for r in p4 if r.family in allowed]
    groups=defaultdict(list)
    for r in p4:groups[r.shape].append(r)
    exclusions=[]
    for shape,rs in sorted(groups.items()):
        full=min(rs,key=lambda r:(r.measured_ms,r.family))
        sub=min((r for r in rs if r.family in allowed),key=lambda r:(r.measured_ms,r.family))
        exclusions.append({'shape':shape,'full_best':full.family,'supported_best':sub.family,
                           'full_best_excluded':full.family not in allowed,
                           'loss_from_candidate_exclusion':sub.measured_ms/full.measured_ms-1,
                           'excluded_families':[r.family for r in rs if r.family not in allowed]})
    a.out.mkdir(parents=True)
    forecasts={v:[] for v in design['versions']};folds=[]
    for group in sorted({r.group_id for r in common}):
        train=[r for r in common if r.group_id!=group];test=[r for r in common if r.group_id==group]
        assert not {r.shape for r in train}&{r.shape for r in test}
        original=TwoStageSelector.fit(train,ablation='full',ridge=1.)
        models={'original_full_log':original.to_dict()}
        for row in test:
            ms=(original.host_model if row.is_host else original.fused_model).predict_ms(row)
            forecasts['original_full_log'].append({'shape':row.shape,'group':group,'family':row.family,
                'predicted_us':1000*ms,'observed_us':1000*row.measured_ms})
        for variant,rho in (('regime_max_refit',0.),('retained_0.25_refit',.25)):
            fitted=refit_expression(train,rho);models[variant]=fitted
            for row in test:
                forecasts[variant].append({'shape':row.shape,'group':group,'family':row.family,
                    'predicted_us':expression_predict(fitted,row,rho),'observed_us':1000*row.measured_ms})
        folds.append({'excluded_group':group,'training_shapes':sorted({r.shape for r in train}),
                      'evaluation_shapes':sorted({r.shape for r in test}),'models':models})
        write(a.out/'progress.json',{'folds_done':len(folds),'total_folds':9})
    metric_checks=[metric_evidence(row.shape,row.family) for row in common]
    p8_refusal=None
    try:geometry((8,64,4096,512),'splitk2')
    except ValueError as exc:p8_refusal=str(exc)
    if p8_refusal is None:raise ValueError('P8 unexpectedly admitted')
    scores={k:summarize(v) for k,v in forecasts.items()}
    result={'scope':'historical P4 expression refit, not frozen model transfer',
        'original_full_counts':manifest['counts'],'common_candidates':len(common),'common_shapes':len(groups),
        'original_P4_rows':len(p4),'excluded_rows':len(p4)-len(common),
        'full_best_excluded_shapes':sum(r['full_best_excluded'] for r in exclusions),
        'oracle_loss_from_exclusion_mean':st.mean(r['loss_from_candidate_exclusion'] for r in exclusions),
        'oracle_loss_from_exclusion_max':max(r['loss_from_candidate_exclusion'] for r in exclusions),
        'P8_current_model_refusal':p8_refusal,'scores':scores,'independent_validation':False,'compiler_admission':False}
    for name,value in {'result':result,'predictions':forecasts,'folds':folds,'candidate-exclusions':exclusions,
                       'metric-evidence':metric_checks,'configurations':[asdict(r) for r in common],'design':design}.items():write(a.out/(name+'.json'),value)
    inputs=[Path(__file__),BASE/'design.json',DATA/'manifest.json',DATA/'selector_measurements.jsonl',
            *[REPO/'EDM-IR/edmir/compiler'/f for f in ('two_stage_selection.py','measurement_dataset.py','regime_prediction.py','mechanism_prediction.py','metric_evidence.py')]]
    write(a.out/'analysis-seal.json',{'inputs':{str(f.relative_to(REPO)):sha(f) for f in inputs},
        'outputs':{f.name:sha(f) for f in a.out.iterdir() if f.is_file()},'elapsed_seconds':time.monotonic()-start,'GPU_runs':0})
    print(json.dumps({**{v:{k:r[k] for k in ('macro_shape_MAPE','selection','wrong_pairs','pairs')} for v,r in scores.items()},
                     'candidate_exclusions':result['full_best_excluded_shapes'],'elapsed_seconds':time.monotonic()-start},ensure_ascii=False))


if __name__=='__main__':main()
