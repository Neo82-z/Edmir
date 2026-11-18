#!/usr/bin/env python3
"""Score bounded candidate lists using sealed whole-shape-out predictions.

No refit, new GPU measurement, compiler selection, or change to old artifacts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.candidate_budget import rank_predictions, shortlist, score, training_only_order, uniform_random_reference

ROOT=Path(__file__).resolve().parents[2]
DESIGN=ROOT/'experiments/2026-10-25-bounded-candidate-selection/design.json'
OLD=ROOT/'experiments/2026-10-25-final-local-revision/results'
FINE=ROOT/'experiments/2026-10-25-fine-grained-prediction/final'
HOST='host_merged_cublaslt@merged'


def read(path):return json.loads(path.read_text())
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(folder):
    seal=read(folder/'analysis-seal.json')
    for rel,h in seal.get('outputs',seal.get('output_sha256',{})).items():
        if sha(folder/rel)!=h:raise ValueError('sealed evidence changed: '+rel)
    for rel,h in seal.get('sources',seal.get('source_sha256',{})).items():
        if sha(ROOT/rel)!=h:raise ValueError('sealed source changed: '+rel)


def evaluate():
    verify_seal(OLD);verify_seal(FINE)
    design=read(DESIGN);pred=read(OLD/'heldout-predictions.json')
    pred['row_service']=read(FINE/'heldout-predictions.json')['row_service']
    configs=[r for r in read(OLD/'configurations.json') if r['primary_scope']]
    stable=[r for r in configs if r['eligible_for_calibration']]
    if len({r['context_id'] for r in stable})!=1:raise ValueError('mixed execution contexts')
    estimates={v:{r['config_id']:r for r in pred[v]} for v in design['models']}
    for v,index in estimates.items():
        if len(index)!=len(pred[v]):raise ValueError('duplicate model row')
        for r in stable:
            z=index[r['config_id']]
            for key in ('shape','candidate_id','observed_us','quality_flags','eligible_for_calibration'):
                if z[key]!=r[key]:raise ValueError('model comparison data identity mismatch: '+v+'/'+key)
    groups=defaultdict(list)
    for r in stable:groups[tuple(r['shape'])].append(r)
    observations=[];skipped=[];folds=[]
    def add(shape,rows,policy,k,ids):
        if not ids or set(ids)-{r['candidate_id'] for r in rows}:
            skipped.append({'shape':shape,'policy':policy,'budget':k,'required_candidates':ids,'reason':'missing measured stable candidate; no replacement or imputation'})
            return
        observations.append({'shape':shape,'policy':policy,'budget':k,**score(rows,ids,near=design['near_optimal_loss_fraction'])})
    for shape,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['candidate_id'])
        trained=training_only_order([r for r in stable if tuple(r['shape'])!=shape],shape)
        folds.append({'held_out_shape':shape,**trained})
        for k in design['budgets']:
            for v in design['models']:
                ids=shortlist(rank_predictions([estimates[v][r['config_id']] for r in rows]),k)
                add(shape,rows,v,k,ids)
            ref=uniform_random_reference(rows,k,near=design['near_optimal_loss_fraction'])
            observations.append({'shape':shape,'policy':'uniform_random','budget':k,**ref})
            add(shape,rows,'training_only_fixed_order',k,trained['order'][:k])
            order=rank_predictions([estimates['regime_max'][r['config_id']] for r in rows])
            ids=shortlist(order,k,protected=HOST) if HOST in order else [HOST]
            add(shape,rows,'protected_host_plus_regime',k,ids)
        add(shape,rows,'always_splitk4',1,['splitk4@prepared'])
        add(shape,rows,'always_merged_cublaslt',1,[HOST])
    summaries=[]
    for policy,k in sorted({(r['policy'],r['budget']) for r in observations}):
        rs=[r for r in observations if r['policy']==policy and r['budget']==k]
        random=policy=='uniform_random'
        loss='expected_choice_loss_fraction' if random else 'choice_loss_fraction'
        retention='near_optimal_retention_probability' if random else 'near_optimal_retained'
        summaries.append({'policy':policy,'budget':k,'covered_shapes':len(rs),'total_shapes':len(groups),
            'shape_list':[r['shape'] for r in rs],
            'macro_mean_loss':statistics.fmean(r[loss] for r in rs),
            'max_shape_loss_or_max_shape_expected_loss':max(r[loss] for r in rs),
            'random_expectation_not_realized_worst_case':random,
            'macro_near_optimal_retention':statistics.fmean(r[retention] for r in rs),
            'macro_candidate_count_reduction':statistics.fmean(r['candidate_count_reduction'] for r in rs),
            'mean_candidates_evaluated':statistics.fmean(r['evaluated_candidates'] for r in rs)})
    # Compare baselines and primary only on exactly matched shape lists.
    paired=[]
    for summary in summaries:
        policy,k=summary['policy'],summary['budget']
        if policy=='regime_max':continue
        targets={tuple(s) for s in summary['shape_list']}
        rs=[r for r in observations if r['policy']=='regime_max' and r['budget']==k and tuple(r['shape']) in targets]
        paired.append({'other_policy':policy,'budget':k,'matched_shapes':sorted(targets),
            'primary_mean_loss':statistics.fmean(r['choice_loss_fraction'] for r in rs),
            'other_mean_loss_or_expected_loss':summary['macro_mean_loss'],
            'primary_max_loss':max(r['choice_loss_fraction'] for r in rs),
            'other_max_loss_or_max_expected_loss':summary['max_shape_loss_or_max_shape_expected_loss']})
    return {'schema':'edmir.candidate-budget-results.v1','design':design,'summary':summaries,
        'per_shape':observations,'matched_comparisons':paired,'training_only_baseline_folds':folds,
        'unavailable_policy_cases':skipped,
        'quality_excluded':[{'shape':r['shape'],'candidate_id':r['candidate_id'],'quality_flags':r['quality_flags']} for r in configs if not r['eligible_for_calibration']],
        'evaluation_scope':'Ideal shortlist replay on previously inspected development data; measured stable subsets vary by shape.',
        'calibration_and_tuning_wall_seconds':None,'new_fit':False,'GPU_runs':0,
        'deployment_policy_validated':False,'compiler_admission':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    start=time.monotonic();result=evaluate();a.out.mkdir(parents=True)
    write(a.out/'results.json',result)
    inputs=[DESIGN,OLD/'heldout-predictions.json',OLD/'configurations.json',OLD/'analysis-seal.json',
            FINE/'heldout-predictions.json',FINE/'analysis-seal.json',Path(__file__),
            ROOT/'EDM-IR/edmir/compiler/candidate_budget.py']
    write(a.out/'seal.json',{'inputs':{str(f.relative_to(ROOT)):sha(f) for f in inputs},
        'results_sha256':sha(a.out/'results.json'),'seconds':time.monotonic()-start,'new_fit':False,'GPU_runs':0})
    for r in result['summary']:
        print(r['policy'],r['budget'],r['covered_shapes'],f"mean={100*r['macro_mean_loss']:.3f}%",f"max={100*r['max_shape_loss_or_max_shape_expected_loss']:.3f}%")


if __name__=='__main__':main()
