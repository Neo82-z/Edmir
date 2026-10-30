#!/usr/bin/env python3
"""Read sealed fresh measurements; score frozen predictions without fitting."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / 'experiments/2026-10-25-final-local-revision/results'
HOST = 'host_merged_cublaslt@merged'
INPUTS = ROOT / 'experiments/2026-10-25-regime-gpu-validation/bundle/inputs.json'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sign(x):
    return (x > 0) - (x < 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('fresh output required')
    review = read(a.review)
    if review['cross_stage_context_count'] != 1:
        raise ValueError('different execution contexts require separate analyses')
    if any(not stage['complete'] for stage in review['stages']):
        raise ValueError('final quantitative analysis requires complete sealed stages')
    rows = [dict(c, stage=s['stage']) for s in review['stages'] for c in s['configurations']]
    old = {(tuple(r['shape']), r['candidate_id']): r for r in read(OLD/'configurations.json') if r['primary_scope']}
    input_ids = {tuple(r['shape']):r for r in read(INPUTS)['workloads'].values()}
    groups = defaultdict(list)
    for r in rows:
        if r['primary_timing_label'] and r['complete_blocks']:
            groups[(r['stage'], tuple(r['shape']))].append(r)
    comparisons = []
    for r in rows:
        o = old.get((tuple(r['shape']), r['candidate_id']))
        if o and r['complete_blocks'] and r['primary_timing_label']:
            comparisons.append({'stage':r['stage'], 'shape':r['shape'], 'candidate_id':r['candidate_id'],
                'old_us':o['observed_us'], 'new_us':r['observed_us'], 'new_over_old':r['observed_us']/o['observed_us'],
                'old_quality_flags':o['quality_flags'], 'new_quality_flags':r['quality_flags'],
                'input_and_reference_unchanged':all(o[k]==input_ids[tuple(r['shape'])][k] for k in ('input_sha256','reference_sha256')),
                'algorithm_unchanged':o['algorithm_ids']==r['algorithm_ids'],
                'old_algorithm_ids':o['algorithm_ids'], 'new_algorithm_ids':r['algorithm_ids']})
    summaries=[]; pairs=[]; decisions=[]
    for variant in ('regime_max','residual_depth'):
        for scope in ('all_complete','stable'):
            errs=[]; macro=[]; scopes=[]; vp=[]
            for (stage,shape),rs in sorted(groups.items()):
                rs=[r for r in rs if scope=='all_complete' or not r['quality_flags']]
                if not rs: continue
                es=[r['forecast_errors'][variant]['absolute_percentage_error'] for r in rs]
                errs.extend(es); macro.append(st.fmean(es))
                for x,y in combinations(sorted(rs,key=lambda r:r['candidate_id']),2):
                    observed=y['observed_us']-x['observed_us']
                    predicted=y['forecast_errors'][variant]['predicted_us']-x['forecast_errors'][variant]['predicted_us']
                    deltas=[b-a for a,b in zip(x['process_medians_us'],y['process_medians_us'])]
                    wrong=sign(observed)*sign(predicted)<0
                    v={'model':variant,'scope':scope,'stage':stage,'shape':shape,'a':x['candidate_id'],'b':y['candidate_id'],
                       'observed_b_minus_a_us':observed,'predicted_b_minus_a_us':predicted,'wrong_direction':wrong,
                       'prediction_exact_tie':predicted==0,'process_deltas_us':deltas,
                       'all_process_directions_agree':len({sign(d) for d in deltas})==1,
                       'wrong_direction_relative_loss':abs(observed)/min(x['observed_us'],y['observed_us']) if wrong else 0}
                    vp.append(v)
                for selection,subset in [('all',rs),('fused',[r for r in rs if not r['family'].startswith('host')]),
                        ('compiler_host_fused',[r for r in rs if r['family']=='host' or not r['family'].startswith('host')])]:
                    if len(subset)<2: continue
                    ranked=sorted(subset,key=lambda r:(r['forecast_errors'][variant]['predicted_us'],r['candidate_id']))
                    best=min(subset,key=lambda r:(r['observed_us'],r['candidate_id']))
                    for k in (1,2,3):
                        short=ranked[:k]
                        chosen=min(short,key=lambda r:(r['observed_us'],r['candidate_id']))
                        item={'model':variant,'scope':scope,'stage':stage,'shape':shape,'selection_scope':selection,'budget':k,
                              'ranked_shortlist':[r['candidate_id'] for r in short], 'actual_count':len(short),
                              'subset_size':len(subset),'measured_subset_best':best['candidate_id'],
                              'choice_loss':chosen['observed_us']/best['observed_us']-1,
                              'ideal_shortlist_replay':k>1,'full_candidate_oracle':False}
                        scopes.append(item)
            pairs.extend(vp); decisions.extend(scopes)
            if errs:
                summaries.append({'model':variant,'scope':scope,'configurations':len(errs),'shape_groups':len(macro),
                                  'macro_shape_mape':st.fmean(macro),'max_ape':max(errs),
                                  'wrong_pairs':sum(r['wrong_direction'] for r in vp),'total_pairs':len(vp),
                                  'wrong_pairs_with_consistent_process_direction':sum(r['wrong_direction'] and r['all_process_directions_agree'] for r in vp),
                                  'wrong_pairs_above_10pct_loss':sum(r['wrong_direction_relative_loss']>.10 for r in vp)})
    choice_summary=[]
    for key in sorted({(r['model'],r['scope'],r['selection_scope'],r['budget']) for r in decisions}):
        rs=[r for r in decisions if (r['model'],r['scope'],r['selection_scope'],r['budget'])==key]
        choice_summary.append(dict(zip(('model','scope','selection_scope','budget'),key),
            shape_groups=len(rs),mean_loss=st.fmean(r['choice_loss'] for r in rs),max_loss=max(r['choice_loss'] for r in rs)))
    anchors={(tuple(r['shape']),r['candidate_id']):r for r in rows if r['stage']=='anchors' and r['complete_blocks']}
    old_errors=[]
    for o in read(OLD/'pair-audit.json')['regime_max']:
        if not o['wrong_direction']: continue
        x=anchors.get((tuple(o['shape']),o['a'])); y=anchors.get((tuple(o['shape']),o['b']))
        z={'shape':o['shape'],'a':o['a'],'b':o['b'],'old_observed_delta_us':o['observed_b_minus_a_us'],
           'old_heldout_predicted_delta_us':o['predicted_b_minus_a_us'],'complete':bool(x and y)}
        if x and y:
            d=y['observed_us']-x['observed_us']; pred=y['forecast_errors']['regime_max']['predicted_us']-x['forecast_errors']['regime_max']['predicted_us']
            z.update(new_observed_delta_us=d,new_frozen_predicted_delta_us=pred,
                new_process_deltas_us=[b-a for a,b in zip(x['process_medians_us'],y['process_medians_us'])],
                old_measured_direction_repeated=sign(d)==sign(o['observed_b_minus_a_us']),
                new_frozen_direction_wrong=sign(d)*sign(pred)<0,quality_flags={x['candidate_id']:x['quality_flags'],y['candidate_id']:y['quality_flags']})
        old_errors.append(z)
    host_comparison=[]
    for (stage,shape),rs in sorted(groups.items()):
        rs=[r for r in rs if not r['quality_flags']]
        host=next((r for r in rs if r['candidate_id']==HOST),None)
        if not host: continue
        best=min(rs,key=lambda r:(r['observed_us'],r['candidate_id']))
        ranked=sorted(rs,key=lambda r:(r['forecast_errors']['regime_max']['predicted_us'],r['candidate_id']))
        host_comparison.append({'stage':stage,'shape':shape,'candidates':len(rs),'best':best['candidate_id'],
            'fixed_merged_cublaslt_loss':host['observed_us']/best['observed_us']-1,
            'regime_top1_loss':ranked[0]['observed_us']/best['observed_us']-1,
            'regime_top2_ideal_loss':min(r['observed_us'] for r in ranked[:2])/best['observed_us']-1})
    warmup=[]
    for r in rows:
        if r['stage']!='warmup' or not r['complete_blocks']:continue
        before=anchors.get((tuple(r['shape']),r['candidate_id']))
        if before: warmup.append({'shape':r['shape'],'candidate_id':r['candidate_id'],'warmup20_us':before['observed_us'],
            'warmup200_us':r['observed_us'],'ratio_200_over_20':r['observed_us']/before['observed_us'],
            'warmup20_flags':before['quality_flags'],'warmup200_flags':r['quality_flags'],
            'causal_warmup_effect_identified':False})
    output={'schema':'edmir.hourly-frozen-validation.v1','new_fit':False,'compiler_admission':False,
        'summary':summaries,'choice_summary':choice_summary,'old_new_comparisons':comparisons,
        'original_nine_pairs':old_errors,'pairs':pairs,'decisions':decisions,'strong_host_matched_comparisons':host_comparison,
        'warmup_comparisons':warmup,'quality_flagged':[r for r in rows if r['quality_flags']],
        'limitations':['New environment; no pooling or rescaling against old measurements.',
            'Targeted shapes selected using previous results; not independent generalization validation.',
            'Frozen full-training forecasts and old leave-shape-out forecasts are different evaluations.',
            'Shortlist scores are ideal replay within measured candidates; not actual noisy selection or time savings.'],
        'inputs_sha256':{str(a.review):sha(a.review),str(OLD/'configurations.json'):sha(OLD/'configurations.json'),
            str(OLD/'pair-audit.json'):sha(OLD/'pair-audit.json'),str(INPUTS):sha(INPUTS),str(Path(__file__)):sha(Path(__file__))}}
    a.out.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'summary':summaries,'choice_summary':choice_summary,'old_errors_retested':len(old_errors),
                      'flagged':len(output['quality_flagged'])},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
