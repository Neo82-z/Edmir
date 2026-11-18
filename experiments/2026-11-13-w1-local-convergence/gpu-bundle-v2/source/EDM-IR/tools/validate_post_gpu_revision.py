#!/usr/bin/env python3
"""Recalculate reported metrics directly, without the evaluation metric helpers."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post_gpu_evidence import REPO, read, sha, CAMPAIGN


def close(actual, expected):
    if not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9):
        raise AssertionError((actual,expected))


def verify(run, host_run=None):
    seal=read(run/'analysis-seal.json')
    for name,expected in seal['sources'].items():
        if sha(REPO/name)!=expected:raise ValueError('current analysis source mismatch: '+name)
    for name,expected in seal['outputs'].items():
        if sha(run/name)!=expected:raise ValueError('analysis output mismatch: '+name)
    predictions=read(run/'heldout-predictions.json');metrics=read(run/'metrics.json')
    metric_checks=0
    for tag,rows in predictions.items():
        for quality in ('all_complete','stable'):
            selected=[r for r in rows if quality=='all_complete' or not r['quality_flags']]
            groups=defaultdict(list)
            for r in selected:
                groups[tuple(r['shape'])].append(r)
                close(r['absolute_percentage_error'],abs(r['predicted_us']/r['observed_us']-1))
                close(r['predicted_us'],math.fsum(r['contributions_us'].values()))
            got=metrics[tag][quality]
            close(got['macro_shape_mean_absolute_percentage_error'],st.mean(st.mean(abs(r['predicted_us']/r['observed_us']-1) for r in g) for g in groups.values()))
            close(got['max_absolute_percentage_error'],max(abs(r['predicted_us']/r['observed_us']-1) for r in selected))
            wrong=severe=count=0
            losses=defaultdict(list)
            for group in groups.values():
                for a,b in combinations(group,2):
                    count+=1
                    pred=min((a,b),key=lambda r:(r['predicted_us'],r['candidate_id']))
                    obs=min((a,b),key=lambda r:(r['observed_us'],r['candidate_id']))
                    wrong+=pred['candidate_id']!=obs['candidate_id']
                    severe+=pred['observed_us']/obs['observed_us']-1>.1
                for scope in ('all_measured_stable','compiler_host_vs_fused','fused_only'):
                    if scope=='compiler_host_vs_fused' and not any(r['family']=='host' for r in group):continue
                    sub=[r for r in group if scope=='all_measured_stable' or not r['family'].startswith('host') or (scope=='compiler_host_vs_fused' and r['family']=='host')]
                    if len(sub)<2:continue
                    winner=min(sub,key=lambda r:(r['predicted_us'],r['candidate_id']))
                    losses[scope].append(winner['observed_us']/min(r['observed_us'] for r in sub)-1)
            assert (got['wrong_pair_directions'],got['wrong_pairs_over_10_percent_loss'],got['pairs'])==(wrong,severe,count)
            for scope,values in losses.items():
                close(got['decision_metrics'][scope]['mean_choice_loss_fraction'],st.mean(values))
                close(got['decision_metrics'][scope]['max_choice_loss_fraction'],max(values))
            metric_checks+=1
    folds=list((run/'folds').rglob('*.json'))
    for path in folds:
        f=read(path);axis=f['axis'];h=f['held_out']
        group=lambda shape:shape if axis=='shape' else shape[1 if axis=='M' else 3]
        assert all(group(s)!=h for s in f['model']['training_shapes'])
        assert all(group(r['shape'])==h for r in f['predictions']+f['repeat_predictions'])
        assert not set(f['model']['training_config_ids'])&{r['config_id'] for r in f['predictions']}
    for tag,rows in predictions.items():
        if not tag.endswith('/retained_0'):continue
        controls=predictions[tag.removesuffix('retained_0')+'regime_max']
        assert [r['config_id'] for r in rows]==[r['config_id'] for r in controls]
        for a,b in zip(rows,controls):close(a['predicted_us'],b['predicted_us'])
    choices=read(run/'selection.json');choice_checks=0
    for tag,reports in choices.items():
        for report in reports:
            times={r['candidate_id']:r['observed_us'] for r in predictions[tag] if r['shape']==report['shape'] and not r['quality_flags']}
            for policy,value in report['policies'].items():
                if policy=='random_top2':continue
                selected=value['selected_candidates']
                close(value['choice_loss_fraction'],min(times[c] for c in selected)/min(times.values())-1)
                assert value['evaluated_candidates']==len(selected)
                if policy=='two_merged_hosts':assert len(selected)==2
                choice_checks+=1
    campaign=read(CAMPAIGN/'analysis-seal.json')
    for name,expected in campaign['sources_and_outputs'].items():
        if sha(CAMPAIGN/name)!=expected:raise ValueError('original GPU campaign was modified: '+name)
    host_checks=0
    if host_run is not None:
        host_seal=read(host_run/'analysis-seal.json')
        for name,expected in host_seal['sources'].items():assert sha(REPO/name)==expected,name
        for name,expected in host_seal['outputs'].items():assert sha(host_run/name)==expected,name
        host_predictions=read(host_run/'heldout-predictions.json');host_metrics=read(host_run/'metrics.json')
        assert len(read(host_run/'added-configurations.json'))==8
        assert len(read(host_run/'configurations.json'))==65
        for fold in read(host_run/'folds.json'):
            assert all(s!=fold['shape'] for s in fold['model']['training_shapes'])
        for label,rows in host_predictions.items():
            if label in ('primary','repeat'):
                prior={r['config_id']:r for r in predictions[f'{label}/shape/regime_max']}
                assert set(prior)=={r['config_id'] for r in rows}
                for row in rows:
                    if not row['family'].startswith('host'):assert row==prior[row['config_id']]
            for quality in ('all_complete','stable'):
                selected=[r for r in rows if quality=='all_complete' or not r['quality_flags']]
                grouped=defaultdict(list)
                for row in selected:grouped[tuple(row['shape'])].append(row)
                result=host_metrics[label][quality]
                close(result['macro_shape_mean_absolute_percentage_error'],st.mean(st.mean(abs(r['predicted_us']/r['observed_us']-1) for r in g) for g in grouped.values()))
                close(result['max_absolute_percentage_error'],max(abs(r['predicted_us']/r['observed_us']-1) for r in selected))
                wrong=severe=total=0;losses=[]
                for group in grouped.values():
                    if len(group)>1:
                        chosen=min(group,key=lambda r:(r['predicted_us'],r['candidate_id']))
                        losses.append(chosen['observed_us']/min(r['observed_us'] for r in group)-1)
                    for a,b in combinations(group,2):
                        total+=1
                        pred=min((a,b),key=lambda r:(r['predicted_us'],r['candidate_id']))
                        obs=min((a,b),key=lambda r:(r['observed_us'],r['candidate_id']))
                        wrong+=pred['candidate_id']!=obs['candidate_id']
                        severe+=pred['observed_us']/obs['observed_us']-1>.1
                assert (wrong,severe,total)==(result['wrong_pair_directions'],result['wrong_pairs_over_10_percent_loss'],result['pairs'])
                close(result['decision_metrics']['all_measured_stable']['mean_choice_loss_fraction'],st.mean(losses))
                close(result['decision_metrics']['all_measured_stable']['max_choice_loss_fraction'],max(losses))
                host_checks+=1
    return {'passed':True,'metric_scope_checks':metric_checks,'heldout_folds':len(folds),
            'choice_checks':choice_checks,'original_campaign_files_unchanged':len(campaign['sources_and_outputs']),
            'current_source_hashes_match':True,'rho_zero_matches_previous_max':True,
            'host_coverage_metric_scopes_checked':host_checks,'fused_predictions_unchanged_in_host_revision':host_run is not None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--host-run',type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('fresh output file required')
    result=verify(a.run,a.host_run)
    result['validator_sha256']=sha(Path(__file__))
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
