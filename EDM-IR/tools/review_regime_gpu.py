#!/usr/bin/env python3
"""Validate a fresh collection against frozen predictions, without fitting."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


def read(path):return json.loads(path.read_text())


def review(package,work):
    package=package.resolve();work=work.resolve()
    sys.path[:0]=[str(package/'source/EDM-IR'),str(package/'source/EDM-IR/tools')]
    from prepare_mechanism_gpu import verify_bundle,STAGES
    from run_mechanism_gpu import check_row
    from independent_selector_validation import digest
    p=verify_bundle(package);inputs=read(package/'inputs.json')['workloads']
    forecast=read(package.parent/'frozen-predictions.json')
    predictions={(tuple(r['shape']),r['candidate_id'],r['stage']):r for r in forecast['rows']}
    files=list(work.glob('*/measurements.jsonl'));results=[];contexts={};stages=[]
    for file in sorted(files):
        folder=file.parent;ctx=read(folder/'context.json');stage=ctx['stage']
        if stage not in STAGES or ctx['package_sha256']!=digest(p):raise ValueError('collection/package identity mismatch')
        if stage in stages:raise ValueError('multiple runs of the same stage: choose an explicit work tree, never silently pool')
        stages.append(stage);contexts[stage]=ctx
        jobs=[j for j in p['jobs'] if j['stage']==stage]
        raw=[json.loads(s) for s in file.read_text().splitlines()]
        if len(raw)>len(jobs):raise ValueError('extra rows')
        raw_hash=hashlib.sha256(file.read_bytes()).hexdigest()
        sealed=(folder/'run.json').exists()
        if sealed:
            seal=read(folder/'run.json')
            if seal['context']!=ctx or seal['raw_sha256']!=raw_hash or seal['rows']!=raw or len(raw)!=len(jobs):
                raise ValueError('final run seal mismatch')
        grouped=defaultdict(list)
        for i,(r,j) in enumerate(zip(raw,jobs)):
            info=inputs[j['workload_id']]
            if check_row(r,j,info,p)!=r or check_row(read(folder/f'stdout_{i:05d}.txt'),j,info,p)!=r:
                raise ValueError('raw row/stdout/protocol mismatch')
            if r['runtime']!=ctx['runtime']:raise ValueError('runtime identity changed')
            grouped[(tuple(r['shape']),r['candidate_id'])].append(r)
        configs=[]
        for (shape,candidate),rows in sorted(grouped.items()):
            rows.sort(key=lambda r:r['repetition'])
            complete=[r['repetition'] for r in rows]==[0,1,2]
            medians=[r['median_ms']*1000 for r in rows]
            flags=[]
            if not complete:flags.append('incomplete_process_blocks')
            if max(medians)/min(medians)>1.10:flags.append('between_process_spread')
            if any(not .9<=statistics.median(r['samples_ms'][-20:])/statistics.median(r['samples_ms'][:20])<=1.1 for r in rows):flags.append('within_process_drift')
            algorithms=sorted({r['gemm_algorithm_id'] for r in rows})
            if len(algorithms)!=1:flags.append('algorithm_variation')
            observed=statistics.median(medians) if complete else None
            f=predictions[(shape,candidate,stage)]
            configs.append({'shape':shape,'candidate_id':candidate,'family':rows[0]['family'],
                'process_medians_us':medians,'observed_us':observed,'quality_flags':flags,'algorithm_ids':algorithms,
                'warmup':rows[0]['warmup'],'complete_blocks':complete,
                'forecast_errors':{v:{'predicted_us':z['predicted_us'],'error_us':z['predicted_us']-observed,
                    'absolute_percentage_error':abs(z['predicted_us']/observed-1)} for v,z in (f['conditional_predictions'] or {}).items()} if complete else {},
                'primary_timing_label':stage!='warmup','new_context_not_pooled_with_old':True})
        decisions=[]
        for shape in sorted({tuple(r['shape']) for r in configs}):
            good=[r for r in configs if tuple(r['shape'])==shape and not r['quality_flags'] and r['primary_timing_label']]
            if len(good)<2:continue
            actual=min(good,key=lambda r:(r['observed_us'],r['candidate_id']))
            for variant in ('regime_max','residual_depth'):
                chosen=min(good,key=lambda r:(r['forecast_errors'][variant]['predicted_us'],r['candidate_id']))
                decisions.append({'shape':shape,'model':variant,'chosen':chosen['candidate_id'],
                    'measured_subset_best':actual['candidate_id'],
                    'choice_loss_fraction':chosen['observed_us']/actual['observed_us']-1,
                    'missing_or_quality_excluded':sorted({j['candidate_id'] for j in jobs if tuple(j['shape'])==shape}-{r['candidate_id'] for r in good}),
                    'full_candidate_oracle':False})
        results.append({'stage':stage,'completed_processes':len(raw),'planned_processes':len(jobs),
            'complete':len(raw)==len(jobs) and sealed,'all_processes_saved':len(raw)==len(jobs),'finalized_seal_present':sealed,'raw_sha256':raw_hash,
            'configurations':configs,'conditional_forecast_decisions':decisions})
    if not files:raise ValueError('no saved measurements found')
    # A changed build/device context across stages is exposed, never averaged.
    context_groups={digest({k:v for k,v in c.items() if k!='stage'}) for c in contexts.values()}
    return {'schema':'edmir.fresh-regime-review.v1','stages':results,'contexts':contexts,
        'cross_stage_context_count':len(context_groups),'fit_performed':False,'compiler_admission':False,
        'scope':'Frozen conditional forecasts scored on fresh hardware measurements; source/environment comparability and algorithm changes still require review. Warmup diagnostics remain separate. No automatic pooling or refitting.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent/'bundle')
    p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('choose a new review filename')
    output=review(a.package,a.work)
    a.out.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'review':str(a.out),'stages':[{k:r[k] for k in ('stage','completed_processes','complete')} for r in output['stages']]}))


if __name__=='__main__':main()
