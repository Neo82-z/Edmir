#!/usr/bin/env python3
"""Separate, bounded coverage extension using already frozen host executables."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import statistics
import sys
import time

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('check','collect','review'))
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('baseline-extension-plan.json'))
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--report',type=Path)
    a=ap.parse_args(); root=a.root.resolve();package=root/'bundle'
    sys.path[:0]=[str(package/'source/EDM-IR'),str(package/'source/EDM-IR/tools')]
    import run_mechanism_gpu as runner
    from prepare_mechanism_gpu import verify_bundle
    p=verify_bundle(package); plan=read(a.plan); inputs=read(package/'inputs.json')['workloads']
    assert plan['source_plan_sha256']==sha(package/'plan.json')
    assert len(plan['jobs'])==36
    assert plan['warmup']==p['warmup']==20 and plan['iterations']==p['iterations']==100
    assert plan['timing_scope']==p['timing_scope'] and plan['measurement_policy']==p['measurement_policy']
    allowed={'w002','w003','w004','w006'}
    families={'host','host_merged_cutlass','host_merged_cublaslt'}
    expected={(w,f,r) for w in allowed for f in families for r in range(3)}
    assert {(j['workload_id'],j['family'],j['repetition']) for j in plan['jobs']}==expected
    for i,j in enumerate(plan['jobs']):
        assert j['ordinal']==j['trial_order']==i and j['stage']=='baseline_extension'
        program=p['programs'][j['executable']]
        assert program['shape']==inputs[j['workload_id']]['shape']==j['shape']
        assert program['compiled_family']=='host'
        assert j['implementation_id']==runner.base.implementation_id(j['family'],j['metadata_mode'])
        assert j['candidate_id']==j['family']+'@'+j['metadata_mode']
    identity={'plan_sha256':sha(a.plan),'collector_sha256':sha(Path(__file__))}
    if a.action=='check':
        print(json.dumps({'validated_jobs':36,'uses_frozen_executables':True,**identity}));return
    out=a.out.resolve()
    if a.action=='collect':
        if out.exists():raise ValueError('fresh output required')
        _,context,environment=runner.preflight(package,root/'work/build')
        out.mkdir(parents=True)
        write(out/'plan.json',plan);write(out/'context.json',context)
        write(out/'identity.json',identity)
        raw=out/'measurements.jsonl';rows=[]
        deadline=datetime(2026,10,25,6,29,10,tzinfo=timezone.utc).timestamp()
        for i,j in enumerate(plan['jobs']):
            if deadline-time.time()<180:
                write(out/'progress.json',{'state':'budget_exhausted','completed':i,'total':36})
                raise SystemExit(75)
            if i%12==0: runner.telemetry(out,f'b{j["repetition"]}_before_{i:05d}')
            output=runner.base.execute_job(runner.measurement_args(package,root/'work/build',p,j),j,out,i,timeout=120)
            row=runner.check_row(json.loads(output),j,inputs[j['workload_id']],p)
            if row['runtime']!=context['runtime']:raise ValueError('runtime changed')
            with raw.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row)
            write(out/'progress.json',{'state':'collecting','completed':i+1,'total':36})
            print(f'baseline_extension {i+1}/36 {j["workload_id"]} {j["candidate_id"]}',flush=True)
        runner.telemetry(out,'complete')
        write(out/'run.json',{'context':context,'environment':environment,'identity':identity,'rows':rows,'raw_sha256':sha(raw),
                              'adaptive_extension':True,'independent_generalization':False,'fit':False})
        write(out/'progress.json',{'state':'complete','completed':36,'total':36})
    elif a.action=='review':
        if not a.report or a.report.exists():raise ValueError('fresh report path required')
        seal=read(out/'run.json');raw=out/'measurements.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()]
        assert read(out/'plan.json')==plan and read(out/'identity.json')==identity==seal['identity']
        assert len(rows)==36 and seal['raw_sha256']==sha(raw) and seal['rows']==rows
        assert seal['context']==read(out/'context.json')
        grouped=defaultdict(list)
        for i,(row,job) in enumerate(zip(rows,plan['jobs'])):
            info=inputs[job['workload_id']]
            assert runner.check_row(row,job,info,p)==row
            assert runner.check_row(read(out/f'stdout_{i:05d}.txt'),job,info,p)==row
            assert row['runtime']==seal['context']['runtime']
            grouped[(tuple(row['shape']),row['candidate_id'])].append(row)
        configs=[];pairs=[]
        for (shape,cid),rs in sorted(grouped.items()):
            rs.sort(key=lambda r:r['repetition']);assert [r['repetition'] for r in rs]==[0,1,2]
            med=[r['median_ms']*1000 for r in rs];flags=[]
            if max(med)/min(med)>1.1:flags.append('between_process_spread')
            if any(not .9<=statistics.median(r['samples_ms'][-20:])/statistics.median(r['samples_ms'][:20])<=1.1 for r in rs):flags.append('within_process_drift')
            algorithms=sorted({r['gemm_algorithm_id'] for r in rs})
            if len(algorithms)>1:flags.append('algorithm_variation')
            configs.append({'shape':shape,'candidate_id':cid,'observed_us':statistics.median(med),
                'process_medians_us':med,'quality_flags':flags,'algorithm_ids':algorithms})
        for shape in sorted({tuple(r['shape']) for r in configs}):
            for x,y in combinations([r for r in configs if tuple(r['shape'])==shape],2):
                pairs.append({'shape':shape,'a':x['candidate_id'],'b':y['candidate_id'],
                    'b_minus_a_us':[b-a for a,b in zip(x['process_medians_us'],y['process_medians_us'])]})
        write(a.report,{'complete':True,'processes':36,'samples':3600,'configurations':configs,'pairs':pairs,
            'context':seal['context'],'identity':identity,'raw_sha256':sha(raw),'fit':False,'adaptive_extension':True,
            'no_frozen_prediction_score_for_added_candidates':True,'full_candidate_oracle':False})
        print(json.dumps({'complete':True,'configurations':len(configs),'flagged':sum(bool(r['quality_flags']) for r in configs)}))

if __name__=='__main__':main()
