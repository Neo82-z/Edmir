#!/usr/bin/env python3
"""Independent verification of saved replay labels, choices, gates and seals."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'metric mismatch {a} != {b}')


def validate(folder):
    seal=read(folder/'seal.json')
    for name,value in seal['inputs'].items():
        if digest(REPO/name)!=value:raise ValueError('input changed: '+name)
    for name,value in seal['outputs'].items():
        if digest(folder/name)!=value:raise ValueError('output changed: '+name)
    raw=[json.loads(x) for x in (REPO/'experiments/2026-10-20-offline-predictor/data/selector_measurements.jsonl').read_text().splitlines()]
    expected=defaultdict(dict)
    for r in raw:
        m=r['measurement'];shape=(m['world_size'],m['rows_per_peer'],m['k'],m['n'])
        ms=r['scoring_override']['median_ms'] if r.get('scoring_override') else m['measured_ms']
        expected[r['historical_role']][(shape,m['family'])]=1000*ms
    predictions=read(folder/'predictions.json');reports=read(folder/'result.json')['reports']
    models=read(folder/'models.json'); design=read(folder/'design.json')
    configs=decisions=pairs=0
    for method,protocols in predictions.items():
        for protocol,rows in protocols.items():
            wanted=expected['development' if protocol=='p4_group_heldout' else 'heldout']
            if len(rows)!=len(wanted) or len({(tuple(r['shape']),r['family']) for r in rows})!=len(rows):
                raise ValueError('duplicate or missing evaluation candidate')
            grouped=defaultdict(list)
            for row in rows:
                close(row['observed_us'],wanted[(tuple(row['shape']),row['family'])])
                grouped[tuple(row['shape'])].append(row);configs+=1
                if method.startswith('scale_'):
                    if row['compiler_admission']:raise ValueError('unvalidated promotion')
                    terms=list(row['potential_terms_us'].values())
                    rho=0. if method=='scale_max' else .25
                    total=(sum(terms) if row['family']=='host' else sum(terms[:-2])+max(terms[-2:])+rho*min(terms[-2:]))
                    close(total,row['predicted_us'])
            stat=reports[method][protocol];losses={k:[] for k in (1,2,3)}
            wrong=dir_correct=regressions=0;errors=[]
            for shape,rs in grouped.items():
                ordered=sorted(rs,key=lambda x:(x['predicted_us'],x['family']))
                best=min(rs,key=lambda x:(x['observed_us'],x['family']))
                host=next(x for x in rs if x['family']=='host')
                for k in losses:
                    actual=min(x['observed_us'] for x in ordered[:k])/best['observed_us']-1
                    losses[k].append(actual)
                    if method in ('host','train_best_fixed') and k>1:continue
                    d=next(x for x in stat['decisions'] if tuple(x['shape'])==shape and x['budget']==k)
                    close(actual,d['loss']);decisions+=1
                dir_correct+=(ordered[0]['family']=='host')==(best['family']=='host')
                regressions+=ordered[0]['observed_us']>host['observed_us']*(1+1e-12)
                errors.append(sum(abs(x['predicted_us']/x['observed_us']-1) for x in rs)/len(rs))
                for a,b in combinations(rs,2):
                    wrong+=((a['predicted_us'],a['family'])<(b['predicted_us'],b['family']))!=((a['observed_us'],a['family'])<(b['observed_us'],b['family']))
                    pairs+=1
            for k in losses:
                if str(k) not in stat['selection']:continue
                val=stat['selection'][str(k)];close(sum(losses[k])/len(losses[k]),val['mean_loss']);close(max(losses[k]),val['max_loss'])
                if sum(x==0 for x in losses[k])!=val['exact_winners']:raise ValueError('winner count mismatch')
            close(dir_correct/len(grouped),stat['direction_accuracy'])
            if regressions!=stat['host_regressions']:raise ValueError('host regression mismatch')
            gates=(dir_correct/len(grouped)>=.99 and regressions==0 and sum(losses[1])/len(losses[1])<=.005 and max(losses[1])<=.01)
            if gates!=stat['historical_gates_pass']:raise ValueError('gate mismatch')
            if method not in ('host','train_best_fixed'):
                close(sum(errors)/len(errors),stat['macro_shape_MAPE'])
                close(max(abs(x['predicted_us']/x['observed_us']-1) for x in rows),stat['max_APE'])
                if wrong!=stat['wrong_pairs']:raise ValueError('pair mismatch')
        if method.startswith('scale_'):
            native=models[method]['p8_four_shape_refit']
            if set(map(tuple,native['training_shapes']))!=set(map(tuple,design['calibration_shapes'])):
                raise ValueError('P8 calibration shapes changed')
            for kind,count in (('host',4),('fused',40)):
                if len(native['training_vectors'][kind])!=count:raise ValueError('unexpected calibration vector count')
            if any(s[0]!=4 for s in models[method]['p4_to_p8_no_calibration']['training_shapes']):
                raise ValueError('P8 data entered transfer calibration')
    return {'status':'passed','scored_configurations_checked':configs,
            'selection_summaries_checked':decisions,'candidate_pair_checks':pairs,
            'input_hashes_checked':len(seal['inputs']),'output_hashes_checked':len(seal['outputs']),
            'independent_GPU_validation':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    args=parser.parse_args();result=validate(args.folder)
    (args.folder/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
