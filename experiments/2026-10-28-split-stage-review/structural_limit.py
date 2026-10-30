"""Necessary error bound from nonnegative feature-difference ordering.

This diagnoses the shared prior, not the noise-free device or the local
anchor corrections. It does not fit coefficients or use held-out six shapes.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR')]
from edmir.compiler import split_stage_selection as model


def main():
    data_path=ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json'
    rows=json.loads(data_path.read_text())['p4']['rows']
    by_n={}
    for r in rows:
        p,m,k,n=r['shape']
        if (p,m,k)==(4,64,512) and r['candidate_id'] in ('direct','splitk2'):
            by_n.setdefault(n,{})[r['candidate_id']]=r
    pairs=[]
    for na,a in sorted(by_n.items()):
        for nb,b in sorted(by_n.items()):
            if na>=nb:continue
            certificates=[]
            for fold in range(4):
                s=json.loads((HERE/f'final-validation/depth-fold{fold}.json').read_text())['model']['stage_prior']
                names,fa=model.features(a['direct'],'depth',s['knots'],s['capacity'])
                ga=model.features(a['splitk2'],'depth',s['knots'],s['capacity'])[1]
                fb=model.features(b['direct'],'depth',s['knots'],s['capacity'])[1]
                gb=model.features(b['splitk2'],'depth',s['knots'],s['capacity'])[1]
                change=[(v-u)-(y-x) for u,v,x,y in zip(fb,gb,fa,ga)]
                if min(change)<-1e-12:raise ValueError('nonnegative cone ordering not certified')
                certificates.append({'fold':fold,'positive_feature_differences':{
                    n:v for n,v in zip(names,change) if abs(v)>1e-12}})
            da=a['splitk2']['latency_us']-a['direct']['latency_us']
            db=b['splitk2']['latency_us']-b['direct']['latency_us']
            pairs.append({'N_small':na,'N_large':nb,'measured_delta_small_us':da,'measured_delta_large_us':db,
                'any_shared_prior_minimum_max_pair_gap_error_us':max(0.,(da-db)/2),
                'opposite_winners_impossible_for_monotone_gap':da>0 and db<0,
                'certificates':certificates})
    result={'shape_scope':{'P':4,'M':64,'K':512,'N':sorted(by_n)},
        'formula':'delta(N)=T(q2)-T(direct)=f(8)-f(16)+serial_stages+ctas*ceil(N/128)/108',
        'assumptions':'q1/q2 pressure=0, g=1, common nonnegative coefficients, no local anchor correction',
        'meaning':'necessary representational error on observed labels, not proof of hardware causality',
        'not_a_cross_validation_error_lower_bound':True,
        'reason':'each fold has different coefficients; local corrections are outside the certified family',
        'pair_gap_error_bound_us':max(p['any_shared_prior_minimum_max_pair_gap_error_us'] for p in pairs),
        'pairs':pairs,'frozen_six_used':False,
        'source_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()}
    (HERE/'structural-limit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':main()
