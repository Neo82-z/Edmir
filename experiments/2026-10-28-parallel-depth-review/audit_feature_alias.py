"""Exact prior-feature collisions; uses old development labels only."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import split_stage_selection as model


def main():
    source=ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json'
    rows=json.loads(source.read_text())['p4']['rows']
    state=json.loads((HERE/'run-06/depth_parallel_group-fold0.json').read_text())['model']['stage_prior']
    groups={}
    for r in rows:
        if not r['is_host']:groups.setdefault(tuple(r['shape']),{})[r['candidate_id']]=r
    result={}
    for variant in ('depth_parallel_group','depth_parallel_group_footprint'):
        seen={}
        for shape,rr in sorted(groups.items()):
            for a,b in itertools.combinations(sorted(rr),2):
                _,x=model.features(rr[a],variant,state['knots'],state['capacity'])
                _,y=model.features(rr[b],variant,state['knots'],state['capacity'])
                key=(a,b,tuple(v-u for u,v in zip(x,y)))
                seen.setdefault(key,[]).append({'shape':shape,'gap_b_minus_a_us':rr[b]['latency_us']-rr[a]['latency_us']})
        conflicts=[]
        for (a,b,x),values in seen.items():
            gaps=[v['gap_b_minus_a_us'] for v in values]
            if min(gaps)<0<max(gaps):
                conflicts.append({'pair':[a,b],'observations':values,'minimax_gap_error_us':(max(gaps)-min(gaps))/2})
        result[variant]=sorted(conflicts,key=lambda x:-x['minimax_gap_error_us'])
    output={'exact_opposite_order_collisions':result,'capacity':state['capacity'],'depth_knots':state['knots'],
        'scope':'shared prior only, not local M/N corrections; label conflict is not a noise-free physical law',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_six_used':False,'new_fit':False}
    (HERE/'feature-collisions.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
