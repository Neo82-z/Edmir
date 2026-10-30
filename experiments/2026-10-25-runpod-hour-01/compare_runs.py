#!/usr/bin/env python3
"""Compare separate collections without pooling labels or changing predictions."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--first',required=True,type=Path)
p.add_argument('--second',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
if a.out.exists():p.error('fresh output required')
read=lambda f:json.loads(f.read_text())
x=read(a.first);y=read(a.second)
def index(review):
    return {(s['stage'],tuple(r['shape']),r['candidate_id']):r for s in review['stages'] for r in s['configurations'] if r['primary_timing_label']}
aa=index(x);bb=index(y);keys=sorted(aa.keys()&bb.keys())
rows=[]
for key in keys:
    u,v=aa[key],bb[key]
    assert u['forecast_errors'].keys()==v['forecast_errors'].keys()
    for variant in u['forecast_errors']:
        assert u['forecast_errors'][variant]['predicted_us']==v['forecast_errors'][variant]['predicted_us']
    rows.append({'stage':key[0],'shape':key[1],'candidate_id':key[2],
        'first_us':u['observed_us'],'second_us':v['observed_us'],'second_over_first':v['observed_us']/u['observed_us'],
        'first_process_us':u['process_medians_us'],'second_process_us':v['process_medians_us'],
        'quality_flags_first':u['quality_flags'],'quality_flags_second':v['quality_flags'],
        'algorithm_equal':u['algorithm_ids']==v['algorithm_ids']})
pairs=[]
for stage,shape in sorted({(k[0],k[1]) for k in keys}):
    ids=sorted(k[2] for k in keys if k[:2]==(stage,shape))
    for c,d in combinations(ids,2):
        first=aa[stage,shape,d]['observed_us']-aa[stage,shape,c]['observed_us']
        second=bb[stage,shape,d]['observed_us']-bb[stage,shape,c]['observed_us']
        pairs.append({'stage':stage,'shape':shape,'a':c,'b':d,'first_b_minus_a_us':first,'second_b_minus_a_us':second,
            'direction_reversed':first*second<0,
            'first_process_deltas_us':[v-u for u,v in zip(aa[stage,shape,c]['process_medians_us'],aa[stage,shape,d]['process_medians_us'])],
            'second_process_deltas_us':[v-u for u,v in zip(bb[stage,shape,c]['process_medians_us'],bb[stage,shape,d]['process_medians_us'])]})
result={'matched_configurations':len(keys),'primary_missing_from_second':[list(k) for k in sorted(aa.keys()-bb.keys())],
    'max_absolute_between_run_fraction':max(abs(r['second_over_first']-1) for r in rows),
    'median_absolute_between_run_fraction':statistics.median(abs(r['second_over_first']-1) for r in rows),
    'changed_algorithms':sum(not r['algorithm_equal'] for r in rows),
    'same_context_for_common_stages':{s:x['contexts'][s]==y['contexts'][s] for s in x['contexts'].keys()&y['contexts'].keys()},
    'pair_directions_reversed':sum(r['direction_reversed'] for r in pairs),'total_pairs':len(pairs),
    'configurations':rows,'pairs':pairs,'pooled':False,'refit':False,
    'scope':'Two separate collections. No causal attribution of timing changes or confidence intervals.'}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('configurations','pairs')},indent=2))
