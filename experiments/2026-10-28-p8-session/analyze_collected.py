#!/usr/bin/env python3
"""Descriptive development results only; no training or unseen-shape claim."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text()); reports=[]
for stage in plan['stages']:
 directory=a.results/'work'/(stage+'-01')
 progress=json.loads((directory/'progress.json').read_text()) if (directory/'progress.json').exists() else {'state':'not_started'}
 record={'stage':stage,'progress':progress,'complete':progress.get('state')=='complete'}
 if record['complete']:
  summary=json.loads((directory/'summary.json').read_text());w=summary['workloads'][0]
  configs=[c for c in plan['configurations'] if c['stage']==stage and c['primary']]
  values=w['median_of_process_medians_ms'];primary={c['candidate_id']:values[c['candidate_id']] for c in configs}
  host={c:v for c,v in primary.items() if c.startswith('host')};fused={c:v for c,v in primary.items() if not c.startswith('host')}
  h=min(host,key=host.get);f=min(fused,key=fused.get)
  record.update(shape=w['shape'],primary_candidates=len(primary),process_repeats=3,best_host=h,best_fused=f,best_host_us=1000*host[h],best_fused_us=1000*fused[f],fused_vs_best_host_speedup=host[h]/fused[f],best_primary=min(primary,key=primary.get),primary_median_us={c:1000*v for c,v in primary.items()},process_max_over_min=w['process_max_over_min'],drift_review=w['review_drift_candidates'])
 reports.append(record)
result={'independent_validation':False,'compiler_admission':False,'training_performed':False,'note':'Historical development shapes, current P8 runtime. Three fresh process blocks; all observations retained. Best-of-measured medians are descriptive, not statistical confidence or model-selection performance. No frozen-six labels used.','complete_shapes':sum(x['complete'] for x in reports),'shapes':reports}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
