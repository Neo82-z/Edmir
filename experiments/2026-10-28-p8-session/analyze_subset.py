#!/usr/bin/env python3
"""Audit declared candidate subset; never infer labels for failed/unmeasured jobs."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=a.results/'work/subset';plan=json.loads((root/'subset-plan.json').read_text());jobs=plan['jobs'];reports=[]
key=lambda j:(j['stage'],j['candidate_id'],j['repetition'])
assert len(jobs)==len({key(j) for j in jobs})==66
for stage in plan['selection']:
 d=root/stage;sj=[j for j in jobs if j['stage']==stage];expected={key(j):j for j in sj}
 rows=[json.loads(x) for x in (d/'measurements.jsonl').read_text().splitlines()] if (d/'measurements.jsonl').exists() else []
 failures=[json.loads(f.read_text()) for f in sorted(d.glob('failure_*.json'))]
 index={key(r):r for r in rows};assert len(index)==len(rows)
 for k,r in index.items():
  j=expected[k]
  assert all(r[z]==j[z] for z in ('shape','candidate_id','family','implementation_id','metadata_mode','dependency_policy','ordinal'))
  assert r['correctness_passed'] and r['iterations']==100 and r['warmup']==20 and len(r['samples_ms'])==100
  assert r['checked_elements']==j['shape'][0]**2*j['shape'][1]*j['shape'][3]
  assert abs(statistics.median(r['samples_ms'])-r['median_ms'])<1e-10
 fk={key(f['job']) for f in failures};assert len(fk)==len(failures) and not fk.intersection(index) and fk.issubset(expected)
 report={'stage':stage,'shape':sj[0]['shape'],'planned_processes':len(sj),'valid_processes':len(rows),'failed_processes':len(failures),'all_jobs_attempted':len(rows)+len(failures)==len(sj),'all_jobs_passed':len(rows)==len(sj),'failures':failures,'candidates':{}}
 for candidate in sorted({j['candidate_id'] for j in sj}):
  rr=sorted([r for r in rows if r['candidate_id']==candidate],key=lambda r:r['repetition']);ff=[f for f in failures if f['job']['candidate_id']==candidate]
  complete=len(rr)==3 and {r['repetition'] for r in rr}=={0,1,2} and not ff
  times=[r['median_ms']*1000 for r in rr]
  report['candidates'][candidate]={'complete_and_correct':complete,'valid_processes':len(rr),'failed_processes':len(ff),'repetitions':[r['repetition'] for r in rr],'process_medians_us':times}
  if complete:
   report['candidates'][candidate].update(median_us=statistics.median(times),max_over_min=max(times)/min(times),drift_review=max(times)/min(times)>1.10,max_abs_error=max(r['max_abs_error'] for r in rr),max_rel_error=max(r['max_rel_error'] for r in rr),median_process_wall_seconds=statistics.median(r['process_wall_seconds'] for r in rr))
 valid={c:v for c,v in report['candidates'].items() if v['complete_and_correct']};hosts={c:v for c,v in valid.items() if c.startswith('host')};fused={c:v for c,v in valid.items() if not c.startswith('host')}
 if len(hosts)==3 and fused:
  hc=min(hosts,key=lambda c:hosts[c]['median_us']);fc=min(fused,key=lambda c:fused[c]['median_us']);ht=hosts[hc];ft=fused[fc]
  report['comparison']={'best_measured_host':hc,'best_measured_fused':fc,'host_us':ht['median_us'],'fused_us':ft['median_us'],'host_over_fused':ht['median_us']/ft['median_us'],'paired_fused_minus_host_us':[f-h for f,h in zip(ft['process_medians_us'],ht['process_medians_us'])],'per_block_fastest_host':[min(hosts,key=lambda c:hosts[c]['process_medians_us'][i]) for i in range(3)],'per_block_fastest_fused':[min(fused,key=lambda c:fused[c]['process_medians_us'][i]) for i in range(3)],'warning':'Descriptive best of correct, completely measured subset. Post-measurement selection; no full oracle, confidence bound or predictor accuracy claim.'}
 reports.append(report)
partial=a.results/'work/shape5-01';pr=[json.loads(x) for x in (partial/'measurements.jsonl').read_text().splitlines()] if (partial/'measurements.jsonl').exists() else []
report={'schema':'edmir.budget-subset-analysis.v1','subset_plan_sha256':hashlib.sha256((root/'subset-plan.json').read_bytes()).hexdigest(),'independent_validation':False,'compiler_admission':False,'full_candidate_oracle':False,'training_performed':False,'frozen_six_used':False,'valid_subset_processes':sum(r['valid_processes'] for r in reports),'failed_subset_processes':sum(r['failed_processes'] for r in reports),'full_matrix_partial':{'valid_rows':len(pr),'failure':json.loads((partial/'failure.json').read_text()) if (partial/'failure.json').exists() else None,'included_in_subset_statistics':False},'shapes':reports}
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'valid':report['valid_subset_processes'],'failed':report['failed_subset_processes'],'shapes':[{'stage':s['stage'],'passed':s['all_jobs_passed'],'valid':s['valid_processes'],'failed':s['failed_processes'],'comparison':s.get('comparison')} for s in reports]},indent=2))
