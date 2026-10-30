"""Separate, predeclared warmup contrast for the observed short-K host drift."""
from pathlib import Path
import json,os,random,statistics,sys,time,hashlib
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'bundle-r01/source/EDM-IR'),str(ROOT/'bundle-r01/source/EDM-IR/tools')]
import run_mechanism_gpu as runner
from independent_selector_validation import digest

package=ROOT/'bundle-r01';build=ROOT/'work/build';out=ROOT/'work/short-host-diagnostic-01'
p,context,env=runner.preflight(package,build)
infos=json.loads((package/'inputs.json').read_text())['workloads']
variants=('host@legacy','host@prepared','host_merged_cutlass@merged','host_merged_cublaslt@merged')
rng=random.Random(20261022);jobs=[]
for block in range(3):
 group=[]
 for candidate in variants:
  original=next(j for j in p['jobs'] if j['workload_id']=='w000' and j['candidate_id']==candidate)
  for warmup in (20,200):
   group.append({**original,'stage':'short_host_warmup_diagnostic','repetition':block,'diagnostic_warmup':warmup})
 rng.shuffle(group)
 for j in group:jobs.append({**j,'trial_order':len(jobs),'ordinal':len(jobs)})
out.mkdir(exist_ok=False)
design={'question':'Does increasing warmup remove the short-K host process variation?',
 'jobs':jobs,'trigger':'Three host configurations at w000 exceeded max/min=1.10 across process medians',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'selection_after_main_data':True,'primary_dataset':False,'independent_validation':False,
 'profiler':False,'discarded_samples':0,'compiler_admission':False,
 'interpretation':'24 fresh processes; separate timing protocol, no replacement of original observations.'}
runner.base.write_json(out/'design.json',design)
runner.base.write_json(out/'context.json',context)
rows=[];start=time.monotonic()
for i,j in enumerate(jobs):
 if time.monotonic()-start>480:raise TimeoutError('diagnostic budget exhausted')
 if i%8==0:runner.telemetry(out,f'b{j["repetition"]}_before')
 trial={**p,'warmup':j['diagnostic_warmup']}
 text=runner.base.execute_job(runner.measurement_args(package,build,trial,j),j,out,i,timeout=120)
 row=runner.check_row(json.loads(text),j,infos[j['workload_id']],trial)
 if row['runtime']!=context['runtime']:raise ValueError('runtime changed')
 row['diagnostic_design_sha256']=digest(design)
 rows.append(row)
 with (out/'measurements.jsonl').open('a') as f:
  f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 print(i+1,len(jobs),j['candidate_id'],'warmup',trial['warmup'],'median_us',row['median_ms']*1000,flush=True)
summary={}
for candidate in variants:
 groups={n:[next(r['median_ms'] for r in rows if r['candidate_id']==candidate and r['warmup']==n and r['repetition']==b)
            for b in range(3)] for n in (20,200)}
 summary[candidate]={'process_medians_ms':groups,
  'warmup200_over_20_by_block':[a/b for a,b in zip(groups[200],groups[20])],
  'process_max_over_min':{n:max(v)/min(v) for n,v in groups.items()}}
runner.telemetry(out,'complete')
runner.base.write_json(out/'summary.json',{'design_sha256':digest(design),'candidates':summary,
 'processes':len(rows),'timed_samples':2400,'primary_dataset':False,
 'raw_sha256':runner.sha256(out/'measurements.jsonl'),'elapsed_seconds':time.monotonic()-start})
runner.base.write_json(out/'progress.json',{'state':'complete','completed':len(rows),'total':len(jobs)})
