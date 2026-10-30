#!/usr/bin/env python3
"""Budget-limited development diagnostic, separate from frozen full matrix."""
from pathlib import Path
import sys,json,time,os,traceback,hashlib
root=Path('/root/2026-10-25-p8-gpu-collection');package=root/'bundle';build=root/'work/build'
sys.path[:0]=[str(package/'source/EDM-IR/tools'),str(package/'source/EDM-IR')]
import run_p8_gpu as r
base=r.base
DEADLINE=1793165400
outroot=root/'work/subset';outroot.mkdir(exist_ok=False)
selection={'shape5':['host','host_merged_cutlass','host_merged_cublaslt','splitk4','splitk4_groupedn2','splitk8_groupedn2'],'shape6':['host','host_merged_cutlass','host_merged_cublaslt','splitk8','splitk8_groupedn2','splitk16_groupedn2'],'shape3':['host','host_merged_cutlass','host_merged_cublaslt','splitk2','splitk8'],'shape7':['host','host_merged_cutlass','host_merged_cublaslt','splitk8_groupedn2','splitk16_groupedn2']}
p,context,env=r.preflight(package,build)
jobs=[j for stage,families in selection.items() for j in p['jobs'] if j['stage']==stage and j['primary'] and j['family'] in families]
assert len(jobs)==66,len(jobs)
manifest={'schema':'edmir.budget-subset.v1','created_epoch':time.time(),'original_package_sha256':context['package_sha256'],'selection':selection,'jobs':jobs,'deadline_epoch':DEADLINE,'reason':'Fresh-process startup/teardown limits complete full matrix within rental cap. Fixed mechanism contrasts selected from prior mechanism plan, not measured speed. Full matrix failed numerical correctness on shape5 splitk16 g1 before this collection. All original rows retained separately.','independent_validation':False,'compiler_admission':False,'full_candidate_oracle':False,'frozen_six_used':False}
base.write_json(outroot/'subset-plan.json',manifest)
base.write_json(outroot/'preflight.json',{'context':context,'environment':env})
inputs=json.loads((package/'inputs.json').read_text())['workloads']
finished=[];failed=[]
for stage in selection:
    out=outroot/stage;out.mkdir();sj=[j for j in jobs if j['stage']==stage];rows=[];failures=[];attempted=0;previous=None
    base.write_json(out/'context.json',{**context,'stage':stage,'subset_plan_sha256':hashlib.sha256((outroot/'subset-plan.json').read_bytes()).hexdigest()})
    start=time.time()
    for i,job in enumerate(sj):
        if time.time()>DEADLINE-95:break
        if previous!=job['repetition']:
            r.telemetry(out,f'b{job["repetition"]}_before_{i:05d}');previous=job['repetition']
        t=time.time()
        try:
            value=base.execute_job(r.measurement_args(package,build,p,job),job,out,i,timeout=min(p['process_timeout_seconds'],DEADLINE-time.time()-5))
            row=r.check_row(json.loads(value),job,inputs[job['workload_id']],p)
            if row['runtime']!=context['runtime']:raise ValueError('runtime differs')
            row['process_wall_seconds']=time.time()-t
            with (out/'measurements.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row)
            print(stage,i+1,len(sj),job['candidate_id'],'PASS',row['median_ms'],flush=True)
        except Exception as error:
            failure={'job':job,'job_index':i,'exception':type(error).__name__,'detail':str(error),'process_wall_seconds':time.time()-t}
            failures.append(failure)
            base.write_json(out/f'failure_{i:05d}.json',failure)
            print(stage,i+1,len(sj),job['candidate_id'],'FAIL',type(error).__name__,flush=True)
        attempted=i+1
        base.write_json(out/'progress.json',{'state':'collecting','attempted':attempted,'successful':len(rows),'failed':len(failures),'total':len(sj),'stage':stage})
    r.telemetry(out,'end')
    base.write_json(out/'failures.json',failures)
    base.write_json(out/'run.json',{'schema':'edmir.budget-subset-run.v1','context':context,'environment':env,'rows':rows,'failures':failures,'independent_validation':False,'compiler_admission':False})
    if len(rows)==len(sj):
        summary=r.summarize({**p,'jobs':sj},rows,stage)
        summary.update(scope='declared budget subset only; excludes unmeasured original candidates',original_candidate_count=len({j['candidate_id'] for j in p['jobs'] if j['stage']==stage}),declared_candidate_count=len(sj)//3)
        base.write_json(out/'summary.json',summary);finished.append(stage)
    else:failed.append(stage)
    base.write_json(out/'progress.json',{'state':'complete' if len(rows)==len(sj) else 'complete_with_failures' if attempted==len(sj) else 'deadline','attempted':attempted,'successful':len(rows),'failed':len(failures),'total':len(sj),'elapsed_seconds':time.time()-start,'stage':stage})
    base.write_json(outroot/'progress.json',{'finished':finished,'incomplete_or_failed':failed,'current_stage':stage,'epoch':time.time()})
    if time.time()>DEADLINE-95:break
base.write_json(outroot/'final.json',{'finished':finished,'incomplete_or_failed':failed,'epoch':time.time(),'independent_validation':False,'compiler_admission':False})
