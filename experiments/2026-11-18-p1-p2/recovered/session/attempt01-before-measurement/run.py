#!/usr/bin/env python3
"""Execute the sealed P1/P2 matrix once, with a platform-bound spending cutoff."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'bundle/source/EDM-IR'),str(HERE/'bundle/source/EDM-IR/tools')]
import run_w1_gpu as w1
from run_p8_gpu import plain_topology
from run_selector_gpu import environment, measurement_environment
sha=w1.sha256
save=w1.save

def cutoff(guard, now=None):
    now=time.time() if now is None else now
    if guard.get('authorization_confirmed') is not True or guard.get('platform_verified') is not True:
        raise ValueError('verified authorization and platform auto-delete required')
    if not guard.get('instance_id') or guard.get('experiment_schema')!='edmir.p1-p2.closeout.v1':
        raise ValueError('wrong experiment/instance guard')
    keys=['billed_from_epoch','platform_stop_epoch','gpu_hourly_usd','storage_hourly_usd','compute_cap_usd','total_budget_usd']
    if any(type(guard.get(k)) not in (int,float) or not math.isfinite(guard[k]) for k in keys):
        raise ValueError('nonfinite budget')
    start,end,rate,storage,cap,budget=(guard[k] for k in keys)
    if not (0<budget<=25 and 0<cap<=24 and rate>0 and storage>=0 and start<=now<end):
        raise ValueError('invalid spending window')
    if (end-start)*rate/3600>cap+1e-6 or (end-start)*(rate+storage)/3600>budget:
        raise ValueError('platform deadline exceeds spending bound')
    if end-600<=now:raise TimeoutError('reserved recovery window reached')
    return end-600

class Clock(w1.Deadline):
    def __init__(self,guard):self.end=cutoff(guard)

def validate_plan(plan,pred):
    if plan['schema']!='edmir.p1-p2.closeout.v1' or plan['no_training'] is not True:
        raise ValueError('wrong schema')
    if sha(HERE/'preregistered-predictions.json')!=plan['predictions_sha256']:
        raise ValueError('predictions changed')
    if sha(HERE/'frozen-primary-model.json')!=plan['model_sha256']:
        raise ValueError('model changed')
    if set(pred['predictions'])!=set(plan['shapes'])-{'m512'}:
        raise ValueError('six-shape identity differs')
    if len(plan['primary'])!=13 or len(set(plan['primary']))!=13:
        raise ValueError('candidate surface differs')
    for tag,x in pred['predictions'].items():
        if x['shape']!=plan['shapes'][tag]:raise ValueError('shape differs')
    for tag in plan['shapes']:
        checks=[j for j in plan['jobs'] if j['shape']==tag and j['phase']=='check']
        if [j['seed'] for j in checks]!=plan['seeds']:raise ValueError('all correctness seeds required')
        if tag!='m512':
            verify=[j for j in plan['jobs'] if j['shape']==tag and j['phase']=='verify']
            select=[j for j in plan['jobs'] if j['shape']==tag and j['phase']=='select']
            if len(verify)!=3 or len(select)!=1 or any(set(j['order'])!=set(plan['primary']) for j in verify):
                raise ValueError('verification and selection coverage differs')

def selection_order(plan,pred,tag,failed):
    rank=[f for f in pred['predictions'][tag]['frozen_ranking'] if (tag,f) not in failed]
    fixed=[f for f in plan['fixed_order'] if (tag,f) not in failed]
    return sorted(set(rank[:3]+fixed))

def preflight(a,plan,clock):
    b=a.build_dir;receipt=json.loads((b/'receipt.json').read_text())
    if receipt['package']!=sha(a.bundle/'files.json'):raise ValueError('build package differs')
    for name,h in receipt['dependencies'].items():
        if sha(Path(name))!=h:raise ValueError('dependency changed')
    loaded={}
    for tag in plan['shapes']:
        binary=b/f'w1_{tag}_batch'
        if sha(binary)!=receipt['binaries'][tag]:raise ValueError('binary changed')
        linked=clock.text(['ldd',binary]);libs=re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',linked,re.M)
        for _,p in libs:
            if p not in loaded:loaded[p]=sha(Path(p))
        if 'not found' in linked or [loaded[p] for lib,p in libs if lib.startswith('libnccl.so')]!=[receipt['nccl_library']]:
            raise ValueError('loaded NCCL differs')
    runtime=json.loads(clock.text([b/'w1_m512_batch','--inspect']))
    inventory=clock.text(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'])
    topo=plain_topology(clock.text(['nvidia-smi','topo','-m']))
    hardware=environment(runtime,topo,inventory,8)
    if runtime['cuda_version']!='12080' or runtime['nccl_version']!='22809':
        raise ValueError('runtime differs from sealed CUDA12.8/NCCL2.28.9')
    active=clock.text(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    if active:raise ValueError('GPUs are not idle')
    return dict(package=receipt['package'],build=sha(b/'receipt.json'),hardware=hardware,
        gpu_inventory=clock.text(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader']),
        loaded_libraries=loaded,topology=topo,runtime=runtime,measurement_env=measurement_environment(os.environ),
        sealed_runner=sha(__file__),plan=sha(a.bundle/'plan.json'))

def collect(a,plan,records,pred,clock,context):
    failed=set();context_hash=sha(a.out/'context.json')
    for i,job in enumerate(plan['jobs']):
        tag=job['shape'];final=a.out/f'collect-{i:03d}.json'
        order=selection_order(plan,pred,tag,failed) if job['phase']=='select' else [f for f in job['order'] if job['phase']=='check' or (tag,f) not in failed]
        if final.exists():
            rec=json.loads(final.read_text())
            if rec['job']!=job or rec['context']!=context_hash or rec['order']!=order or w1.done_raw_changed(final,a.out):
                raise ValueError('resume record differs')
            rows=rec['rows']
        else:
            data=a.inputs/tag/str(job['seed']);stem=f'collect-{i:03d}-{time.time_ns()}'
            stdout=a.out/(stem+'.stdout');stderr=a.out/(stem+'.stderr')
            began=time.time()
            with stdout.open('x') as o,stderr.open('x') as e:
                if order:clock.run([a.build_dir/f'w1_{tag}_batch','--check' if job['phase']=='check' else '--measure',
                         data/'inputs.bin',data/'reference.bin',str(job['seed']),*order],300,stdout=o,stderr=e)
            rows=[json.loads(s) for s in stdout.read_text().splitlines()]
            w1.validate_rows(rows,job,order,plan['shapes'][tag],context['runtime'])
            save(final,dict(job=job,context=context_hash,input=records[tag][str(job['seed'])],rows=rows,order=order,
                 excluded_by_numerical_check=sorted(f for t,f in failed if t==tag),stdout=stdout.name,
                 stdout_sha256=sha(stdout),stderr=stderr.name,process_seconds=time.time()-began))
        w1.validate_rows(rows,job,order,plan['shapes'][tag],context['runtime'])
        for row in rows:
            if row['correctness_passed'] is not True:failed.add((tag,row['family']))
        print(f"{i+1}/{len(plan['jobs'])} {tag} {job['phase']} rows={len(rows)}",flush=True)
    save(a.out/'collect-complete.json',dict(groups=len(plan['jobs']),failed=sorted(failed),finished_epoch=time.time()))

def traces(a,plan,clock,context):
    if not (a.out/'collect-complete.json').exists():raise ValueError('unprofiled collection must finish first')
    version=clock.text([a.nsys,'--version'])
    if '2025.5.2' not in version:raise ValueError('wrong prechecked profiler')
    statuses={}
    for name,(tag,family) in plan['p1_trace_cases'].items():
        evidence=[json.loads(p.read_text()) for p in sorted(a.out.glob('collect-[0-9][0-9][0-9].json'))]
        rows=[r for x in evidence if x['job']['shape']==tag for r in x['rows'] if r['family']==family]
        if not rows or any(not r['correctness_passed'] for r in rows):raise ValueError('trace requires admitted candidate')
        dest=a.out/'diagnostic'/name;dest.mkdir(parents=True,exist_ok=False)
        seed=plan['seeds'][0];data=a.inputs/tag/str(seed)
        cmd=[a.nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-event-trace=false',
            '--capture-range=nvtx','--capture-range-end=stop','-p','edmir.w1.capture',
            '-e','NSYS_NVTX_PROFILER_REGISTER_ONLY=0','-o',dest/'timeline',
            a.build_dir/f'w1_{tag}_batch','--trace',data/'inputs.bin',data/'reference.bin',str(seed),family]
        save(dest/'intent.json',dict(command=list(map(str,cmd)),version=version,context=sha(a.out/'context.json'),purpose='diagnostic_only'))
        with (dest/'stdout').open('x') as o,(dest/'stderr').open('x') as e:clock.run(cmd,150,stdout=o,stderr=e)
        report=dest/'timeline.nsys-rep'
        if not report.exists() or '"diagnostic_only":true' not in (dest/'stdout').read_text():raise ValueError('trace missing')
        with (dest/'export.stdout').open('x') as o,(dest/'export.stderr').open('x') as e:
            clock.run([a.nsys,'export','--type=sqlite','--output',dest/'timeline.sqlite',report],120,stdout=o,stderr=e)
        save(dest/'receipt.json',dict(report_sha256=sha(report),sqlite_sha256=sha(dest/'timeline.sqlite'),
            diagnostic_only=True,completeness_requires_sqlite_audit=True))
        statuses[name]='captured_requires_audit';print('trace',name,'captured',flush=True)
    save(a.out/'traces-complete.json',statuses)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['verify','collect','trace'],required=True)
    p.add_argument('--bundle',type=Path,default=HERE/'bundle');p.add_argument('--inputs',type=Path,default=HERE/'inputs')
    p.add_argument('--build-dir',type=Path,default=HERE/'build');p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--guard',type=Path,default=HERE/'guard.json');p.add_argument('--nsys',default='/opt/edmir-nsys/opt/nvidia/nsight-systems/2025.5.2/bin/nsys')
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    plan,records=w1.verify(a.bundle,a.inputs)
    pred=json.loads((HERE/'preregistered-predictions.json').read_text());validate_plan(plan,pred)
    if a.action=='verify':print('sealed P1/P2 package and inputs verified; GPU not used');return
    guard=json.loads(a.guard.read_text());clock=Clock(guard)
    if guard.get('runner_sha256')!=sha(__file__) or guard.get('bundle_sha256')!=sha(a.bundle/'files.json'):
        raise ValueError('guard not bound to sealed runner/package')
    context=preflight(a,plan,clock);a.out.mkdir(exist_ok=True,parents=True)
    with (a.out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cp=a.out/'context.json'
        if cp.exists() and json.loads(cp.read_text())!=context:raise ValueError('new execution context cannot resume old collection')
        save(cp,context);save(a.out/'guard-receipt.json',guard)
        status=dict(state='running',action=a.action,started_epoch=time.time())
        try:
            if a.action=='collect':collect(a,plan,records,pred,clock,context)
            else:traces(a,plan,clock,context)
            status['state']='complete'
        except BaseException as exc:status.update(state='failed',error=repr(exc));raise
        finally:status['finished_epoch']=time.time();save(a.out/(a.action+'-status.json'),status)

if __name__=='__main__':main()
