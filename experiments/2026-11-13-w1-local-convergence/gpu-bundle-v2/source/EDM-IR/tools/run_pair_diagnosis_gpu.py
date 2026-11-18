#!/usr/bin/env python3
"""Bounded fixed-candidate collection. No fitting, rental or automatic retry."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import tarfile
import time

from prepare_pair_diagnosis_gpu import verify_bundle,CUTLASS_REVISION,NCCL_REVISION
from prepare_selector_gpu import REPO
from independent_selector_validation import digest
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.two_stage_selection import parse_schedule_family
import run_selector_gpu as base
from run_boundary_gpu import telemetry
from run_p8_gpu import inventory,plain_topology


def require_frozen_runner(package):
    if REPO.resolve()!=(package/'source').resolve():
        raise ValueError('run package/source/EDM-IR/tools/run_pair_diagnosis_gpu.py')


def build(package,directory,*,cutlass,nccl_source,nccl_include,nccl_library,jobs):
    require_frozen_runner(package);p=verify_bundle(package)
    if not 1<=jobs<=16:raise ValueError('build jobs must be 1..16')
    directory.mkdir(parents=True,exist_ok=True);receipt=directory/'selector_build.json'
    if receipt.exists():raise ValueError('build receipt already exists')
    identities={'cutlass_commit':base.git_commit(cutlass),'nccl_commit':base.git_commit(nccl_source)}
    if identities!={'cutlass_commit':CUTLASS_REVISION,'nccl_commit':NCCL_REVISION}:raise ValueError('dependency revision mismatch')
    for source in (cutlass,nccl_source):
        if base.command(['git','-C',source,'status','--porcelain','--untracked-files=no'],capture_output=True).stdout.strip():
            raise ValueError('dependency has tracked changes')
    dep={str(path):sha256(path) for path in (nccl_library,nccl_include/'nccl.h',nccl_include/'nccl_device.h')}
    tools={n:base.command(cmd,capture_output=True).stdout for n,cmd in
           (('nvcc',['nvcc','--version']),('cxx',['c++','--version']),('cmake',['cmake','--version']))}
    match=re.search(r'cmake version (\d+)\.(\d+)',tools['cmake'])
    if not match or tuple(map(int,match.groups()))<(3,30):raise ValueError('CMake >= 3.30 required for the verified CUDA C++20 setup')
    base.command(['cmake','-S',REPO/'EDM-IR/runtime','-B',directory,'-DCMAKE_BUILD_TYPE=Release',
      '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v','-DEDMIR_ENABLE_CUDA=ON','-DEDMIR_ENABLE_CUTLASS=ON',
      '-DEDMIR_ENABLE_NCCL_DEVICE=ON',f'-DEDMIR_CUTLASS_DIR={cutlass}',f'-DEDMIR_EXPERIMENT_DIR={package}',
      f'-DNCCL_INCLUDE_DIR={nccl_include}',f'-DNCCL_LIBRARY={nccl_library}',f'-DCMAKE_BUILD_RPATH={nccl_library.parent}'],timeout=180)
    with (directory/'build.stdout.txt').open('w') as out,(directory/'build.stderr.txt').open('w') as err:
        base.command(['cmake','--build',directory,'-j',jobs],timeout=900,stdout=out,stderr=err)
    base.command(['ctest','--test-dir',directory,'--output-on-failure','--timeout','30'],timeout=180)
    base.write_json(receipt,{'package_sha256':digest(p),'dependencies':dep,'environment':identities,'toolchain':tools,
        'paired_build_complete':True,'binaries':{n:sha256(directory/n) for n in p['programs']},
        'nccl_library_sha256':sha256(nccl_library),'cutlass_source':str(cutlass),'nccl_source':str(nccl_source)})


def preflight(package,directory,*,deadline=None):
    deadline=time.monotonic()+180 if deadline is None else deadline
    def call(args,**kwargs):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('preflight wall budget exhausted')
        kwargs['timeout']=min(kwargs.get('timeout',30),remaining)
        return base.command(args,**kwargs)
    require_frozen_runner(package);p=verify_bundle(package)
    timing=base.measurement_environment(os.environ)
    receipt=json.loads((directory/'selector_build.json').read_text())
    if receipt.get('package_sha256')!=digest(p) or receipt.get('paired_build_complete') is not True:
        raise ValueError('build receipt does not cover package')
    if set(receipt['binaries'])!=set(p['programs']) or set(receipt.get('toolchain',{}))!={'nvcc','cxx','cmake'}:
        raise ValueError('incomplete binary/toolchain receipt')
    for path,h in receipt['dependencies'].items():
        if sha256(Path(path))!=h:raise ValueError('build dependency changed')
    libraries={}
    for name,h in receipt['binaries'].items():
        if sha256(directory/name)!=h:raise ValueError('binary changed')
        linked=call(['ldd',directory/name],capture_output=True,timeout=30).stdout
        if 'not found' in linked:raise ValueError('missing runtime library')
        nccl=[]
        for lib,path in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',linked,re.MULTILINE):
            resolved=str(Path(path).resolve())
            if resolved not in libraries:libraries[resolved]=sha256(Path(resolved))
            if lib.startswith('libnccl.so'):nccl.append(resolved)
        if len(nccl)!=1 or libraries[nccl[0]]!=receipt['nccl_library_sha256']:raise ValueError('loaded NCCL differs')
    raw=call(['nvidia-smi','topo','-m'],capture_output=True,timeout=30).stdout
    topo=plain_topology(raw)
    inv=call(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'],capture_output=True,timeout=30).stdout
    uuids=call(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],capture_output=True,timeout=30).stdout
    runtime=json.loads(call([directory/p['jobs'][0]['executable'],'--inspect'],capture_output=True,timeout=60).stdout)
    env=base.environment(runtime,topo,inv,p['world_size'])
    context={'package_sha256':digest(p),'build_sha256':sha256(directory/'selector_build.json'),
      'profile':p['profile'],'runtime':runtime,'topology':topo,'raw_topology':raw,'inventory':inv,
      'device_uuid_inventory':uuids,'loaded_library_sha256':libraries,'measurement_environment':timing,
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    context['environment_id']='pair-diagnosis-'+p['profile']+'-'+digest(context)[:20]
    env.update(receipt['environment']);env['collector_sha256']=p['collector_sha256']
    return p,context,env


def args_for(package,directory,p,job):
    data=package/'data'/job['workload_id']
    return [directory/job['executable'],job['family'],data/'inputs.bin',data/'reference.bin',p['warmup'],p['iterations'],job['metadata_mode']]


def check_row(row,job,inputs,p):
    checked=base.validate_measurement(row,job,inputs,p)
    q=1 if job['family']=='host' else parse_schedule_family(job['family'])[0]
    expected='sm80_64x128x32'+(f'_splitk{q}' if q>1 else '')
    if row['gemm_algorithm_id']!=expected:raise ValueError('algorithm differs from frozen candidate')
    return {**checked,'ordinal':job['ordinal'],'workload_id':job['workload_id']}


def summarize(p,rows):
    index={};expected={(j['workload_id'],j['family'],j['repetition']) for j in p['jobs']}
    for r in rows:
        key=(r['workload_id'],r['family'],r['repetition'])
        if key not in expected or key in index:raise ValueError('duplicate or unexpected row')
        index[key]=r
    works=[]
    for w in p['workloads']:
        complete=[b for b in range(w['blocks']) if all((w['id'],f,b) in index for f in w['families'])]
        stats={}
        for f in w['families']:
            observed=[index[(w['id'],f,b)] for b in range(w['blocks']) if (w['id'],f,b) in index]
            stats[f]={'block_ids':[r['repetition'] for r in observed],
              'process_medians_ms':[r['median_ms'] for r in observed],
              'process_means_ms':[r['mean_ms'] for r in observed],
              'process_p95_ms':[sorted(r['samples_ms'])[math.ceil(.95*len(r['samples_ms']))-1] for r in observed]}
        pairs=[('host',f) for f in w['families'] if f!='host']
        if 'direct' in w['families']:pairs.append(('direct','splitk2'))
        else:pairs.extend([('splitk4','splitk8'),('splitk4_groupedn2','splitk8_groupedn2'),
                           ('splitk4','splitk4_groupedn2'),('splitk8','splitk8_groupedn2'),('splitk4','splitk8_groupedn2')])
        gaps=[]
        for a,b in pairs:
            block_ids=[i for i in range(w['blocks']) if (w['id'],a,i) in index and (w['id'],b,i) in index]
            delta=[1000*(index[(w['id'],b,i)]['median_ms']-index[(w['id'],a,i)]['median_ms']) for i in block_ids]
            gaps.append({'contrast':b+' - '+a,'block_ids':block_ids,'median_gaps_us':delta,
              'mean_gaps_us':[1000*(index[(w['id'],b,i)]['mean_ms']-index[(w['id'],a,i)]['mean_ms']) for i in block_ids],
              'negative_count':sum(v<0 for v in delta),'positive_count':sum(v>0 for v in delta),
              'all_scheduled_pair_blocks_present':len(block_ids)==w['blocks'],'confidence_interval':None})
        interaction=[]
        if 'splitk8_groupedn2' in w['families']:
            for b in complete:
                t=lambda n:index[(w['id'],n,b)]['median_ms']
                interaction.append({'block':b,'q_by_g_interaction_us':1000*(t('splitk8_groupedn2')-t('splitk4_groupedn2')-t('splitk8')+t('splitk4'))})
        works.append({**w,'complete_block_ids':complete,'candidates':stats,'paired_contrasts':gaps,'grouped_interactions':interaction})
    return {'complete':set(index)==expected,'completed_rows':len(rows),'scheduled_rows':len(expected),'workloads':works,
       'independent_validation':False,'compiler_admission':False,'full_candidate_oracle':False,
       'causal_component_times_identified':False,'no_ci_from_within_process_iterations':True}


def collect(package,directory,out,*,resume=False,budget_seconds=None):
    started=time.time();monotonic_start=time.monotonic()
    if budget_seconds is not None and (not math.isfinite(budget_seconds) or budget_seconds<=0):raise ValueError('invalid time budget')
    p,context,env=preflight(package,directory,deadline=monotonic_start+min(180,budget_seconds or 180))
    budget=p['collection_budget_seconds'] if budget_seconds is None else budget_seconds
    if not math.isfinite(budget) or not 0<budget<=p['collection_budget_seconds']:raise ValueError('budget exceeds frozen cap')
    if out.exists() and not resume:raise ValueError('fresh output or explicit resume required')
    if resume and not (out/'context.json').exists():raise ValueError('missing resume context')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'context.json').exists() and json.loads((out/'context.json').read_text())!=context:
        raise ValueError('resume identity differs')
    base.write_json(out/'context.json',context);base.write_json(out/'environment.json',env)
    if not (out/'budget.json').exists():base.write_json(out/'budget.json',{'started_unix':started,'deadline_unix':started+budget,'budget_seconds':budget})
    clock=json.loads((out/'budget.json').read_text())
    if not 0<clock['budget_seconds']<=p['collection_budget_seconds'] or clock['deadline_unix']!=clock['started_unix']+clock['budget_seconds']:
        raise ValueError('invalid persisted deadline')
    remaining=lambda:min(clock['deadline_unix']-time.time(),budget-(time.monotonic()-monotonic_start))
    inputs=json.loads((package/'inputs.json').read_text())['workloads'];raw=out/'measurements.jsonl'
    existing=[json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    if len(existing)>len(p['jobs']):raise ValueError('extra rows')
    rows=[];previous=None
    # Validate every persisted prefix row even when the old deadline expired.
    for i,row in enumerate(existing):
        job=p['jobs'][i]
        if check_row(row,job,inputs[job['workload_id']],p)!=row or row['runtime']!=context['runtime']:
            raise ValueError('persisted row identity differs')
        rows.append(row)
    try:
        for i in range(len(rows),len(p['jobs'])):
            job=p['jobs'][i]
            if remaining()<p['process_timeout_seconds']+30:
                base.write_json(out/'progress.json',{'state':'budget_exhausted','completed':len(rows),'total':len(p['jobs']),
                  'deadline_unix':clock['deadline_unix'],'resume_does_not_extend_budget':True})
                return False
            block=(job['workload_id'],job['repetition'])
            if block!=previous:
                telemetry(out,f'before_{i:05d}');previous=block
            # Telemetry also consumes the same wall budget.
            if remaining()<p['process_timeout_seconds']+5:
                base.write_json(out/'progress.json',{'state':'budget_exhausted','completed':len(rows),'total':len(p['jobs'])})
                return False
            logs=out
            if (out/f'stdout_{i:05d}.txt').exists():
                attempts=out/'attempts';attempts.mkdir(exist_ok=True)
                logs=attempts/f'{i:05d}-retry-{len(list(attempts.glob(str(i).zfill(5)+"-retry-*"))):03d}'
                logs.mkdir(exist_ok=False)
            text=base.execute_job(args_for(package,directory,p,job),job,logs,i,timeout=p['process_timeout_seconds'])
            row=check_row(json.loads(text),job,inputs[job['workload_id']],p)
            if row['runtime']!=context['runtime']:raise ValueError('runtime changed during collection')
            with raw.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row);base.write_json(out/'progress.json',{'state':'collecting','completed':len(rows),'total':len(p['jobs'])})
            print(f'{p["profile"]} {i+1}/{len(p["jobs"])} {job["workload_id"]} {job["family"]}',flush=True)
        base.write_json(out/'progress.json',{'state':'complete','completed':len(rows),'total':len(p['jobs'])})
        return True
    except BaseException as error:
        base.write_json(out/f'failure_after_{len(rows):05d}.json',{'completed':len(rows),'error_type':type(error).__name__,'error':str(error),
          'policy':'stop; retain failure without a timing label or automatic retry'})
        base.write_json(out/'progress.json',{'state':'failed','completed':len(rows),'total':len(p['jobs'])})
        raise
    finally:
        base.write_json(out/'summary.json',summarize(p,rows))


def pack_results(out,archive):
    out=out.resolve();archive=archive.resolve()
    if not out.is_dir() or archive.exists() or archive.is_relative_to(out):raise ValueError('fresh archive outside result directory required')
    if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('state') in ('running','collecting'):
        raise ValueError('stop collection before sealing a recovery archive')
    paths=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='recovery-manifest.json')
    if any(p.is_symlink() for p in out.rglob('*')):raise ValueError('result symlinks are not archived')
    base.write_json(out/'recovery-manifest.json',{'files':{str(p.relative_to(out)):sha256(p) for p in paths},
      'measurement_complete':json.loads((out/'summary.json').read_text()).get('complete',False) if (out/'summary.json').exists() else False})
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=out.name)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha256(archive)+'  '+archive.name+'\n')


def trace(package,directory,out,job_index,nsys):
    p,context,_=preflight(package,directory)
    if out.exists() or not 0<=job_index<len(p['jobs']):raise ValueError('fresh trace path and valid job index required')
    out.mkdir(parents=True);job=p['jobs'][job_index]
    command=[nsys,'profile','--trace=cuda,osrt','--sample=none','--cpuctxsw=none','--force-overwrite=false',
             '--output',out/'timeline',*args_for(package,directory,p,job)]
    base.write_json(out/'diagnostic.json',{'kind':'profiled_diagnostic_not_a_timing_label','profiler':True,'job':job,
      'context':context,'note':'external profiler overrides binary timing-policy field; never import this stdout as timing labels'})
    with (out/'profiled.stdout.txt').open('w') as a,(out/'profiled.stderr.txt').open('w') as b:
        base.command(command,timeout=180,stdout=a,stderr=b)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inventory');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('pack');a.add_argument('--out',type=Path,required=True);a.add_argument('--archive',type=Path,required=True)
    for name in ('build','preflight','run','trace'):
        a=sub.add_parser(name);a.add_argument('--package',type=Path,required=True);a.add_argument('--build-dir',type=Path,required=True)
        if name=='build':
            for flag in ('cutlass','nccl-source','nccl-include','nccl-library'):a.add_argument('--'+flag,type=Path,required=True)
            a.add_argument('--jobs',type=int,default=4)
        else:
            a.add_argument('--out',type=Path,required=True)
            if name=='run':a.add_argument('--resume',action='store_true');a.add_argument('--budget-seconds',type=int)
            if name=='trace':a.add_argument('--job-index',type=int,required=True);a.add_argument('--nsys',default='nsys')
    a=parser.parse_args()
    if a.command=='inventory':inventory(a.out.resolve())
    elif a.command=='pack':pack_results(a.out,a.archive)
    else:
        package=a.package.resolve();directory=a.build_dir.resolve()
        if a.command=='build':build(package,directory,cutlass=a.cutlass.resolve(),nccl_source=a.nccl_source.resolve(),
          nccl_include=a.nccl_include.resolve(),nccl_library=a.nccl_library.resolve(),jobs=a.jobs)
        elif a.command=='run':
            if not collect(package,directory,a.out.resolve(),resume=a.resume,budget_seconds=a.budget_seconds):raise SystemExit(75)
        elif a.command=='trace':trace(package,directory,a.out.resolve(),a.job_index,a.nsys)
        else:
            if a.out.exists():raise ValueError('fresh preflight output required')
            _,ctx,env=preflight(package,directory);base.write_json(a.out,{'context':ctx,'environment':env})
