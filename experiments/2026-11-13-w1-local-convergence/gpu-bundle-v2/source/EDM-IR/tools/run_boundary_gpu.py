#!/usr/bin/env python3
"""Bounded build, collection, and separate profiling for the frozen diagnosis."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from prepare_boundary_gpu import verify_bundle, FAMILIES, SHAPES
from prepare_selector_gpu import REPO
from independent_selector_validation import digest
from edmir.compiler.measurement_dataset import sha256
import run_selector_gpu as base


def require_frozen_runner(package):
    if REPO.resolve()!=(package/'source').resolve():
        raise ValueError('execute package/source/EDM-IR/tools/run_boundary_gpu.py so imports and build use the frozen source')


def build(package,directory,**dependencies):
    require_frozen_runner(package);verify_bundle(package)
    base.build(package,directory,**dependencies)
    # Capture actual compiler tools as part of the build receipt, before run.
    path=directory/'selector_build.json';receipt=json.loads(path.read_text())
    receipt['toolchain']={name:base.command(args,capture_output=True,timeout=60).stdout
                          for name,args in [('nvcc',['nvcc','--version']),('cxx',['c++','--version']),('cmake',['cmake','--version'])]}
    receipt['boundary_build_complete']=True
    base.write_json(path,receipt)


def preflight(package,directory):
    require_frozen_runner(package);p=verify_bundle(package)
    timing=base.measurement_environment(os.environ)
    receipt=json.loads((directory/'selector_build.json').read_text())
    if receipt['package_sha256']!=digest(p) or set(receipt['binaries'])!=set(p['programs']):
        raise ValueError('build receipt does not cover this package')
    if receipt.get('boundary_build_complete') is not True or set(receipt.get('toolchain',{}))!={'nvcc','cxx','cmake'}:
        raise ValueError('boundary build/toolchain receipt is incomplete')
    for path,value in receipt['dependencies'].items():
        if sha256(Path(path))!=value:raise ValueError('build dependency changed')
    libraries={}
    for name,value in receipt['binaries'].items():
        if sha256(directory/name)!=value:raise ValueError('executable changed')
        linked=base.command(['ldd',directory/name],capture_output=True,timeout=60).stdout
        if 'not found' in linked:raise ValueError('missing runtime library')
        nccl=[]
        for lib,path in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',linked,re.MULTILINE):
            resolved=str(Path(path).resolve())
            if resolved not in libraries:libraries[resolved]=sha256(Path(resolved))
            if lib.startswith('libnccl.so'):nccl.append(resolved)
        if len(nccl)!=1 or libraries[nccl[0]]!=receipt['nccl_library_sha256']:
            raise ValueError('loaded NCCL differs from build receipt')
    topo=base.command(['nvidia-smi','topo','-m'],capture_output=True,timeout=60).stdout
    inventory=base.command(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'],capture_output=True,timeout=60).stdout
    uuids=base.command(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],capture_output=True,timeout=60).stdout
    runtime=json.loads(base.command([directory/p['jobs'][0]['executable'],'--inspect'],capture_output=True,timeout=60).stdout)
    environment=base.environment(runtime,topo,inventory,4)
    context={'package_sha256':digest(p),'build_sha256':sha256(directory/'selector_build.json'),
             'topology':topo,'inventory':inventory,'device_uuid_inventory':uuids,'runtime':runtime,
             'loaded_library_sha256':libraries,'measurement_environment':timing,
             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    environment.update(receipt['environment']);environment['collector_sha256']=p['collector_sha256']
    environment['loaded_library_sha256']=libraries
    context['environment_id']='boundary-a100-p4-'+digest(context)[:20]
    return p,context,environment


def check_row(row,job,info,p):
    checked=base.validate_measurement(row,job,info,p)
    q=1 if job['family'] in ('host','direct') else int(job['family'].removeprefix('splitk'))
    algorithm='sm80_64x128x32'+(f'_splitk{q}' if q>1 else '')
    if row['gemm_algorithm_id']!=algorithm:raise ValueError('GEMM algorithm differs from the generated family')
    return {**checked,'ordinal':job['ordinal']}


def summarize(p,rows):
    """Paired process-block contrasts; no significance or full-oracle claim."""
    expected={(tuple(j['shape']),j['family'],j['repetition']) for j in p['jobs']}
    index={}
    for r in rows:
        key=(tuple(r['shape']),r['family'],r['repetition'])
        if key not in expected or key in index:raise ValueError('duplicate or unexpected measurement')
        samples=r['samples_ms']
        if len(samples)!=p['iterations'] or any(not math.isfinite(v) or v<=0 for v in samples):raise ValueError('invalid timing samples')
        if not math.isclose(statistics.median(samples),r['median_ms'],rel_tol=1e-12):raise ValueError('median changed')
        index[key]=r
    if set(index)!=expected:raise ValueError('missing process observations')
    workloads=[]
    for work in p['workloads']:
        s=tuple(work['shape']);c={}
        for f in FAMILIES:
            medians=[index[(s,f,b)]['median_ms'] for b in range(3)]
            hosts=[index[(s,'host',b)]['median_ms'] for b in range(3)]
            c[f]={'process_medians_ms':medians,'median_of_process_medians_ms':statistics.median(medians),
                  'paired_fused_over_host':[v/h for v,h in zip(medians,hosts)]}
        workloads.append({'shape':list(s),'candidates':c,'best_of_measured_subset':min(c,key=lambda f:c[f]['median_of_process_medians_ms'])})
    contrast=[]
    for f in FAMILIES:
        val=lambda m,n,b:index[((4,m,3072,n),f,b)]['median_ms']
        contrast.append({'family':f,
             'same_grid_M192_N512_minus_M64_N1536_ms':[val(192,512,b)-val(64,1536,b) for b in range(3)],
             'crossed_interaction_ms':[val(192,1536,b)-val(192,512,b)-val(64,1536,b)+val(64,512,b) for b in range(3)]})
    return {'status':'development diagnosis only; no automatic model promotion','workloads':workloads,'paired_contrasts':contrast,
            'unit_of_repetition':'fresh process block; iteration samples are not independent repetitions',
            'full_candidate_oracle':False,'independent_validation':False}


def measurement_args(package,directory,p,job):
    data=package/'data'/job['workload_id']
    return [directory/job['executable'],job['family'],data/'inputs.bin',data/'reference.bin',p['warmup'],p['iterations'],'legacy']


def telemetry(out,label):
    """Outside measurement processes; record clocks/load without changing them."""
    result=base.command(['nvidia-smi','-q','-x'],capture_output=True,timeout=60)
    (out/f'telemetry_{label}.xml').write_text(result.stdout)


def collect(package,directory,out,*,resume=False):
    p,context,env=preflight(package,directory)
    inputs=json.loads((package/'inputs.json').read_text())['workloads']
    if out.exists() and not resume:raise ValueError('use a new output directory, or explicitly resume the same run')
    if resume and not (out/'context.json').exists():raise ValueError('resume requires an existing run context')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'context.json').exists() and json.loads((out/'context.json').read_text())!=context:
        raise ValueError('resume environment, package or build differs')
    base.write_json(out/'context.json',context)
    raw=out/'measurements.jsonl'
    existing=[json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    if len(existing)>len(p['jobs']):raise ValueError('extra previous measurements')
    rows=[]
    for i,job in enumerate(p['jobs']):
        try:
            if i<len(existing):
                row=check_row(existing[i],job,inputs[job['workload_id']],p)
                if row!=existing[i]:raise ValueError('saved measurement provenance differs')
            else:
                if i==len(existing) or job['repetition']!=p['jobs'][i-1]['repetition']:
                    telemetry(out,f'before_{i:05d}')
                output=base.execute_job(measurement_args(package,directory,p,job),job,out,i,timeout=p['process_timeout_seconds'])
                row=check_row(json.loads(output),job,inputs[job['workload_id']],p)
            if row['runtime']!=context['runtime']:raise ValueError('runtime identity changed after preflight')
            base.environment(row['runtime'],context['topology'],context['inventory'],4)
            if i>=len(existing):
                with raw.open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            rows.append(row)
            if i>=len(existing) and (i+1==len(p['jobs']) or p['jobs'][i+1]['repetition']!=job['repetition']):
                telemetry(out,f'after_{i:05d}')
            base.write_json(out/'progress.json',{'state':'collecting','completed':i+1,'total':len(p['jobs'])})
        except BaseException as error:
            failure={'job_index':i,'job':job,'reason':type(error).__name__,'detail':str(error)}
            base.write_json(out/'failure.json',failure);base.write_json(out/'progress.json',{'state':'failed',**failure});raise
    summary=summarize(p,rows);raw_hash=sha256(raw)
    run={'schema':'edmir.boundary-diagnosis-run.v1','package_sha256':digest(p),'plan_sha256':p['plan_sha256'],
         'environment_id':context['environment_id'],'environment':env,'context_sha256':sha256(out/'context.json'),
         'timing_scope':p['timing_scope'],'baseline_contracts':p['baseline_contracts'],'dtype':'float16','latency_unit':'ms',
         'fresh_process_per_repetition':True,'independent_validation':False,'compiler_admission':False,
         'rows':[{**r,'source':{'path':raw.name,'sha256':raw_hash,'line':i+1}} for i,r in enumerate(rows)]}
    base.write_json(out/'run.json',run);base.write_json(out/'summary.json',summary)
    base.write_json(out/'progress.json',{'state':'complete','completed':len(rows),'total':len(p['jobs'])})
    return summary


def trace(package,directory,out,*,job_index,nsys):
    p,context,_=preflight(package,directory)
    if not 0<=job_index<len(p['jobs']):raise ValueError('job index out of range')
    if out.exists():raise ValueError('profiler output requires a fresh directory')
    out.mkdir(parents=True)
    job=p['jobs'][job_index]
    version=base.command([nsys,'--version'],capture_output=True,timeout=30).stdout
    args=[nsys,'profile','--trace=cuda,osrt','--sample=none','--cpuctxsw=none',
          '--force-overwrite=false','--output',out/'timeline',*measurement_args(package,directory,p,job)]
    receipt={'kind':'profiled_diagnostic_not_a_timing_label','profiler':True,'job':job,'context':context,
             'nsys_version':version,'command':[str(v) for v in args],
             'note':'Binary stdout self-reports its uninstrumented policy; this external profiler receipt overrides that policy. Never import stdout as a primary measurement.'}
    base.write_json(out/'diagnostic.json',receipt)
    try:
        with (out/'profiled.stdout.txt').open('w') as stdout,(out/'profiled.stderr.txt').open('w') as stderr:
            base.command(args,timeout=180,stdout=stdout,stderr=stderr)
        reports=list(out.glob('*.nsys-rep'))
        if not reports:raise ValueError('Nsight did not produce an nsys-rep')
        receipt['reports']={r.name:sha256(r) for r in reports};receipt['state']='complete'
    except BaseException as error:
        receipt.update(state='failed',error=str(error));base.write_json(out/'diagnostic.json',receipt);raise
    base.write_json(out/'diagnostic.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('build','run','preflight','trace'):
        p=sub.add_parser(name);p.add_argument('--package',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True)
        if name=='build':
            for flag in ('cutlass','nccl-source','nccl-include','nccl-library'):p.add_argument('--'+flag,type=Path,required=True)
            p.add_argument('--jobs',type=int,default=4)
        else:
            p.add_argument('--out',type=Path,required=True)
            if name=='run':p.add_argument('--resume',action='store_true')
            if name=='trace':p.add_argument('--job-index',type=int,required=True);p.add_argument('--nsys',default='nsys')
    args=parser.parse_args();package=args.package.resolve();directory=args.build_dir.resolve()
    if args.command=='build':
        if not 1<=args.jobs<=16:parser.error('build jobs must be between 1 and 16')
        build(package,directory,cutlass=args.cutlass.resolve(),nccl_source=args.nccl_source.resolve(),
              nccl_include=args.nccl_include.resolve(),nccl_library=args.nccl_library.resolve(),jobs=args.jobs)
    elif args.command=='run':collect(package,directory,args.out.resolve(),resume=args.resume)
    elif args.command=='trace':trace(package,directory,args.out.resolve(),job_index=args.job_index,nsys=args.nsys)
    else:
        if args.out.exists():raise ValueError('preflight receipt already exists')
        _,context,env=preflight(package,directory);base.write_json(args.out,{'context':context,'environment':env})
