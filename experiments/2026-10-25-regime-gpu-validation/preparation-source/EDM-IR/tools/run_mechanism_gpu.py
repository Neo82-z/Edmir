#!/usr/bin/env python3
"""Staged, bounded collection from a frozen package. Never rent a server."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import time
from prepare_mechanism_gpu import verify_bundle, STAGES, CUTLASS_REVISION, NCCL_REVISION
from prepare_selector_gpu import REPO
from independent_selector_validation import digest
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.two_stage_selection import parse_schedule_family
import run_selector_gpu as base
from run_boundary_gpu import telemetry


def require_frozen_runner(package):
    if REPO.resolve()!=(package/'source').resolve():
        raise ValueError('execute package/source/EDM-IR/tools/run_mechanism_gpu.py with the frozen source')


def plain_topology(value):
    # Recent nvidia-smi drivers underline their header even without a TTY.
    # Strip terminal styling only; keep topology cells and the raw evidence.
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)

def build(package, directory, cutlass, nccl_source, nccl_include, nccl_library, jobs):
    require_frozen_runner(package)
    p = verify_bundle(package)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/"selector_build.json"
    if receipt.exists():
        raise ValueError("build receipt exists; use a new directory for a new build")
    # Record source identities plus the actual headers/library, not just names.
    versions = (cutlass/"include/cutlass/version.h").read_text()
    version = ".".join(re.search(rf"#define CUTLASS_{part}\s+(\d+)", versions).group(1)
                       for part in ("MAJOR", "MINOR", "PATCH"))
    identities = {"cutlass_commit": base.git_commit(cutlass), "cutlass_version": version,
                  "nccl_commit": base.git_commit(nccl_source)}
    if identities['cutlass_commit'] != CUTLASS_REVISION or identities['nccl_commit'] != NCCL_REVISION:
        raise ValueError('use the source revisions frozen in plan.json')
    for source in (cutlass, nccl_source):
        if base.command(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
                   capture_output=True).stdout.strip():
            raise ValueError(f"dependency has tracked modifications; record a clean commit first: {source}")
    dependencies = {str(nccl_library): sha256(nccl_library),
                    str(nccl_include/"nccl.h"): sha256(nccl_include/"nccl.h"),
                    str(nccl_include/"nccl_device.h"): sha256(nccl_include/"nccl_device.h")}
    base.command(["cmake", "-S", REPO/"EDM-IR/runtime", "-B", directory, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_FLAGS=--ptxas-options=-v",
             "-DEDMIR_ENABLE_CUDA=ON", "-DEDMIR_ENABLE_CUTLASS=ON", "-DEDMIR_ENABLE_NCCL_DEVICE=ON",
             f"-DEDMIR_CUTLASS_DIR={cutlass}", f"-DEDMIR_EXPERIMENT_DIR={package}",
             f"-DNCCL_INCLUDE_DIR={nccl_include}", f"-DNCCL_LIBRARY={nccl_library}",
             f"-DCMAKE_BUILD_RPATH={nccl_library.parent}"], timeout=300)
    with (directory/'build.stdout.txt').open('w') as stdout, (directory/'build.stderr.txt').open('w') as stderr:
        base.command(["cmake", "--build", directory, "-j", jobs], timeout=1800, stdout=stdout, stderr=stderr)
    base.command(["ctest", "--test-dir", directory, "--output-on-failure", "--timeout", "30"], timeout=600)
    toolchain = {name:base.command(args,capture_output=True,timeout=60).stdout
                 for name,args in [('nvcc',['nvcc','--version']),('cxx',['c++','--version']),('cmake',['cmake','--version'])]}
    binaries = {name: sha256(directory/name) for name in p["programs"]}
    base.write_json(receipt, {"package_sha256": digest(p), "dependencies": dependencies,
                        "environment": identities, "toolchain": toolchain, "mechanism_build_complete": True, "binaries": binaries,
                        "nccl_library_sha256": sha256(nccl_library),
                        "nccl_source": str(nccl_source), "cutlass_source": str(cutlass),
                        "note": "Source commits and linked library hash recorded; verify the library was built from this NCCL source."})


def preflight(package,directory):
    require_frozen_runner(package);p=verify_bundle(package)
    timing=base.measurement_environment(os.environ)
    receipt=json.loads((directory/'selector_build.json').read_text())
    if receipt['package_sha256']!=digest(p) or set(receipt['binaries'])!=set(p['programs']):
        raise ValueError('build receipt does not cover this package')
    if receipt.get('mechanism_build_complete') is not True or set(receipt.get('toolchain',{}))!={'nvcc','cxx','cmake'}:
        raise ValueError('mechanism build/toolchain receipt is incomplete')
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
    raw_topo=base.command(['nvidia-smi','topo','-m'],capture_output=True,timeout=60).stdout
    topo=plain_topology(raw_topo)
    inventory=base.command(['nvidia-smi','--query-gpu=index,pci.bus_id','--format=csv,noheader'],capture_output=True,timeout=60).stdout
    uuids=base.command(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],capture_output=True,timeout=60).stdout
    runtime=json.loads(base.command([directory/p['jobs'][0]['executable'],'--inspect'],capture_output=True,timeout=60).stdout)
    environment=base.environment(runtime,topo,inventory,4)
    context={'package_sha256':digest(p),'build_sha256':sha256(directory/'selector_build.json'),
             'topology':topo,'raw_topology':raw_topo,'inventory':inventory,'device_uuid_inventory':uuids,'runtime':runtime,
             'loaded_library_sha256':libraries,'measurement_environment':timing,
             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    environment.update(receipt['environment']);environment['collector_sha256']=p['collector_sha256']
    environment['loaded_library_sha256']=libraries
    context['environment_id']='mechanism-a100-p4-'+digest(context)[:20]
    return p,context,environment


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


def inventory(out):
    """Read-only check before CUDA compilation."""
    if out.exists(): raise ValueError('choose a fresh inventory path')
    records = {}
    for name,args in (
        ('gpu',['nvidia-smi','-q','-x']),('topology',['nvidia-smi','topo','-m']),
        ('peer',['nvidia-smi','topo','-p2p','r']),('cpu',['lscpu']),('ram',['free','-b']),
        ('nvcc',['nvcc','--version']),('cxx',['c++','--version']),('cmake',['cmake','--version']),
        ('nsys',['nsys','--version'])):
        try:
            r=base.command(args,capture_output=True,timeout=30,check=False)
            records[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except OSError as e: records[name]={'unavailable':str(e)}
    base.write_json(out,{'platform':platform.platform(),'cpu_count':os.cpu_count(),
        'disk_bytes':shutil.disk_usage(out.parent)._asdict(),'records':records,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_validated':False,'note':'Inventory only; full runtime/P2P check follows compilation.'})


def measurement_args(package,directory,p,job):
    data=package/'data'/job['workload_id']
    return [directory/job['executable'],job['family'],data/'inputs.bin',data/'reference.bin',
            job.get('diagnostic_warmup',p['warmup']),p['iterations'],job['metadata_mode']]


def check_row(row,job,info,p):
    trial={**p,'warmup':job['diagnostic_warmup']} if job.get('diagnostic_warmup') else p
    result=base.validate_measurement(row,job,info,trial)
    family=job['family']; algorithm=row['gemm_algorithm_id']
    if family=='host_compiler_cublas':
        expected='library_default'
    elif family=='host_merged_cublaslt':
        expected=None
        for rank in range(job['shape'][0]):
            if not re.search(rf'rank{rank}:(?:\d+=-?\d+;)+workspace=\d+;budget=33554432\|',algorithm):
                raise ValueError('missing cuBLASLt algorithm details')
    elif family=='host_merged_cutlass':
        expected='sm80_64x128x32'
    else:
        q,_=(1,1) if family=='host' else parse_schedule_family(family)
        expected='sm80_64x128x32'+(f'_splitk{q}' if q>1 else '')
    if expected is not None and algorithm!=expected: raise ValueError('GEMM algorithm differs')
    return {**result,'stage':job['stage'],'ordinal':job['ordinal']}


def summarize(p,rows,stage):
    jobs=[j for j in p['jobs'] if j['stage']==stage]
    key=lambda r:(tuple(r['shape']),r['candidate_id'],r['repetition'])
    index={key(r):r for r in rows}
    if len(index)!=len(rows) or set(index)!={key(j) for j in jobs}: raise ValueError('incomplete/duplicate stage')
    works=[]
    for w in p['workloads']:
        candidates=sorted({j['candidate_id'] for j in jobs if j['workload_id']==w['id']})
        if not candidates: continue
        measured={c:[index[(tuple(w['shape']),c,b)]['median_ms'] for b in range(3)] for c in candidates}
        contrasts=[]
        from itertools import combinations
        pairs=list(combinations(candidates,2))
        for a,b in pairs:
            if a in measured and b in measured:
                contrasts.append({'a':a,'b':b,'b_minus_a_us':[1000*(y-x) for x,y in zip(measured[a],measured[b])],
                                  'b_over_a':[y/x for x,y in zip(measured[a],measured[b])]})
        means={c:statistics.median(v) for c,v in measured.items()}
        works.append({**w,'process_medians_ms':measured,'median_of_process_medians_ms':means,
            'best_of_measured_subset':min(means,key=means.get),'paired_contrasts':contrasts,
            'process_max_over_min':{c:max(v)/min(v) for c,v in measured.items()},
            'review_drift_candidates':[c for c,v in measured.items() if max(v)/min(v)>1.10]})
    geometry=[]
    if stage=='geometry':
        for c in works[0]['process_medians_ms']:
            val=lambda m,n,b:index[((4,m,3072,n),c,b)]['median_ms']
            geometry.append({'candidate_id':c,
                'equal_tile_M192_N512_minus_M64_N1536_us':[1000*(val(192,512,b)-val(64,1536,b)) for b in range(3)],
                'crossed_interaction_us':[1000*(val(192,1536,b)-val(192,512,b)-val(64,1536,b)+val(64,512,b)) for b in range(3)]})
    return {'stage':stage,'workloads':works,'geometry_contrasts':geometry,
        'independent_validation':False,'compiler_admission':False,'full_candidate_oracle':False,
        'unit_of_repetition':'fresh process block; iterations are not independent repeats',
        'drift_rule':'max/min > 1.10 requests review; all observations retained; not a confidence bound',
        'causal_limit':'Timing contrasts do not identify physical traffic or semaphore stall time.'}


def collect(package,directory,out,*,stage,resume=False,budget_seconds=900):
    if stage not in STAGES or not math.isfinite(budget_seconds) or budget_seconds<=0:
        raise ValueError('invalid stage or time budget')
    start=time.monotonic()
    p,context,env=preflight(package,directory)
    context={**context,'stage':stage}
    inputs=json.loads((package/'inputs.json').read_text())['workloads']
    jobs=[j for j in p['jobs'] if j['stage']==stage]
    if out.exists() and not resume: raise ValueError('use fresh output or explicit --resume')
    if resume and not (out/'context.json').exists(): raise ValueError('resume requires existing context')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'context.json').exists() and json.loads((out/'context.json').read_text())!=context:
        raise ValueError('resume context differs')
    base.write_json(out/'context.json',context)
    raw=out/'measurements.jsonl'
    existing=[json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() else []
    if len(existing)>len(jobs): raise ValueError('extra previous measurements')
    rows=[]; previous_block=None
    for i,job in enumerate(jobs):
        info=inputs[job['workload_id']]
        if i<len(existing):
            row=existing[i]
            if check_row(row,job,info,p)!=row: raise ValueError('previous row provenance differs')
        else:
            if budget_seconds-(time.monotonic()-start)<p['process_timeout_seconds']+60:
                base.write_json(out/'progress.json',{'state':'budget_exhausted','completed':i,'total':len(jobs),
                    'stage':stage,'elapsed_seconds':time.monotonic()-start,'resume_supported':True})
                return False
            if previous_block!=job['repetition']:
                telemetry(out,f'b{job["repetition"]}_before_{i:05d}')
                previous_block=job['repetition']
            output=base.execute_job(measurement_args(package,directory,p,job),job,out,i,
                                    timeout=p['process_timeout_seconds'])
            try:
                row=check_row(json.loads(output),job,info,p)
                if row['runtime']!=context['runtime']: raise ValueError('runtime differs from preflight')
            except (ValueError,KeyError,TypeError) as e:
                base.write_json(out/'failure.json',{'job':job,'job_index':i,'reason':str(e)})
                base.write_json(out/'progress.json',{'state':'failed','completed':i,'total':len(jobs)})
                raise
            with raw.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
            print(f'{stage} {i+1}/{len(jobs)} {job["workload_id"]} {job["candidate_id"]}',flush=True)
        if row['runtime']!=context['runtime']: raise ValueError('row runtime differs')
        rows.append(row)
        base.write_json(out/'progress.json',{'state':'collecting','completed':i+1,'total':len(jobs),'stage':stage})
    telemetry(out,'complete')
    base.write_json(out/'run.json',{'schema':'edmir.mechanism-run.v1','stage':stage,
        'context':context,'environment':env,'raw_sha256':sha256(raw),'rows':rows,
        'independent_validation':False,'compiler_admission':False})
    base.write_json(out/'summary.json',summarize(p,rows,stage))
    base.write_json(out/'progress.json',{'state':'complete','completed':len(rows),'total':len(jobs),'stage':stage,
        'elapsed_seconds':time.monotonic()-start})
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    inv=sub.add_parser('inventory');inv.add_argument('--out',type=Path,required=True)
    for name in ('build','preflight','run','trace'):
        a=sub.add_parser(name);a.add_argument('--package',type=Path,required=True);a.add_argument('--build-dir',type=Path,required=True)
        if name=='build':
            for flag in ('cutlass','nccl-source','nccl-include','nccl-library'):a.add_argument('--'+flag,type=Path,required=True)
            a.add_argument('--jobs',type=int,default=4)
        else:
            a.add_argument('--out',type=Path,required=True)
            if name=='run':
                a.add_argument('--stage',choices=STAGES,required=True);a.add_argument('--resume',action='store_true')
                a.add_argument('--budget-seconds',type=int,default=900)
            if name=='trace': a.add_argument('--job-index',type=int,required=True);a.add_argument('--nsys',default='nsys')
    args=parser.parse_args()
    if args.command=='inventory': inventory(args.out.resolve())
    else:
        package=args.package.resolve();directory=args.build_dir.resolve()
        if args.command=='build':
            if not 1<=args.jobs<=16: parser.error('build jobs must be 1..16')
            build(package,directory,cutlass=args.cutlass.resolve(),nccl_source=args.nccl_source.resolve(),
                  nccl_include=args.nccl_include.resolve(),nccl_library=args.nccl_library.resolve(),jobs=args.jobs)
        elif args.command=='run':
            if not collect(package,directory,args.out.resolve(),stage=args.stage,resume=args.resume,budget_seconds=args.budget_seconds):
                raise SystemExit(75)
        elif args.command=='trace': trace(package,directory,args.out.resolve(),job_index=args.job_index,nsys=args.nsys)
        else:
            if args.out.exists(): raise ValueError('fresh preflight path required')
            _,ctx,env=preflight(package,directory);base.write_json(args.out,{'context':ctx,'environment':env})
