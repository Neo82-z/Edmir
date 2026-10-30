#!/usr/bin/env python3
"""Freeze the seven-shape boundary diagnosis without accessing a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from prepare_selector_gpu import REPO, program_artifact, source_manifest, write_inputs, implementation_id
from independent_selector_validation import digest, WALL_TIMER
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from edmir.compiler.measurement_dataset import sha256

SCHEMA='edmir.boundary-gpu-package.v1'
FAMILIES=('host','direct','splitk2','splitk4','splitk8')
SHAPES=((4,512,4096,512),(4,64,4096,2048),(4,64,3072,512),
        (4,64,3072,1536),(4,192,3072,512),(4,192,3072,1536),(4,512,3072,512))


def validate_plan(plan):
    if (plan.get('schema')!='edmir.discriminating-measurement-plan.v1'
            or plan.get('baseline_id')!=PER_SOURCE_CUTLASS.baseline_id
            or plan.get('families')!=list(FAMILIES)
            or plan.get('anchor_shapes')!=[list(s) for s in SHAPES[:2]]
            or plan.get('new_shapes')!=[list(s) for s in SHAPES[2:]]
            or plan.get('configurations')!=35 or plan.get('process_runs')!=105):
        raise ValueError('expected the frozen seven-shape diagnosis')
    timing=plan['timing']
    if any(type(timing.get(k)) is not int or timing[k]!=v for k,v in
           [('warmup_iterations',20),('timed_iterations',100),('fresh_process_blocks',3),('process_timeout_seconds',120)]):
        raise ValueError('timing plan changed')
    runs=plan['run_order']
    if len(runs)!=105 or [r['ordinal'] for r in runs]!=list(range(105)):
        raise ValueError('run ordinals incomplete')
    for block in range(3):
        selected=[r for r in runs if r['block']==block]
        expected={(s,f) for s in SHAPES for f in FAMILIES}
        if len(selected)!=35 or {(tuple(r['shape']),r['family']) for r in selected}!=expected:
            raise ValueError('duplicate or missing block configuration')
    return plan


def targets(programs):
    lines=[]
    for name in programs:
        for suffix,source,library,guard in (
            ('_layout','tools/check_selector_layout.cpp','edmir_runtime',None),
            ('_api_trace','tests/cuda_api_trace_test.cpp','edmir_cuda_api_trace_runtime','EDMIR_CUDA_API_TRACE_TESTS'),
            ('','tools/benchmark_selector.cpp','edmir_cuda_runtime','EDMIR_ENABLE_CUDA')):
            target=name+suffix
            if guard:lines.append(f'if({guard})')
            lines += [f'add_executable({target} "${{CMAKE_CURRENT_SOURCE_DIR}}/{source}")',
                      f'target_include_directories({target} PRIVATE "${{EDMIR_EXPERIMENT_DIR}}/generated/{name}")',
                      f'target_link_libraries({target} PRIVATE {library})']
            if suffix:lines += [f'add_test(NAME {target} COMMAND {target})',f'set_tests_properties({target} PROPERTIES TIMEOUT 30)']
            if guard:lines.append('endif()')
    return '\n'.join(lines)+'\n'


def prepare(out,plan_path,*,with_inputs=False):
    plan=validate_plan(json.loads(plan_path.read_text()))
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    works=[{'id':f'w{i:03d}','shape':list(s)} for i,s in enumerate(SHAPES)]
    ids={tuple(w['shape']):w['id'] for w in works};programs={}
    for work in works:
        for family in FAMILIES:
            name=f'boundary_{work["id"]}_{family}'
            program_artifact(work['shape'],family).write_to(out/'generated'/name)
            programs[name]={'shape':work['shape'],'compiled_family':family,
                            'header':f'generated/{name}/edmir_program.hpp'}
    (out/'generated_targets.cmake').write_text(targets(programs))
    jobs=[]
    for row in plan['run_order']:
        w=ids[tuple(row['shape'])];family=row['family']
        jobs.append({'workload_id':w,'shape':row['shape'],'family':family,
                     'repetition':row['block'],'trial_order':row['ordinal'],'ordinal':row['ordinal'],
                     'executable':f'boundary_{w}_{family}','candidate_id':family+'@legacy',
                     'metadata_mode':'legacy','implementation_id':implementation_id(family,'legacy')})
    from freeze_boundary_predictions import freeze
    model_sources=freeze(out,SHAPES,FAMILIES)
    sources=source_manifest()
    sources.update({str(p.relative_to(REPO)):sha256(p) for p in model_sources})
    for path,expected in sources.items():
        source=REPO/path;dest=out/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if sha256(dest)!=expected:raise ValueError('source changed while freezing')
    p={'schema':SCHEMA,'mode':'boundary_diagnosis_development','status':'local_package_not_gpu_validated',
       'timing_scope':WALL_TIMER,'baseline_contracts':{'host':PER_SOURCE_CUTLASS.to_dict()},
       'measurement_policy':{'profiler':False,'nvtx':False,'discarded_samples':0},
       'warmup':20,'iterations':100,'repetitions':3,'process_timeout_seconds':120,
       'workloads':works,'programs':programs,'jobs':jobs,'sources':sources,'collector_sha256':digest(sources),
       'files':{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file() and 'source' not in f.relative_to(out).parts},
       'plan_sha256':sha256(out/'plan.json'),'source_root':'source','independent_validation':False,
       'hardware_contract':'A100-SXM4-80GB x4, SM80/108 SM, NV12 peer links',
       'full_candidate_oracle':False,'compiler_admission':False}
    (out/'package.json').write_text(json.dumps({'package_sha256':digest(p),'package':p},indent=2)+'\n')
    if with_inputs:prepare_inputs(out)
    return p


def verify_bundle(root,*,require_inputs=True):
    envelope=json.loads((root/'package.json').read_text());p=envelope['package']
    if p.get('schema')!=SCHEMA or digest(p)!=envelope['package_sha256']:
        raise ValueError('wrong boundary package schema or digest')
    contract={'mode':'boundary_diagnosis_development','timing_scope':WALL_TIMER,
              'baseline_contracts':{'host':PER_SOURCE_CUTLASS.to_dict()},
              'measurement_policy':{'profiler':False,'nvtx':False,'discarded_samples':0},
              'warmup':20,'iterations':100,'repetitions':3,'process_timeout_seconds':120,
              'source_root':'source','independent_validation':False,'full_candidate_oracle':False,'compiler_admission':False}
    if any(p.get(k)!=v for k,v in contract.items()) or p['collector_sha256']!=digest(p['sources']):
        raise ValueError('frozen measurement contract changed')
    plan=validate_plan(json.loads((root/'plan.json').read_text()))
    if sha256(root/'plan.json')!=p['plan_sha256']:raise ValueError('plan hash mismatch')
    for mapping,base in ((p['sources'],root/'source'),(p['files'],root)):
        for rel,expected in mapping.items():
            path=Path(rel)
            if path.is_absolute() or '..' in path.parts or not path.parts:raise ValueError('unsafe package path')
            if sha256(base/path)!=expected:raise ValueError(f'frozen content changed: {rel}')
    works=[{'id':f'w{i:03d}','shape':list(s)} for i,s in enumerate(SHAPES)]
    if p['workloads']!=works or len(p['programs'])!=35 or len(p['jobs'])!=105:
        raise ValueError('incomplete boundary package')
    expected_programs={f'boundary_{w["id"]}_{f}':{'shape':w['shape'],'compiled_family':f,
        'header':f'generated/boundary_{w["id"]}_{f}/edmir_program.hpp'} for w in works for f in FAMILIES}
    if p['programs']!=expected_programs:raise ValueError('program mapping changed')
    for job,row in zip(p['jobs'],plan['run_order']):
        expected_id=next(w['id'] for w in works if w['shape']==row['shape'])
        expected={'workload_id':expected_id,'shape':row['shape'],'family':row['family'],
                  'repetition':row['block'],'trial_order':row['ordinal'],'ordinal':row['ordinal'],
                  'executable':f'boundary_{expected_id}_{row["family"]}',
                  'candidate_id':row['family']+'@legacy','metadata_mode':'legacy',
                  'implementation_id':implementation_id(row['family'],'legacy')}
        if job!=expected:raise ValueError('execution order or identity changed')
    if require_inputs:
        info=json.loads((root/'inputs.json').read_text());seal=json.loads((root/'seal.json').read_text())
        if (info['package_sha256']!=digest(p) or seal!={'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')}):
            raise ValueError('input seal mismatch')
        if set(info['workloads'])!={w['id'] for w in works}:raise ValueError('missing input workloads')
        for w in works:
            entry=info['workloads'][w['id']];q,m,k,n=w['shape']
            if entry['shape']!=w['shape'] or entry['input_bytes']!=2*(q*q*m*k+q*k*n) or entry['reference_bytes']!=4*q*q*m*n:
                raise ValueError('input layout or lengths mismatch')
            for name,key,length in [('inputs.bin','input_sha256','input_bytes'),('reference.bin','reference_sha256','reference_bytes')]:
                path=root/'data'/w['id']/name
                if path.stat().st_size!=entry[length] or sha256(path)!=entry[key]:raise ValueError('input/reference bytes changed')
    return p


def prepare_inputs(root):
    p=verify_bundle(root,require_inputs=False)
    if (root/'inputs.json').exists() or (root/'data').exists():raise ValueError('input preparation requires fresh data paths')
    # Fresh subprocess per shape gives input generation a whole-process bound.
    from run_selector_gpu import command,write_json
    entries={}
    for w in p['workloads']:
        write_json(root/'preparation_progress.json',{'state':'generating_cpu_reference','workload':w})
        args=[sys.executable,Path(__file__),'one-input','--out',root/'data'/w['id'],'--shape',*[str(v) for v in w['shape']]]
        result=command(args,timeout=180,capture_output=True)
        entries[w['id']]=json.loads(result.stdout)
    write_json(root/'inputs.json',{'package_sha256':digest(p),'workloads':entries})
    write_json(root/'seal.json',{'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')})
    verify_bundle(root)
    write_json(root/'preparation_progress.json',{'state':'cpu_inputs_complete','gpu_validated':False,'workloads':len(entries)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare');make.add_argument('--out',type=Path,required=True);make.add_argument('--plan',type=Path,required=True);make.add_argument('--with-inputs',action='store_true')
    verify=sub.add_parser('verify');verify.add_argument('--package',type=Path,required=True)
    one=sub.add_parser('one-input');one.add_argument('--out',type=Path,required=True);one.add_argument('--shape',nargs=4,type=int,required=True)
    args=p.parse_args()
    if args.command=='one-input':print(json.dumps(write_inputs(args.shape,args.out)))
    else:
        obj=prepare(args.out,args.plan,with_inputs=args.with_inputs) if args.command=='prepare' else verify_bundle(args.package)
        print(json.dumps({'package_sha256':digest(obj),'programs':len(obj['programs']),'process_runs':len(obj['jobs']),'gpu_validated':False}))
