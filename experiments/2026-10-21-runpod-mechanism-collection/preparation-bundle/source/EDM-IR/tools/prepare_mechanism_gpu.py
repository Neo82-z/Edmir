#!/usr/bin/env python3
"""Freeze staged mechanism measurements; no fitting and no device access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

from prepare_selector_gpu import (REPO, PILOT_MKN, MERGED_FAMILIES, SEED,
                                  program_artifact, source_manifest, implementation_id, write_inputs)
from prepare_boundary_gpu import targets
from independent_selector_validation import BASELINES, WALL_TIMER, digest
from edmir.compiler.execution_ledger import execution_ledger, FAMILIES, CUTLASS_REVISION
from edmir.compiler.measurement_dataset import sha256

SCHEMA = 'edmir.mechanism-gpu-package.v1'
NCCL_REVISION = 'dbc86fd06e8b0c4517b95d8958a09ccacf9520c9'
SHAPES = tuple((4, *s) for s in PILOT_MKN) + ((4,512,4096,512),
    (4,64,3072,512),(4,64,3072,1536),(4,192,3072,512),(4,192,3072,1536),(4,512,3072,512))
STAGES = ('anchors', 'mechanisms', 'geometry')
POLICY = {'profiler':False, 'nvtx':False, 'discarded_samples':0}


def configurations():
    result = []
    def add(stage, wi, families, mode):
        for family in families:
            result.append({'stage':stage, 'workload_id':f'w{wi:03d}', 'shape':list(SHAPES[wi]),
                'family':family, 'candidate_id':family+'@'+mode, 'metadata_mode':mode,
                'implementation_id':implementation_id(family,mode),
                'executable':f'mechanism_w{wi:03d}_'+('host' if family in MERGED_FAMILIES else family)})
    # First establish the two error anchors using the historical implementation.
    for wi in (1,4):
        add('anchors',wi,('host','direct','splitk2','splitk4','splitk8'),'legacy')
    # Both metadata modes and every comparator are interleaved in each block.
    for wi in range(5):
        add('mechanisms',wi,('host','splitk2','splitk4'),'legacy')
        add('mechanisms',wi,('host','splitk2','splitk4','splitk2_groupedn2',
                             'splitk4_groupedn2','host_compiler_cublas'),'prepared')
        add('mechanisms',wi,MERGED_FAMILIES,'merged')
        if wi in (2,3):
            add('mechanisms',wi,('splitk8','splitk8_groupedn2'),'prepared')
    # Optional crossed geometry, only after the five priorities are collected.
    for wi in range(5,10):
        add('geometry',wi,('host','splitk2','splitk4','splitk2_groupedn2','splitk4_groupedn2'),'prepared')
    return result


def plan():
    configs = configurations(); jobs = []; rng = random.Random(SEED+1)
    for stage in STAGES:
        for block in range(3):
            group = [c for c in configs if c['stage']==stage]
            rng.shuffle(group)
            for c in group:
                jobs.append({**c, 'repetition':block, 'trial_order':len(jobs), 'ordinal':len(jobs)})
    return {'schema':'edmir.mechanism-plan.v1', 'configurations':configs, 'jobs':jobs,
            'stages':{s:{'configurations':sum(c['stage']==s for c in configs),
                         'process_runs':sum(j['stage']==s for j in jobs),
                         'optional':s=='geometry'} for s in STAGES},
            'hardware':'4x A100-SXM4-80GB; SM80/108 SM; full pairwise NV12/P2P',
            'cutlass_revision':CUTLASS_REVISION, 'nccl_revision':NCCL_REVISION,
            'timing_scope':WALL_TIMER, 'warmup':20, 'iterations':100, 'repetitions':3,
            'process_timeout_seconds':120, 'measurement_policy':POLICY,
            'new_training':False, 'independent_validation':False, 'compiler_admission':False}


def mappings():
    works = [{'id':f'w{i:03d}', 'shape':list(s)} for i,s in enumerate(SHAPES)]
    programs = {}
    for c in configurations():
        name = c['executable']
        programs[name] = {'shape':c['shape'], 'compiled_family':(
            'host' if c['family'] in MERGED_FAMILIES else c['family']),
            'header':f'generated/{name}/edmir_program.hpp'}
    return works, programs


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare(out):
    out = out.resolve(); out.mkdir(parents=True,exist_ok=False)
    design = plan(); works, programs = mappings()
    write(out/'plan.json',design)
    for name,c in programs.items():
        program_artifact(c['shape'],c['compiled_family']).write_to(out/'generated'/name)
    (out/'generated_targets.cmake').write_text(targets(programs))
    ledgers = {}
    for c in configurations():
        key = c['workload_id']+'_'+c['candidate_id']
        if c['family'] in FAMILIES:
            ledgers[key] = execution_ledger(c['shape'],c['family'],
                environment_id='planned-mechanism-a100-p4-not-measured', metadata_mode=c['metadata_mode'])
        else:
            ledgers[key] = {'status':'no_execution_ledger_adapter', 'implementation_id':c['implementation_id'],
                            'latency_us':None, 'note':'Measured separately; never inherit the CUTLASS per-source ledger.'}
    write(out/'execution_accounts.json',ledgers)
    sources = source_manifest()
    # Include relevant tests in the movable source snapshot.
    for name in ('mechanism_gpu_package_test.py','selector_runner_safety_test.py','execution_evidence_test.py'):
        path = REPO/'EDM-IR/tests'/name
        if path.exists(): sources[str(path.relative_to(REPO))]=sha256(path)
    for rel, value in sources.items():
        dest=out/'source'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/rel,dest)
        if sha256(dest)!=value: raise ValueError('source changed during freeze')
    p = {'schema':SCHEMA, 'mode':'mechanism_diagnosis_development',
         'status':'local_package_not_gpu_validated', 'workloads':works, 'programs':programs,
         'jobs':design['jobs'], 'stages':design['stages'], 'sources':sources,
         'collector_sha256':digest(sources), 'plan_sha256':sha256(out/'plan.json'),
         'baseline_contracts':{n:b.to_dict() for n,b in BASELINES.items()},
         **{k:design[k] for k in ('timing_scope','warmup','iterations','repetitions','process_timeout_seconds',
                                  'measurement_policy','new_training','independent_validation','compiler_admission')},
         'files':{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*'))
                  if f.is_file() and 'source' not in f.relative_to(out).parts}}
    write(out/'package.json',{'package_sha256':digest(p),'package':p})
    return p


def verify_bundle(root, *, require_inputs=True):
    obj=json.loads((root/'package.json').read_text()); p=obj['package']; design=plan()
    works,programs=mappings()
    if p.get('schema')!=SCHEMA or digest(p)!=obj['package_sha256']: raise ValueError('package digest/schema differs')
    expected={'mode':'mechanism_diagnosis_development','workloads':works,'programs':programs,
              'jobs':design['jobs'],'stages':design['stages'],
              'baseline_contracts':{n:b.to_dict() for n,b in BASELINES.items()},
              **{k:design[k] for k in ('timing_scope','warmup','iterations','repetitions','process_timeout_seconds',
                                     'measurement_policy','new_training','independent_validation','compiler_admission')}}
    if any(p.get(k)!=v for k,v in expected.items()): raise ValueError('frozen measurement contract changed')
    if json.loads((root/'plan.json').read_text())!=design or sha256(root/'plan.json')!=p['plan_sha256']:
        raise ValueError('plan changed')
    if p['collector_sha256']!=digest(p['sources']): raise ValueError('source identity changed')
    for mapping,base in ((p['sources'],root/'source'),(p['files'],root)):
        for rel,value in mapping.items():
            path=Path(rel)
            if path.is_absolute() or '..' in path.parts or not path.parts: raise ValueError('unsafe package path')
            if sha256(base/path)!=value: raise ValueError('frozen content changed: '+rel)
    if require_inputs:
        info=json.loads((root/'inputs.json').read_text()); seal=json.loads((root/'seal.json').read_text())
        if info['package_sha256']!=digest(p) or seal!={'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')}:
            raise ValueError('input seal mismatch')
        if set(info['workloads'])!={w['id'] for w in works}: raise ValueError('missing input workload')
        for w in works:
            entry=info['workloads'][w['id']]; P,M,K,N=w['shape']
            if entry['shape']!=w['shape'] or entry['input_seed']!=SEED: raise ValueError('input identity differs')
            for name,key,size in [('inputs.bin','input_sha256',2*(P*P*M*K+P*K*N)),
                                  ('reference.bin','reference_sha256',4*P*P*M*N)]:
                f=root/'data'/w['id']/name
                if f.stat().st_size!=size or sha256(f)!=entry[key]: raise ValueError('input bytes differ')
    return p


def prepare_inputs(root):
    from run_selector_gpu import command
    p=verify_bundle(root,require_inputs=False)
    if (root/'data').exists() or (root/'inputs.json').exists(): raise ValueError('fresh input paths required')
    entries={}
    for w in p['workloads']:
        result=command([sys.executable,Path(__file__),'one-input','--out',root/'data'/w['id'],
                        '--shape',*w['shape']],capture_output=True,timeout=180)
        entries[w['id']]=json.loads(result.stdout)
    write(root/'inputs.json',{'package_sha256':digest(p),'workloads':entries})
    write(root/'seal.json',{'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')})
    verify_bundle(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,required=True);a.add_argument('--with-inputs',action='store_true')
    a=sub.add_parser('verify');a.add_argument('--package',type=Path,required=True)
    a=sub.add_parser('one-input');a.add_argument('--out',type=Path,required=True);a.add_argument('--shape',nargs=4,type=int,required=True)
    args=parser.parse_args()
    if args.command=='one-input': print(json.dumps(write_inputs(args.shape,args.out)))
    else:
        p=prepare(args.out) if args.command=='prepare' else verify_bundle(args.package)
        if args.command=='prepare' and args.with_inputs: prepare_inputs(args.out)
        print(json.dumps({'package_sha256':digest(p),'programs':len(p['programs']),'stages':p['stages'],'gpu_validated':False}))
