#!/usr/bin/env python3
"""Freeze bounded P8 runtime/strong-baseline collection; no device access."""
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
from edmir.compiler.execution_ledger import execution_ledger, ALL_FAMILIES, CUTLASS_REVISION
from edmir.compiler.measurement_dataset import sha256

SCHEMA = 'edmir.p8-gpu-package.v1'
NCCL_REVISION = 'dbc86fd06e8b0c4517b95d8958a09ccacf9520c9'
SHAPES = ((8,32,1024,512),(8,64,2048,1024),(8,32,4096,2048),(8,256,8192,512),
          (8,32,8192,2048),(8,64,4096,2048),(8,64,8192,1024),(8,512,8192,512))
STAGES = tuple(f'shape{i}' for i in range(len(SHAPES)))
POLICY = {'profiler':False, 'nvtx':False, 'discarded_samples':0}
FUSED = tuple(f+suffix for suffix in ('','_groupedn2') for f in ('direct','splitk2','splitk4','splitk8','splitk16'))
PRIMARY = ('host','host_merged_cutlass','host_merged_cublaslt') + FUSED
LEGACY = 'resource_queues_v1'
IMMUTABLE = 'uniform_immutable_peer_v1'


def configurations():
    result=[]
    for wi,shape in enumerate(SHAPES):
        variants=[(f, LEGACY if f.startswith('host') else IMMUTABLE, True) for f in PRIMARY]
        if wi in (4,5,6):
            variants += [(f,LEGACY,False) for f in ('splitk4','splitk4_groupedn2','splitk8','splitk8_groupedn2')]
        if wi==7: variants += [('splitk16_groupedn2',LEGACY,False)]
        for family,policy,primary in variants:
            mode='merged' if family in MERGED_FAMILIES else 'prepared'
            compiled='host' if family in MERGED_FAMILIES else family
            suffix='_immutable' if policy==IMMUTABLE else ''
            result.append({'stage':STAGES[wi],'workload_id':f'w{wi:03d}','shape':list(shape),
                'family':family,'candidate_id':family+'@'+mode+suffix,
                'metadata_mode':mode,'dependency_policy':policy,'primary':primary,
                'implementation_id':implementation_id(family,mode,dependency_policy=policy),
                'executable':f'p8_w{wi:03d}_{compiled}'+suffix})
    return result


def plan():
    configs=configurations();jobs=[];rng=random.Random(20261025)
    for stage in STAGES:
        for block in range(3):
            group=[c for c in configs if c['stage']==stage];rng.shuffle(group)
            for c in group:jobs.append({**c,'repetition':block,'trial_order':len(jobs),'ordinal':len(jobs)})
    return {'schema':'edmir.p8-plan.v1','configurations':configs,'jobs':jobs,
            'stages':{s:{'configurations':sum(c['stage']==s for c in configs),
                        'process_runs':sum(j['stage']==s for j in jobs),'optional':False} for s in STAGES},
            'hardware':'8x A100-SXM4-80GB; SM80/108 SM; full pairwise NV12/P2P',
            'cutlass_revision':CUTLASS_REVISION,'nccl_revision':NCCL_REVISION,
            'timing_scope':WALL_TIMER,'warmup':20,'iterations':100,'repetitions':3,
            'process_timeout_seconds':90,'measurement_policy':POLICY,
            'new_training':False,'independent_validation':False,'compiler_admission':False,
            'purpose':'new runtime validation and strong host comparison; historical shapes are development data',
            'primary_candidates':list(PRIMARY),'legacy_controls_are_not_primary':True,
            'rental_wall_limit_seconds':3600,'collection_stop_before_rental_second':2700,
            'stop_target_before_rental_second':3300,'rental_cost_limit_usd':15,
            'old_latency_parameters_applicable_to_new_policy':False}


def mappings():
    works = [{'id':f'w{i:03d}', 'shape':list(s)} for i,s in enumerate(SHAPES)]
    programs = {}
    for c in configurations():
        name = c['executable']
        programs[name] = {'shape':c['shape'], 'compiled_family':(
            'host' if c['family'] in MERGED_FAMILIES else c['family']),
            'dependency_policy':c['dependency_policy'], 'header':f'generated/{name}/edmir_program.hpp'}
    return works, programs


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare(out):
    out = out.resolve(); out.mkdir(parents=True,exist_ok=False)
    design = plan(); works, programs = mappings()
    write(out/'plan.json',design)
    for name,c in programs.items():
        program_artifact(c['shape'],c['compiled_family'],dependency_policy=c['dependency_policy']).write_to(out/'generated'/name)
    (out/'generated_targets.cmake').write_text(targets(programs))
    ledgers = {}
    for c in configurations():
        key = c['workload_id']+'_'+c['candidate_id']
        if c['family'] in ALL_FAMILIES:
            ledgers[key] = execution_ledger(c['shape'],c['family'],
                environment_id='planned-a100-p8-current-runtime-not-measured', metadata_mode=c['metadata_mode'], dependency_policy=c['dependency_policy'])
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
    p = {'schema':SCHEMA, 'mode':'p8_runtime_development',
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
    expected={'mode':'p8_runtime_development','workloads':works,'programs':programs,
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
