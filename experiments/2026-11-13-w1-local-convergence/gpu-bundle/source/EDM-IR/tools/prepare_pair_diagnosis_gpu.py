#!/usr/bin/env python3
"""Seal minimal paired diagnostics without fitting, renting or touching holdouts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

from prepare_selector_gpu import REPO, SEED, program_artifact, source_manifest, implementation_id, write_inputs
from prepare_boundary_gpu import targets
from independent_selector_validation import WALL_TIMER, digest
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.execution_ledger import CUTLASS_REVISION

SCHEMA='edmir.paired-diagnosis-package.v1'
NCCL_REVISION='dbc86fd06e8b0c4517b95d8958a09ccacf9520c9'
PROFILES=('p4','p8-grouped')
POLICY='resource_queues_v1'
MEASUREMENT={'profiler':False,'nvtx':False,'discarded_samples':0}
GROUPED=('host','splitk4','splitk8','splitk4_groupedn2','splitk8_groupedn2')
SHORT=('host','direct','splitk2')


def plan(profile):
    if profile not in PROFILES:raise ValueError('unknown paired diagnostic profile')
    shapes=([(4,64,8192,1024),(4,64,512,256),(4,64,512,1024),
             (4,128,512,512),(4,64,512,512),(4,64,512,128)]
            if profile=='p4' else [(8,64,8192,1024)])
    works=[];programs={};jobs=[]
    for i,shape in enumerate(shapes):
        families=GROUPED if shape[2]==8192 else SHORT
        work={'id':f'w{i:03d}','shape':list(shape),'families':list(families),'blocks':len(families)}
        works.append(work)
        for family in families:
            name=f'pair_{profile.replace("-","_")}_{work["id"]}_{family}'
            programs[name]={'shape':list(shape),'compiled_family':family,'dependency_policy':POLICY,
                            'header':f'generated/{name}/edmir_program.hpp'}
        for block in range(len(families)):
            for family in families[block:]+families[:block]:
                jobs.append({'workload_id':work['id'],'shape':list(shape),'family':family,
                  'repetition':block,'trial_order':len(jobs),'ordinal':len(jobs),
                  'candidate_id':family+'@prepared','metadata_mode':'prepared','dependency_policy':POLICY,
                  'implementation_id':implementation_id(family,'prepared',dependency_policy=POLICY),
                  'executable':f'pair_{profile.replace("-","_")}_{work["id"]}_{family}'})
    return {'schema':'edmir.paired-diagnosis-plan.v1','profile':profile,'world_size':shapes[0][0],
      'workloads':works,'programs':programs,'jobs':jobs,'warmup':30,'iterations':300,
      'process_timeout_seconds':90,'collection_budget_seconds':1200 if profile=='p4' else 600,
      'cutlass_revision':CUTLASS_REVISION,'nccl_revision':NCCL_REVISION,
      'baseline_contracts':{'host':PER_SOURCE_CUTLASS.to_dict()},'timing_scope':WALL_TIMER,
      'measurement_policy':MEASUREMENT,'precision':{'input':'fp16','weight':'fp16','output':'fp16',
        'accumulator':'fp32','split_partial':'fp16 output writeback, no separate FP32 reduction'},
      'new_training':False,'independent_validation':False,'compiler_admission':False,
      'frozen_six_used':False,'full_candidate_oracle':False,
      'primary_objective':'median full-operator host wall time; retain mean and p95 as secondary',
      'blocks_are_confidence_intervals':False,'numerical_failure_labels_are_latency':False,
      'identity_limit':'new source/input/runtime identity; do not pool with historical labels or immutable-peer P8 runs',
      'prediction_claim':'historical directional hypotheses only; no cross-identity numeric latency forecast',
      'scope':'P4 existing-development diagnosis' if profile=='p4' else 'one previously measured P8 grouped shape, separate mechanism evidence',
      'repetition_budget_change':'230-process preliminary specification reduced before measurement: one balanced positional cycle per shape; not a significance claim'}


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def safe_files(base,mapping):
    for rel,h in mapping.items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('unsafe bundle path')
        if sha256(base/p)!=h:raise ValueError('bundle content changed: '+rel)


def prepare(out,profile):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);design=plan(profile)
    write(out/'plan.json',design)
    for name,c in design['programs'].items():
        program_artifact(c['shape'],c['compiled_family'],dependency_policy=POLICY).write_to(out/'generated'/name)
    (out/'generated_targets.cmake').write_text(targets(design['programs']))
    sources=source_manifest()
    for test in ('pair_diagnosis_gpu_test.py','selector_runner_safety_test.py'):
        path=REPO/'EDM-IR/tests'/test
        if path.exists():sources[str(path.relative_to(REPO))]=sha256(path)
    for rel,h in sources.items():
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
        if sha256(dest)!=h:raise ValueError('source changed during freeze')
    p={**design,'schema':SCHEMA,'status':'local_package_not_gpu_validated','sources':sources,
       'collector_sha256':digest(sources),'plan_sha256':sha256(out/'plan.json'),
       'files':{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*'))
                if f.is_file() and 'source' not in f.relative_to(out).parts}}
    write(out/'package.json',{'package_sha256':digest(p),'package':p})
    return p


def verify_bundle(root,*,require_inputs=True):
    root=root.resolve();envelope=json.loads((root/'package.json').read_text());p=envelope['package']
    if p.get('schema')!=SCHEMA or digest(p)!=envelope['package_sha256']:raise ValueError('package digest/schema mismatch')
    expected=plan(p['profile'])
    if any(p.get(k)!=v for k,v in expected.items() if k!='schema'):raise ValueError('frozen paired design changed')
    if json.loads((root/'plan.json').read_text())!=expected or sha256(root/'plan.json')!=p['plan_sha256']:
        raise ValueError('plan changed')
    if digest(p['sources'])!=p['collector_sha256']:raise ValueError('source identity mismatch')
    safe_files(root/'source',p['sources']);safe_files(root,p['files'])
    if require_inputs:
        info=json.loads((root/'inputs.json').read_text());seal=json.loads((root/'seal.json').read_text())
        if info['package_sha256']!=digest(p) or seal!={'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')}:
            raise ValueError('input seal mismatch')
        if set(info['workloads'])!={w['id'] for w in p['workloads']}:raise ValueError('input workload set differs')
        for w in p['workloads']:
            entry=info['workloads'][w['id']];P,M,K,N=w['shape']
            if entry['shape']!=w['shape'] or entry['input_seed']!=SEED:raise ValueError('input identity mismatch')
            for name,key,size in [('inputs.bin','input_sha256',2*(P*P*M*K+P*K*N)),('reference.bin','reference_sha256',4*P*P*M*N)]:
                path=root/'data'/w['id']/name
                if path.stat().st_size!=size or sha256(path)!=entry[key]:raise ValueError('input data differs')
    return p


def prepare_inputs(root):
    from run_selector_gpu import command
    p=verify_bundle(root,require_inputs=False)
    if (root/'inputs.json').exists() or (root/'data').exists():raise ValueError('fresh input paths required')
    entries={}
    for w in p['workloads']:
        result=command([sys.executable,Path(__file__),'one-input','--out',root/'data'/w['id'],'--shape',*w['shape']],
                       capture_output=True,timeout=180)
        entries[w['id']]=json.loads(result.stdout)
    write(root/'inputs.json',{'package_sha256':digest(p),'workloads':entries})
    write(root/'seal.json',{'package_sha256':digest(p),'inputs_manifest_sha256':sha256(root/'inputs.json')})
    verify_bundle(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,required=True);a.add_argument('--profile',choices=PROFILES,required=True);a.add_argument('--with-inputs',action='store_true')
    a=sub.add_parser('verify');a.add_argument('--package',type=Path,required=True)
    a=sub.add_parser('one-input');a.add_argument('--out',type=Path,required=True);a.add_argument('--shape',nargs=4,type=int,required=True)
    args=parser.parse_args()
    if args.command=='one-input':print(json.dumps(write_inputs(args.shape,args.out)))
    else:
        p=prepare(args.out,args.profile) if args.command=='prepare' else verify_bundle(args.package)
        if args.command=='prepare' and args.with_inputs:prepare_inputs(args.out)
        print(json.dumps({'profile':p['profile'],'programs':len(p['programs']),'process_runs':len(p['jobs']),'gpu_validated':False}))
