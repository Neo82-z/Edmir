#!/usr/bin/env python3
"""Make a movable, bounded next-GPU package from the measured execution source.

Never contacts or rents a server. Old CUDA/runtime sources are hash-checked and
copied; only plan/collector orchestration changes. Predictions remain conditional
on the old measured environment until fresh anchors establish comparability.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clean_mechanism_data import sha
from independent_selector_validation import digest
from edmir.compiler.regime_prediction import RegimeModel
from edmir.compiler.residual_regime_prediction import ResidualRegimeModel

REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'experiments/2026-10-21-runpod-mechanism-collection'
REVISION=REPO/'experiments/2026-10-25-final-local-revision/results'
FROZEN='54aff5ff71ff1dbab33a90348e84ec9d3bc1ea8b755e11d30c6ebd777bd6edc5'
STAGES=('anchors','crossings','library','warmup')
HOSTS=('host','host_compiler_cublas','host_merged_cutlass','host_merged_cublaslt')


def specifications():
    result=[]
    def add(stage,m,k,n,families):
        for f in families:result.append((stage,(4,m,k,n),f))
    add('anchors',256,8192,512,('splitk2','splitk4','splitk8','splitk2_groupedn2','splitk4_groupedn2','splitk8_groupedn2','host','host_merged_cublaslt'))
    add('anchors',64,1024,512,HOSTS)
    add('anchors',64,1024,512,('splitk2_groupedn2','splitk4_groupedn2'))
    add('anchors',64,4096,2048,('host','splitk2','splitk4','splitk4_groupedn2'))
    add('anchors',64,3072,1536,('host','splitk4','splitk4_groupedn2'))
    add('anchors',64,8192,512,('host','splitk4','splitk4_groupedn2','splitk8_groupedn2'))
    add('anchors',512,4096,512,('splitk2_groupedn2','host_merged_cutlass','host_merged_cublaslt'))
    add('anchors',192,3072,512,('host','splitk2','splitk2_groupedn2'))
    for m in (192,320):
        add('crossings',m,8192,512,('splitk2','splitk4','splitk8','splitk4_groupedn2','splitk8_groupedn2'))
    for m in (256,320,576,640):add('crossings',m,3072,512,('splitk2','splitk2_groupedn2'))
    for k in (768,1536):add('library',64,k,512,HOSTS)
    add('warmup',64,1024,512,HOSTS)
    return result


def make_design():
    specs=specifications(); shapes=list(dict.fromkeys(s for _,s,_ in specs));configs=[]
    for stage,s,f in specs:
        wi=shapes.index(s);mode='merged' if f.startswith('host_merged') else 'prepared'
        # Identity strings are populated by the same frozen implementation_id.
        configs.append({'stage':stage,'workload_id':f'w{wi:03d}','shape':list(s),'family':f,
            'candidate_id':f+'@'+mode,'metadata_mode':mode,
            'executable':f'mechanism_w{wi:03d}_'+('host' if mode=='merged' else f),
            **({'diagnostic_warmup':200} if stage=='warmup' else {})})
    return shapes,configs


def run(args,**kw):
    return subprocess.run([str(x) for x in args],check=True,timeout=kw.pop('timeout',180),**kw)


def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def copy_frozen_source(dest):
    envelope=json.loads((OLD/'remote-run-01/revision-01/package.json').read_text());p=envelope['package']
    if digest(p)!=FROZEN or envelope['package_sha256']!=FROZEN:raise ValueError('wrong original frozen source')
    for rel,expected in p['sources'].items():
        candidate=OLD/'remote-run-01/revision-01/source'/rel
        if not candidate.exists():candidate=OLD/'bundle/source'/rel
        if sha(candidate)!=expected:raise ValueError('changed historical source: '+rel)
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,target)
    return p


def adapt_plan(source):
    shapes,configs=make_design()
    path=source/'EDM-IR/tools/prepare_mechanism_gpu.py';text=path.read_text()
    start=text.index('SHAPES =');end=text.index('\ndef mappings():')
    replacement=f'''SHAPES = {repr(tuple(shapes))}
STAGES = {repr(STAGES)}
POLICY = {{'profiler':False,'nvtx':False,'discarded_samples':0}}
FROZEN_CONFIGURATIONS = {repr(configs)}

def configurations():
    return [{{**c,'implementation_id':implementation_id(c['family'],c['metadata_mode'])}} for c in FROZEN_CONFIGURATIONS]

def plan():
    configs=configurations(); jobs=[]; rng=random.Random(20261025)
    for stage in STAGES:
        for block in range(3):
            group=[c for c in configs if c['stage']==stage];rng.shuffle(group)
            for c in group:jobs.append({{**c,'repetition':block,'trial_order':len(jobs),'ordinal':len(jobs)}})
    return {{'schema':'edmir.regime-validation-plan.v1','configurations':configs,'jobs':jobs,
        'stages':{{s:{{'configurations':sum(c['stage']==s for c in configs),'process_runs':sum(j['stage']==s for j in jobs),'optional':s!='anchors'}} for s in STAGES}},
        'hardware':'4x A100-SXM4-80GB; SM80/108 SM; full pairwise NV12/P2P',
        'cutlass_revision':CUTLASS_REVISION,'nccl_revision':NCCL_REVISION,'timing_scope':WALL_TIMER,
        'warmup':20,'iterations':100,'repetitions':3,'process_timeout_seconds':120,
        'measurement_policy':POLICY,'new_training':False,'independent_validation':False,'compiler_admission':False}}

'''
    text=text[:start]+replacement+text[end:]
    text=text.replace('edmir.mechanism-gpu-package.v1','edmir.regime-validation-gpu-package.v1')
    text=text.replace('mechanism_diagnosis_development','regime_validation_development')
    path.write_text(text)
    path=source/'EDM-IR/tools/run_mechanism_gpu.py';text=path.read_text()
    text=text.replace("p['warmup'],p['iterations'],job['metadata_mode']", "job.get('diagnostic_warmup',p['warmup']),p['iterations'],job['metadata_mode']")
    text=text.replace("result=base.validate_measurement(row,job,info,p)",
        "trial={**p,'warmup':job['diagnostic_warmup']} if job.get('diagnostic_warmup') else p\n    result=base.validate_measurement(row,job,info,trial)")
    begin=text.index("        contrasts=[]\n");end=text.index('        for a,b in pairs:',begin)
    text=text[:begin]+"        contrasts=[]\n        from itertools import combinations\n        pairs=list(combinations(candidates,2))\n"+text[end:]
    path.write_text(text)


def freeze_predictions(root,configs):
    original=json.loads((REVISION/'configurations.json').read_text())
    stable=[r for r in original if r['eligible_for_calibration']]
    models={'regime_max':RegimeModel.fit(stable),'residual_depth':ResidualRegimeModel.fit(stable)}
    expected=json.loads((REVISION/'models.json').read_text())
    if any(json.loads(json.dumps(m.to_dict()))!=expected[v] for v,m in models.items()):raise ValueError('model freeze differs from reviewed result')
    rows=[]
    for c in configs:
        rows.append({**c,'observed_us':None,'primary_timing_label':c['stage']!='warmup',
            'conditional_predictions':{v:m.predict(c['shape'],c['family'],context_id=m.context_id) for v,m in models.items()}
                if c['stage']!='warmup' else None,
            'new_measurement_context_qualified':False})
    write(root/'frozen-predictions.json',{'primary':'regime_max','challenger':'residual_depth',
        'source_context_id':models['regime_max'].context_id,'source_analysis_sha256':sha(REVISION/'analysis-seal.json'),
        'note':'Forecasts conditional on old source and measured environment. New build/device environment requires anchor comparison; never relabel new data with the old context.',
        'rows':rows})
    shutil.copyfile(REVISION/'models.json',root/'frozen-models.json')
    shutil.copyfile(REVISION/'decision.json',root/'model-decision.json')


def inputs(root,python):
    bundle=root/'bundle';env=json.loads((bundle/'package.json').read_text());p=env['package']
    old=json.loads((OLD/'remote-run-01/revision-01/inputs.json').read_text())['workloads']
    entries={}
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
    for w in p['workloads']:
        match=next(((k,v) for k,v in old.items() if v['shape']==w['shape']),None)
        dest=bundle/'data'/w['id']
        if match:
            k,v=match;shutil.copytree(OLD/'bundle/data'/k,dest);entries[w['id']]=v
        else:
            r=run([python,bundle/'source/EDM-IR/tools/prepare_mechanism_gpu.py','one-input','--out',dest,'--shape',*w['shape']],capture_output=True,text=True,env=environment)
            entries[w['id']]=json.loads(r.stdout)
    write(bundle/'inputs.json',{'package_sha256':digest(p),'workloads':entries})
    write(bundle/'seal.json',{'package_sha256':digest(p),'inputs_manifest_sha256':sha(bundle/'inputs.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--input-python',type=Path,required=True);args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    staging=root/'preparation-source';old=copy_frozen_source(staging);adapt_plan(staging)
    bundle=root/'bundle'
    run([sys.executable,staging/'EDM-IR/tools/prepare_mechanism_gpu.py','prepare','--out',bundle])
    new=json.loads((bundle/'package.json').read_text())['package']
    protected={r:h for r,h in old['sources'].items() if r.startswith('EDM-IR/runtime/') or r.startswith('EDM-IR/edmir/')}
    for rel,value in protected.items():
        if sha(bundle/'source'/rel)!=value:raise ValueError('execution or lowering source changed: '+rel)
    write(root/'execution-source-identity.json',{'old_package_sha256':FROZEN,'new_package_sha256':digest(new),
        'unchanged_execution_sources':protected,'changed_orchestration_sources':[r for r,h in new['sources'].items() if old['sources'].get(r)!=h],
        'old_policy':'resource_queues_v1','new_current_runtime_included':False,'new_GPU_environment_qualified':False})
    inputs(root,args.input_python)
    shutil.copytree(OLD/'dependencies',root/'dependencies',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    server=(OLD/'server.py').read_text().replace("verify_bundle,CUTLASS_REVISION,NCCL_REVISION","verify_bundle,CUTLASS_REVISION,NCCL_REVISION,STAGES")
    server=server.replace("choices=('anchors','mechanisms','geometry')","choices=STAGES")
    (root/'server.py').write_text(server)
    shutil.copyfile(REPO/'EDM-IR/tools/review_regime_gpu.py',root/'review.py')
    configs=json.loads((bundle/'plan.json').read_text())['configurations']
    freeze_predictions(root,configs)
    run([sys.executable,'-S',root/'server.py','verify'])
    write(root/'local-status.json',{'state':'prepared_not_GPU_validated','stages':new['stages'],
        'compiled_programs_required':len(new['programs']),'gpu_runs':0,'cuda_compilation_verified':False,
        'primary_configs':sum(c['stage']!='warmup' for c in configs),
        'diagnostic_configs':sum(c['stage']=='warmup' for c in configs),'fresh_processes':len(new['jobs'])})
    print(json.dumps({'delivery':str(root),'stages':new['stages'],'programs':len(new['programs'])},indent=2))


if __name__=='__main__':main()
