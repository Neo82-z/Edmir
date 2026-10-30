#!/usr/bin/env python3
"""Fixed loss/host-structure/optimizer controls; no held-out tuning loop."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import calibrated_candidate_selection as model

DATA=ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json'
PROTECTED=ROOT/'experiments/2026-10-28-host-decision-audit/protected-before.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(decisions):
    accepted=[r for r in decisions if r['chosen'] is not None]
    return {'decisions':decisions,'mean_provisional_regret':statistics.mean(r['provisional_regret'] for r in decisions),
            'max_provisional_regret':max(r['provisional_regret'] for r in decisions),
            'mean_shape_mape':statistics.mean(r['mape'] for r in decisions),
            'max_relative_error':max(r['max_relative_error'] for r in decisions),
            'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in decisions),
            'mean_kl':statistics.mean(r['kl'] for r in decisions),
            'host_errors':sum(r['host_error'] for r in decisions),'fused_errors':sum(r['fused_error'] for r in decisions),
            'selection_coverage':len(accepted)/len(decisions),
            'accepted_mean_regret':statistics.mean(r['admitted_regret'] for r in accepted) if accepted else None,
            'accepted_max_regret':max(r['admitted_regret'] for r in accepted) if accepted else None,
            'all_order_invariant':all(r['order_invariant'] for r in decisions),'compiler_admission':False}


def worker(out,ds,objective,basis,optimizer,fold):
    data=json.loads(DATA.read_text())[ds];folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
    test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
    prior=None
    if ds=='p4':prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
    fitted=model.fit(train,optimizer=optimizer,objective=objective,prior=prior,host_basis=basis)
    result={'dataset':ds,'objective':objective,'host_basis':basis,'optimizer':optimizer,'fold':fold,
            'model':fitted,'evaluation':model.evaluate(fitted,test),
            'training_diagnostic':model.evaluate(fitted,train,training_diagnostic=True)}
    path=out/f'{ds}-{objective}-{basis}-{optimizer}-fold{fold}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['evaluation'].items() if k!='decisions'}),flush=True)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    protected=json.loads(PROTECTED.read_text())
    for p,h in protected.items():assert sha(ROOT/p)==h
    jobs=[(ds,obj,basis,opt,fold) for ds in ('p4','p8')
          for obj,basis in [('legacy','original'),('calibrated','original'),('calibrated','workload')]
          if ds=='p4' or basis=='original' for opt in ('adamw','muon') for fold in range(4)]
    sources=[ROOT/'EDM-IR/edmir/compiler/calibrated_candidate_selection.py',Path(__file__),
             ROOT/'EDM-IR/tests/calibrated_candidate_selection_test.py',DATA]
    design={'settings':model.SETTINGS,'jobs':jobs,'job_count':len(jobs),'worker_timeout_seconds':60,'total_seconds_cap':600,
            'purpose':'retrospective development; one fixed pass, no held-out driven retries',
            'data_sha256':sha(DATA),'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'protected_files':protected,'frozen_six_used':False,'new_gpu_cost':0,
            'controls':'same architecture, seeds, schedule; isolate loss then host basis; AdamW/Muon each',
            'muon':'official torch.optim.Muon hidden.weight only, match_rms_adamw; scalar AdamW; actual updates logged',
            'optimizer_limit':'RMS heuristic does not guarantee equal realized effective step; no best-LR search',
            'host_workload_basis':'existing seven empirical terms plus normalized remote MK and padded MKN counts; NNLS; training-fold ridge',
            'calibrated_loss':'weighted log MSE + symmetric pair log-gap error + tau^2 KL(cost soft target || softmin predicted time)',
            'all_candidate_evaluation':'always report provisional regret on complete measured valid subset; never count abstention as zero regret'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    began=time.monotonic();all_results={};failures=[]
    for ds,obj,basis,opt,fold in jobs:
        if time.monotonic()-began>600:raise TimeoutError('fixed review deadline')
        name=f'{ds}-{obj}-{basis}-{opt}-fold{fold}'
        command=[sys.executable,__file__,'--out',str(out),'--worker','--dataset',ds,'--objective',obj,'--basis',basis,'--optimizer',opt,'--fold',str(fold)]
        with (out/(name+'.log')).open('w') as log:
            try:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
                failures.append({'job':name,'error':str(e)});print('FAILED',name,flush=True);continue
        item=json.loads((out/(name+'.json')).read_text());key=f'{ds}-{obj}-{basis}-{opt}'
        all_results.setdefault(key,[]).extend(item['evaluation']['decisions'])
        print('DONE',name,'regret',round(item['evaluation']['mean_provisional_regret'],6),'coverage',round(item['evaluation']['selection_coverage'],3),flush=True)
        (out/'results.json').write_text(json.dumps({'variants':{k:aggregate(v) for k,v in all_results.items()},
             'failures':failures,'seconds':time.monotonic()-began,'compiler_admission':False},indent=2)+'\n')
    assert all(sha(ROOT/p)==h for p,h in {**protected,**design['source_hashes']}.items())
    (out/'integrity-after.json').write_text(json.dumps({'protected_unchanged':True,'sources_unchanged':True,'job_count':len(jobs),'failed_jobs':len(failures)},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--dataset');p.add_argument('--objective');p.add_argument('--basis');p.add_argument('--optimizer');p.add_argument('--fold',type=int);a=p.parse_args()
    if a.worker:worker(a.out,a.dataset,a.objective,a.basis,a.optimizer,a.fold)
    else:run(a.out)
