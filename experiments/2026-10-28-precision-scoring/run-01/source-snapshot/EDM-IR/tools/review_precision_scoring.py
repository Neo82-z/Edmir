#!/usr/bin/env python3
"""Bounded 2x2 optimizer/CE review, with separate historical P4 and current P8 identities."""
from pathlib import Path
import argparse,hashlib,json,math,os,statistics,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import precision_candidate_selection as model
from edmir.compiler import shape_decision_model as base
from edmir.compiler import path_aware_selection as previous
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from edmir.compiler.tile_execution_model import describe_execution,structure_from_ledger
from edmir.compiler.execution_ledger import source_provenance


def precision(family):
    q=1 if family.startswith('host') else model.parse_schedule_family(family)[0]
    return {'input':'fp16','weight':'fp16','output':'fp16','accumulator':'fp32',
            'partial':'unknown' if family=='host_merged_cublaslt' else 'fp16' if q>1 else 'none',
            'accumulator_evidence':'api_requested' if family=='host_merged_cublaslt' else 'source'}


def datasets():
    rows=[];folder=ROOT/'experiments/2026-10-20-offline-predictor/data'
    data,*_=load_identified_selector_dataset(folder);provenance=source_provenance();source=sha256(folder/'manifest.json')
    for key,surface in sorted(data.validate().items()):
        shape=list(key[-4:])
        for name,r in sorted(surface.items()):
            structure=describe_execution(shape,name,environment_id=data.environment_id,metadata_mode='legacy',provenance=provenance)
            rows.append({'shape':shape,'execution_context':data.environment_id,'candidate_id':name,'is_host':name=='host',
                         'features':base.features(structure),'precision':precision(name),'latency_us':r.measured_ms*1000,
                         'measurement_kind':'observed','source_id':source,'split':'train',
                         'validity':{'status':'passed','source_id':source,'scope':'historical accepted dataset, not a new correctness experiment'}})
    fold_json=json.loads((ROOT/'experiments/2026-10-28-shape-evaluation/historical-pilot/design-before-fit.json').read_text())['shape_folds']
    p4={'rows':rows,'folds':fold_json,'prior_kind':'same train-fold archival cost prior; residual only inside training hull'}
    folder=ROOT/'experiments/2026-10-28-p8-session';stats=json.loads((folder/'results.json').read_text())
    ledger=json.loads((ROOT/'experiments/2026-10-25-p8-gpu-collection/bundle/execution_accounts.json').read_text())
    recovered=folder/'recovered/work/subset';ctx=json.loads((recovered/'preflight.json').read_text())['context']
    rows=[];excluded=[]
    for s in stats['shapes']:
        raw=recovered/s['stage']/'measurements.jsonl';source=sha256(raw);observed=[json.loads(x) for x in raw.read_text().splitlines()]
        for name,info in sorted(s['candidates'].items()):
            if not info['complete_and_correct']:
                excluded.append({'shape':s['shape'],'candidate_id':name,'reason':'failed_numerical_tolerance','failed_processes':info['failed_processes'],'latency_us':None});continue
            records=[r for r in observed if r['candidate_id']==name]
            assert len(records)==3 and all(r['correctness_passed'] for r in records)
            wid=f'w{int(s["stage"].replace("shape","")):03d}';account=ledger[wid+'_'+name];ident=account['identity']
            assert all(r['implementation_id']==ident['implementation_id'] and r['dependency_policy']==ident['dependency_policy'] for r in records)
            rows.append({'shape':s['shape'],'execution_context':ctx['environment_id'],'candidate_id':name,'is_host':name.startswith('host'),
                         'features':base.features(structure_from_ledger(account)),'precision':precision(records[0]['family']),
                         'latency_us':info['median_us'],'measurement_kind':'observed','source_id':source,'split':'train',
                         'process_medians_us':info['process_medians_us'],
                         'validity':{'status':'passed','source_id':source,'scope':'three processes on recorded input SHA/runtime only'},
                         'input_sha256':records[0]['input_sha256'],'implementation_id':ident['implementation_id'],
                         'geometry_source':'frozen package ledger, matching measured implementation; planned context is not a measurement'})
    shapes=sorted({tuple(r['shape']) for r in rows});p8={'rows':rows,'folds':[{'shape':list(s),'fold':i} for i,s in enumerate(shapes)],
        'excluded':excluded,'prior_kind':'none; four-shape development only, no P4 coefficient transfer',
        'limitations':'Unequal measured candidate subsets; all four best valid candidates are fused; no bidirectional selection or independent generalization claim.'}
    return {'p4':p4,'p8':p8}


def aggregate(decisions):
    return {'mean_regret':statistics.mean(r['regret'] for r in decisions),'max_regret':max(r['regret'] for r in decisions),
            'host_errors':sum(r['host_error'] for r in decisions),'fused_errors':sum(r['fused_error'] for r in decisions),
            'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in decisions),
            'mean_kl_to_cost_target':statistics.mean(r['kl_to_cost_target'] for r in decisions),
            'top2_retention':statistics.mean(r['top2_contains_measured_best'] for r in decisions),
            'all_order_invariant':all(r['order_invariant'] for r in decisions),'decisions':decisions}


def worker(out,dataset,optimizer,ce,fold):
    data=json.loads((out/'datasets.json').read_text())[dataset];folds={tuple(x['shape']):x['fold'] for x in data['folds']}
    train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=fold]
    test=[{**r,'split':'evaluation'} for r in data['rows'] if folds[tuple(r['shape'])]==fold]
    prior=None
    if dataset=='p4':
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
    fitted=model.fit(train,optimizer=optimizer,ce_weight=ce,prior=prior)
    result={'model':fitted,'evaluation':model.evaluate(fitted,test),'dataset':dataset,'fold':fold}
    (out/f'{dataset}-{optimizer}-ce{int(ce)}-fold{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['evaluation'].items() if k!='decisions'}),flush=True)


def run(out):
    out.mkdir(parents=True,exist_ok=False);data=datasets()
    (out/'datasets.json').write_text(json.dumps(data,indent=2)+'\n')
    protected=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/protected-before.json').read_text())
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    prior_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'experiments/2026-10-28-host-decision-audit/support-and-prior').glob('prior-fold-*.json')}
    design={'purpose':'retrospective development only','settings':model.SETTINGS,'variants':[[o,c] for o in ['adamw','dion3'] for c in [0.,1.]],
            'datasets_sha256':sha256(out/'datasets.json'),'prior_hashes':prior_hashes,'torch_device':'cpu','network_precision':'FP32',
            'optimizer_source':'https://github.com/microsoft/dion/tree/'+model.DION_REVISION,'official_dion_unchanged':True,
            'optimizer_mode':'official CPU eager arithmetic via torch.compiler.set_stance(force_eager), no Triton or Gram kernels',
            'optimizer_comparison_limit':'Fixed shared base learning rate, equal steps/seeds and common scalar AdamW settings; not separate best-hyperparameter tuning. Dion matrix normalization and momentum follow official defaults.',
            'loss':'host/fused-balanced log-time MSE + expected measured regret + previous best-gap; soft cost-target CE weight 0 or 1',
            'soft_target_temperature':.05,'selection_weights_are_best_probabilities':False,
            'whole_shape_splits':True,'frozen_six_used':False,'no_new_gpu':True,'no_bf16_latency_labels':True,
            'compiler_admission':False,'total_seconds_cap':600,'worker_timeout_seconds':50,
            'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'EDM-IR/edmir/compiler/precision_candidate_selection.py']},
            'protected':protected}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    began=time.monotonic();results={};failures=[]
    for ds in ('p4','p8'):
        results[ds]={}
        for optimizer,ce in design['variants']:
            decisions=[];audits=[]
            for fold in range(4):
                if time.monotonic()-began>600:raise TimeoutError('fixed total local review budget')
                name=f'{ds}-{optimizer}-ce{int(ce)}-fold{fold}'
                args=[sys.executable,__file__,'--out',str(out),'--worker','--dataset',ds,'--optimizer',optimizer,'--ce',str(ce),'--fold',str(fold)]
                with (out/(name+'.log')).open('w') as log:
                    try:
                        proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=50,check=True)
                    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
                        failures.append({'case':name,'error':str(e)});print('FAILED',name,flush=True);continue
                item=json.loads((out/(name+'.json')).read_text());decisions+=item['evaluation']['decisions']
                audits.append({'fold':fold,'fit_seconds':item['model']['seconds'],'runs':item['model']['runs'],'selected_seed':item['model']['selected_seed']})
                print('DONE',name,'regret',item['evaluation']['mean_regret'],'CE',item['evaluation']['mean_cross_entropy'],flush=True)
            if len(audits)==4:results[ds][f'{optimizer}-ce{int(ce)}']={**aggregate(decisions),'fits':audits}
            (out/'results.json').write_text(json.dumps({'datasets':results,'failures':failures,'seconds':time.monotonic()-began,'compiler_admission':False},indent=2)+'\n')
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    (out/'protected-after.json').write_text(json.dumps({'all_unchanged':True,'files':protected},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--dataset');p.add_argument('--optimizer');p.add_argument('--ce',type=float);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.worker:worker(a.out,a.dataset,a.optimizer,a.ce,a.fold)
    else:run(a.out)
