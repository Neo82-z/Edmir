"""Mechanism-of-error audit on historical P4 data only. No fresh-shape labels."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),'/tmp/edmir-host-audit-deps']
import numpy as np
from edmir.compiler import shape_decision_model as old
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from edmir.compiler.tile_execution_model import describe_execution
from edmir.compiler.execution_ledger import source_provenance
from edmir.compiler.shared_candidate_selection import forward,canonical_choice

PREVIOUS=ROOT/'experiments/2026-10-28-shape-evaluation/historical-pilot'


def load_rows():
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    provenance=source_provenance(); rows=[]
    source=sha256(ROOT/'experiments/2026-10-20-offline-predictor/data/manifest.json')
    for key,surface in sorted(data.validate().items()):
        shape=list(key[-4:])
        for name,row in sorted(surface.items()):
            s=describe_execution(shape,name,environment_id=data.environment_id,metadata_mode='legacy',provenance=provenance)
            rows.append({'shape':shape,'execution_context':data.environment_id,'candidate_id':name,
                         'is_host':name=='host','features':old.features(s),'latency_us':row.measured_ms*1000,
                         'measurement_kind':'observed','source_id':source,'split':'train'})
    folds={tuple(r['shape']):r['fold'] for r in json.loads((PREVIOUS/'design-before-fit.json').read_text())['shape_folds']}
    return rows,folds


def tensorize(rows, model):
    surfaces=old.validate_rows(rows,split='train')
    s,j,d=len(surfaces),max(map(len,surfaces)),len(old.FEATURES)
    x,y,w=np.zeros((s,j,d)),np.zeros((s,j)),np.zeros((s,j)); mask=np.zeros((s,j),bool)
    for i,surface in enumerate(surfaces):
        nh=sum(r['is_host'] for r in surface)
        for k,r in enumerate(surface):
            x[i,k]=r['features'];y[i,k]=math.log(r['latency_us']);mask[i,k]=True
            w[i,k]=.5/(nh if r['is_host'] else len(surface)-nh)
    x=np.where(mask[...,None],(x-model['means'])/model['scales'],0.)
    return x,y,mask,w


def choices(model, rows):
    result=[]
    for surface in old.validate_rows(rows,split='train'):
        score=old.predict(model,surface);chosen=canonical_choice(score)
        actual={r['candidate_id']:r['latency_us'] for r in surface}
        oracle=canonical_choice(actual);host=actual['host']<min(v for n,v in actual.items() if n!='host')
        result.append({'shape':surface[0]['shape'],'chosen':chosen,'oracle':oracle,'host_wins':host,
                       'wrong_host_to_fused':host and chosen!='host',
                       'wrong_fused_to_host':not host and chosen=='host',
                       'regret':actual[chosen]/min(actual.values())-1,
                       'measured_us':actual,'predicted_us':{n:math.exp(v) for n,v in score.items()}})
    return {'mean_regret':float(np.mean([r['regret'] for r in result])),
            'max_regret':max(r['regret'] for r in result),
            'host_errors':sum(r['wrong_host_to_fused'] for r in result),
            'fused_errors':sum(r['wrong_fused_to_host'] for r in result),'decisions':result}


def main(out):
    from scipy.optimize import minimize
    out.mkdir(parents=True,exist_ok=False); rows,folds=load_rows();results=[]
    design={'question':'optimizer plus explicit L2 control; same features, data-loss and scaler, but this is NOT a pure optimizer ablation because AdamW decay is not equivalent to L2',
            'cases':[0,1,2,3,'full'],'iterations':1500,'maxfun':3500,'max_seconds_per_fit':15,
            'ridge':.001,'purpose':'retrospective development; no fresh labels; no convergence guarantee for nonconvex model'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    for fold in design['cases']:
        train=[r for r in rows if fold=='full' or folds[tuple(r['shape'])]!=fold]
        test=[r for r in rows if fold!='full' and folds[tuple(r['shape'])]==fold]
        model=old.fit(train) if fold=='full' else json.loads((PREVIOUS/f'mlp_decision-fold{fold}.json').read_text())['model']
        train_before=choices(model,train);x,y,mask,w=tensorize(train,model)
        begin=time.monotonic()
        def objective(t):
            if time.monotonic()-begin>15:raise TimeoutError('bounded diagnosis exceeded deadline')
            v,g=old.objective(t,x,y,mask,w,8,1.,1.)
            return v+.001*float(t@t),g+.002*t
        initial=objective(np.asarray(model['theta']))[0]
        result=minimize(objective,np.asarray(model['theta']),method='L-BFGS-B',jac=True,
                        options={'maxiter':1500,'maxfun':3500,'ftol':1e-12,'gtol':1e-6,'maxls':25})
        refined={**model,'theta':result.x.tolist()};value,grad=objective(result.x)
        entry={'fold':fold,'initial_regularized_loss':initial,'final_regularized_loss':value,
               'optimizer_success':bool(result.success),'status':int(result.status),'message':str(result.message),
               'iterations':int(result.nit),'max_gradient':float(np.max(abs(grad))),
               'seconds':time.monotonic()-begin,'before_train':train_before,'after_train':choices(refined,train),
               'after_evaluation':choices(refined,test) if test else None,'model':refined}
        results.append(entry);(out/f'fold-{fold}.json').write_text(json.dumps(entry,indent=2)+'\n')
        print(json.dumps({'fold':fold,'before_train_errors':train_before['host_errors'],
                          'after_train_errors':entry['after_train']['host_errors'],
                          'test_errors':entry['after_evaluation']['host_errors'] if test else None,
                          'test_mean':entry['after_evaluation']['mean_regret'] if test else None,
                          'status':int(result.status),'seconds':entry['seconds']}),flush=True)
    surfaces=old.validate_rows(rows,split='train');host_surfaces=[];diff=[]
    for s in surfaces:
        by={r['candidate_id']:r for r in s};h=by['host'];direct=by['direct']
        diff.append(np.array(h['features'])-direct['features'])
        if h['latency_us']<min(r['latency_us'] for r in s if not r['is_host']):
            fold=folds[tuple(h['shape'])]
            model=json.loads((PREVIOUS/f'mlp_decision-fold{fold}.json').read_text())['model']
            pred={n:math.exp(v) for n,v in old.predict(model,s).items()}
            actual={r['candidate_id']:r['latency_us'] for r in s}
            p_host=math.exp(-math.log(pred['host'])/.05-max(-math.log(t)/.05 for t in pred.values()))
            p_host/=sum(math.exp(-math.log(t)/.05-max(-math.log(v)/.05 for v in pred.values())) for t in pred.values())
            host_surfaces.append({'shape':h['shape'],'fold':fold,'predicted_us':pred,'actual_us':actual,
                'softmin_host_weight':p_host,
                'choice_if_only_host_prediction_repaired':canonical_choice({**pred,'host':actual['host']}),
                'choice_if_only_fused_predictions_repaired':canonical_choice({**actual,'host':pred['host']}),
                'train_host_winners':[r['shape'] for r in results[fold]['before_train']['decisions'] if r['host_wins']]})
    findings={'host_winning_shapes':host_surfaces,
        'host_minus_direct_feature_difference_constant':bool(np.array_equal(np.asarray(diff),np.tile(diff[0],(len(diff),1)))),
        'differing_features':[old.FEATURES[i] for i,v in enumerate(diff[0]) if v],
        'measured_host_direct_ratio_range':[
            min(next(r['latency_us'] for r in s if r['candidate_id']=='host')/next(r['latency_us'] for r in s if r['candidate_id']=='direct') for s in surfaces),
            max(next(r['latency_us'] for r in s if r['candidate_id']=='host')/next(r['latency_us'] for r in s if r['candidate_id']=='direct') for s in surfaces)],
        'host_winner_count':len(host_surfaces),'shape_count':len(surfaces),'fresh_shapes_read_for_training':False}
    (out/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    print(json.dumps(findings,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
