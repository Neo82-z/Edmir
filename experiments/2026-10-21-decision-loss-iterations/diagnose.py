"""Posthoc explanation only. Never selects or refits an outer candidate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.decision_loss_selection import DecisionLossSelector,tensors,objective
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from run_selector_gpu import write_json


def main(path,out):
    results=json.loads(path.read_text());data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    records=[]
    for v in results['variants']:
        method=v['method'];model=DecisionLossSelector.from_dict(v['full_fit'])
        x,b,y,mask,w,norm,means,scales,names=tensors(data,model.prior,model.training_audit['space'])
        t=np.asarray(model.parameters)
        anchor,_=objective(t,x,b,y,mask,w,norm,0,0,0)
        pair=objective(t,x,b,y,mask,w,norm,1,0,0)[0]-anchor
        margin=objective(t,x,b,y,mask,w,norm,0,1,0)[0]-anchor
        # Full fit is a training diagnostic, never a held-out performance claim.
        full=[]
        for i,ns in enumerate(names):
            pred=b[i,:len(ns)]+x[i,:len(ns)]@t;selected=int(np.argmin(pred))
            yy=y[i,:len(ns)];times=np.exp(yy) if model.training_audit['space']=='log' else yy
            full.append(float(times[selected]/min(times)-1))
        boundaries=[]
        for protocol in ('primary','stress'):
            for r in v[protocol]['decisions']:
                if r['shape'][1]!=512 and r['shape']!=[4,64,4096,2048]:continue
                f=next(f for f in v[protocol]['fits'] if f['heldout']==r['heldout'])['model']
                p=r['predictions']['predicted_ms'];observed=r['times_ms']
                boundaries.append({'protocol':protocol,'shape':r['shape'],'choice':r['choices'][method],
                    'observed_best':min(observed,key=observed.get),'regret':r['diagnostics'][method]['regret_all_observed'],
                    'predicted_ms':{n:p[n] for n in ('host','splitk2','splitk4')},
                    'observed_ms':{n:observed[n] for n in ('host','splitk2','splitk4')},
                    'errors_us':{n:1000*(p[n]-observed[n]) for n in ('host','splitk2','splitk4')},
                    'training_loss':f['training_audit']['loss'],
                    'basis':{k:f['prior'][k] for k in ('capacity','input_threshold','depth_threshold') if k in f['prior']}})
        ndelete=next(r for r in v['review']['deletion'] if r['shape']==[4,64,4096,2048])
        records.append({'method':method,'full_training_only':{'anchor':anchor,'pair':pair,'margin':margin,'mean_hard_regret':float(np.mean(full)),'max_hard_regret':max(full)},
                        'boundaries':boundaries,'M512_correct_out_of_6':sum(b['regret']<=1e-12 for b in boundaries if b['shape'][1]==512),
                        'M512_deletion_correct_out_of_24':sum(d['regret']<=1e-12 for r in v['review']['deletion'] if r['shape'][1]==512 for d in r['deletions']),
                        'N_deletion_correct_out_of_8':sum(d['choice']=='host' for d in ndelete['deletions']),
                        'N_stress_correct_out_of_34':sum(r['regret']<=1e-12 for r in v['review']['boundary_stress']['records'])})
    write_json(out,{'results_sha256':sha256(path),'status':'posthoc analysis, no training or hyperparameter selection','variants':records})
    for r in records:print(json.dumps({k:v for k,v in r.items() if k not in ('boundaries','full_training_only')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.results,a.out)
