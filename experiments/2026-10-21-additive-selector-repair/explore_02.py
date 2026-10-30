"""Bounded retrospective architecture development; every trial is saved."""
from pathlib import Path
import sys, json, math, time
from dataclasses import replace
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.execution_cost_selection import ExecutionCostSelector
from evaluate_execution_cost_selector import PRIORITY
D,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')

def hf(r):
 m,k,n=math.ceil(r.rows_per_peer/64),r.k/4096,r.n/512
 # Separate shape axes. No workload identifier or label-dependent feature.
 return [1,k,k*k,k*math.log2(m),k*k*math.log2(m),k*n,k*k*n,k*n*n]

def ff(r,c,variant=0):
 m,k,n,q,g=r.rows_per_peer,r.k/4096,r.n,r.split_k_slices,r.n_tiles_per_cta
 G=math.ceil(m/64)*math.ceil(n/(128*g)); h=max(G*q/c-1,0)
 base=[1,k/q,q-1,G*q/108,k/q*h, (q-1)*h/16, k*G/108]
 if variant>=1:base += [k*k/q*h]
 if variant>=2:base += [k*k/q]
 if variant>=3:base += [max(0,math.log2(m/64))*k]
 # Wider tile has different MMA, epilogue and grid costs; retain K, q.
 return base+[(g-1)*v for v in [1,k/q,q-1,k,G*q/108]]

def fit(d,c,v,ridge):
 surf=d.validate(); hosts=[s['host'] for s in surf.values()]
 X=np.array([hf(r) for r in hosts]); y=np.array([r.measured_ms for r in hosts]); scale=np.maximum(np.sqrt(np.mean(X*X,0)),1e-9)
 X=X/scale
 pen=np.eye(X.shape[1])*math.sqrt(ridge);pen[0,0]=0
 b=np.linalg.lstsq(np.vstack((X,pen)),np.r_[y,np.zeros(len(scale))],rcond=None)[0]/scale
 fused=[r for r in d.rows if not r.is_host]; weights=np.array([1/math.sqrt(sum(not a.is_host for a in surf[r.grouped_shape].values())) for r in fused])
 X=np.array([ff(r,c,v) for r in fused]);y=np.array([r.measured_ms for r in fused]);scale=np.maximum(np.sqrt(np.mean(X*X,0)),1e-9)
 X=X/scale*weights[:,None];y=y*weights
 pen=np.eye(X.shape[1])*math.sqrt(ridge);pen[0,0]=0
 f=nnls(np.vstack((X,pen)),np.r_[y,np.zeros(len(scale))],maxiter=1000)[0]/scale
 return b,f

def eval(c,v,ridge):
 out={}
 for protocol in ['group','k']:
  key=lambda r:r.group_id if protocol=='group' else r.k
  rows=[]
  for held in sorted({key(r) for r in D.rows}):
   train=replace(D,rows=tuple(r for r in D.rows if key(r)!=held));test=replace(D,rows=tuple(r for r in D.rows if key(r)==held))
   b,f=fit(train,c,v,ridge); guard=ExecutionCostSelector.fit(train)
   for s in test.validate().values():
    first=next(iter(s.values()));scores={}
    for n,r in s.items():
     cand=ScheduleCandidate.from_measurement(r,D.environment_id)
     if r.n_tiles_per_cta==2 and not guard.explain(cand,baseline=D.baseline)['eligible']:continue
     scores[n]=float(np.dot(hf(r),b) if r.is_host else np.dot(ff(r,c,v),f))
    choice=min(scores,key=lambda n:(scores[n],n));times={n:r.measured_ms for n,r in s.items()}
    rows.append({'shape':first.shape,'choice':choice,'regret':times[choice]/min(times.values())-1,'host_loss':times[choice]/times['host']-1,'scores':scores})
  out[protocol]={'mean':float(np.mean([r['regret'] for r in rows])),'max':max(r['regret'] for r in rows),'host_losses':sum(r['host_loss']>1e-12 for r in rows),'rows':rows}
 out.update(capacity=c,variant=v,ridge=ridge)
 return out
if __name__=='__main__':
 start=time.monotonic(); results=[]
 for v in [0]:
  for c in [16,32,64,108]:
   for ridge in [.001,.01,.1]:
    r=eval(c,v,ridge);results.append(r)
    print(v,c,ridge,'mean',round(r['group']['mean']*100,3),'max',round(r['group']['max']*100,3),'Kmax',round(r['k']['max']*100,3),'priority',[(x['shape'][1:],x['choice']) for x in r['group']['rows'] if tuple(x['shape']) in PRIORITY],flush=True)
 Path(__file__).with_name('exploration_02.json').write_text(json.dumps({'status':'adaptive retrospective development; not independent validation','trials':results,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
