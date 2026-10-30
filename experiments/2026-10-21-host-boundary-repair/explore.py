"""Bounded adaptive host-model investigation. Prior accepted model unchanged."""
from dataclasses import replace
import json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector, fused_features, _fit_surfaces
from edmir.compiler.relative_selection import ScheduleCandidate
D,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
PREVIOUS=json.loads((ROOT/'experiments/2026-10-21-additive-selector-repair/verified/results.json').read_text())

def features(r,threshold,linear):
 k,n=r.k/4096,r.n/512
 a=r.rows_per_peer*r.k/(64*4096)
 return [1,k,k*k,k*n,k*k*n,max(math.log2(a/threshold),0),*([a/16] if linear else [])]

def fit(surfaces,t,linear,ridge):
 rows=[s['host'] for s in surfaces]
 x=np.array([features(r,t,linear) for r in rows]);y=np.array([r.measured_ms for r in rows]);scale=np.maximum(np.sqrt(np.mean(x*x,0)),1e-9)
 x=x/scale;pen=np.eye(x.shape[1])*math.sqrt(ridge);pen[0,0]=0
 b=nnls(np.vstack((x,pen)),np.r_[y,np.zeros(len(scale))],maxiter=1000)[0]/scale
 return b

def nested(data,linear):
 surfaces=list(data.validate().values());groups=sorted({r.group_id for r in data.rows});trials=[]
 for t in [1,2,4,8]:
  for ridge in [.001,.01,.1]:
   errors=[]
   for g in groups:
    train=[s for s in surfaces if s['host'].group_id!=g];test=[s for s in surfaces if s['host'].group_id==g]
    b=fit(train,t,linear,ridge)
    errors.extend((np.dot(features(s['host'],t,linear),b)/s['host'].measured_ms-1)**2 for s in test)
   trials.append({'threshold':t,'ridge':ridge,'loss':float(np.mean(errors))})
 chosen=min(trials,key=lambda r:(r['loss'],r['threshold'],r['ridge']))
 return fit(surfaces,chosen['threshold'],linear,chosen['ridge']),chosen,trials

if __name__=='__main__':
 start=time.monotonic();out={'status':'adaptive development, no independent claims','variants':[]}
 for linear in [False,True]:
  result={'linear_tail':linear}
  for proto in ['primary','stress']:
   key=lambda r:r.group_id if proto=='primary' else r.k
   records=[];fits=[]
   oldrows={tuple(r['shape']):r for r in PREVIOUS[proto]['decisions']}
   for held in sorted({key(r) for r in D.rows}):
    train=replace(D,rows=tuple(r for r in D.rows if key(r)!=held));test=replace(D,rows=tuple(r for r in D.rows if key(r)==held))
    b,chosen,trials=nested(train,linear)
    fits.append({'heldout':held,'chosen':chosen,'trials':trials})
    for s in test.validate().values():
     r=s['host'];pred=dict(oldrows[r.shape]['predictions']['predicted_ms'])
     pred['host']=float(np.dot(features(r,chosen['threshold'],linear),b))
     choice=min(pred,key=lambda n:(pred[n],n!='host',n));times={n:r.measured_ms for n,r in s.items()}
     records.append({'shape':r.shape,'choice':choice,'regret':times[choice]/min(times.values())-1,'host_loss':times[choice]/times['host']-1,
                     'host_true':times['host'],'host_prediction':pred['host'],'scores':pred})
   result[proto]={'mean':float(np.mean([r['regret'] for r in records])),'max':max(r['regret'] for r in records),'host_losses':sum(r['host_loss']>1e-12 for r in records),'rows':records,'fits':fits}
  out['variants'].append(result)
  print('linear',linear,[(p,result[p]['mean'],result[p]['max'],result[p]['host_losses']) for p in ['primary','stress']],flush=True)
  for p in ['primary','stress']:
   for r in result[p]['rows']:
    if r['shape'][1]==512 or r['shape']==(4,64,4096,2048):print(p,r,flush=True)
 out['elapsed_seconds']=time.monotonic()-start
 Path(__file__).with_name('exploration.json').write_text(json.dumps(out,indent=2)+'\n')
