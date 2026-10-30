"""Training-only nested hyperparameter selection after disclosed development."""
import explore as e
from dataclasses import replace
import json,time
from pathlib import Path
import numpy as np

def tune(data):
 scores=[]
 for c in [16,32,64,108]:
  for ridge in [.001,.01,.1]:
   errors=[]
   for group in sorted({r.group_id for r in data.rows}):
    train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=group));test=replace(data,rows=tuple(r for r in data.rows if r.group_id==group))
    b,f=e.fit(train,c,0,ridge)
    for s in test.validate().values():
     norm=s['host'].measured_ms
     host=(np.dot(e.hf(s['host']),b)/norm-1)**2
     fused=np.mean([((np.dot(e.ff(r,c,0),f)-r.measured_ms)/norm)**2 for r in s.values() if not r.is_host])
     errors.append(.5*(host+fused))
   scores.append({'capacity':c,'ridge':ridge,'loss':float(np.mean(errors))})
 best=min(scores,key=lambda r:(r['loss'],r['capacity'],r['ridge']))
 return best,scores

start=time.monotonic();out={}
for proto in ['group','k']:
 key=lambda r:r.group_id if proto=='group' else r.k
 rows=[];fits=[]
 for held in sorted({key(r) for r in e.D.rows}):
  train=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)!=held));test=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)==held))
  best,scores=tune(train);c,ridge=best['capacity'],best['ridge'];b,f=e.fit(train,c,0,ridge);guard=e.ExecutionCostSelector.fit(train)
  fits.append({'heldout':held,'best':best,'cv':scores})
  for s in test.validate().values():
   first=next(iter(s.values()));pred={}
   for n,r in s.items():
    cand=e.ScheduleCandidate.from_measurement(r,e.D.environment_id)
    if r.n_tiles_per_cta==2 and not guard.explain(cand,baseline=e.D.baseline)['eligible']:continue
    pred[n]=float(np.dot(e.hf(r),b) if r.is_host else np.dot(e.ff(r,c,0),f))
   choice=min(pred,key=lambda n:(pred[n],n));times={n:r.measured_ms for n,r in s.items()}
   rows.append({'shape':first.shape,'choice':choice,'regret':times[choice]/min(times.values())-1,'host_loss':times[choice]/times['host']-1,'scores':pred})
  print(proto,held,best,flush=True)
 out[proto]={'mean':float(np.mean([r['regret'] for r in rows])),'max':max(r['regret'] for r in rows),'host_losses':sum(r['host_loss']>1e-12 for r in rows),'rows':rows,'fits':fits}
out['elapsed_seconds']=time.monotonic()-start
Path(__file__).with_name('nested_01.json').write_text(json.dumps(out,indent=2)+'\n')
print({p:{k:v for k,v in d.items() if k not in ['rows','fits']} for p,d in out.items() if isinstance(d,dict)},flush=True)
