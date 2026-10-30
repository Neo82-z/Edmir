"""Fit N response only from same-M/K host contrasts; bounded nested evaluation."""
import explore as e
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
import math,time,json

OLD=json.loads((Path(__file__).parent/'results.json').read_text())

def nf(r):
 k=r.k/4096
 return np.array([1,k,k*k,k**4])

def bf(r,t):
 k=r.k/4096;a=r.rows_per_peer*r.k/(64*4096)
 return [1,k,k*k,max(math.log2(a/t),0),a/16]

def solve(x,y,w,ridge):
 x=np.array(x);y=np.array(y);w=np.array(w);sc=np.maximum(np.sqrt(np.mean(x*x,0)),1e-9)
 p=np.eye(x.shape[1])*math.sqrt(ridge);p[0,0]=0
 return nnls(np.vstack((x/sc*w[:,None],p)),np.r_[y*w,np.zeros(len(sc))],maxiter=1000)[0]/sc

def nfit(ss,ridge):
 groups={}
 for s in ss:
  r=s['host'];groups.setdefault((r.rows_per_peer,r.k),[]).append(r)
 x=[];y=[];w=[]
 for rs in groups.values():
  pairs=[(a,b) for a,b in combinations(sorted(rs,key=lambda r:r.n),2) if a.n!=b.n]
  for a,b in pairs:
   x.append(nf(a)*math.log2(b.n/a.n));y.append(b.measured_ms-a.measured_ms);w.append(1/math.sqrt(len(pairs)))
 return solve(x,y,w,ridge)

def fit(ss,t,ridge,nridge):
 nc=nfit(ss,nridge);rows=[s['host'] for s in ss]
 ys=[r.measured_ms-float(nf(r)@nc)*math.log2(r.n/512) for r in rows]
 return solve([bf(r,t) for r in rows],ys,[1]*len(rows),ridge),nc

def pred(r,t,co,nc):return float(np.dot(bf(r,t),co)+(nf(r)@nc)*math.log2(r.n/512))

def tune(data):
 ss=list(data.validate().values());groups=sorted({r.group_id for r in data.rows});trials=[]
 for t in [1,2,4,8]:
  for ridge in [.001,.01,.1]:
   for nridge in [.001,.01,.1]:
    errors=[]
    for g in groups:
     train=[s for s in ss if s['host'].group_id!=g];test=[s for s in ss if s['host'].group_id==g]
     co,nc=fit(train,t,ridge,nridge)
     errors.extend((pred(s['host'],t,co,nc)/s['host'].measured_ms-1)**2 for s in test)
    trials.append({'t':t,'ridge':ridge,'nridge':nridge,'loss':float(np.mean(errors))})
 best=min(trials,key=lambda a:(a['loss'],a['t'],a['ridge'],a['nridge']))
 co,nc=fit(ss,best['t'],best['ridge'],best['nridge'])
 return co,nc,best,trials

if __name__=='__main__':
 start=time.monotonic();out={}
 for proto in ['primary','stress']:
  key=lambda r:r.group_id if proto=='primary' else r.k
  rows=[];fits=[];old={tuple(r['shape']):r for r in OLD[proto]['decisions']}
  for held in sorted({key(r) for r in e.D.rows}):
   train=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)!=held));test=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)==held))
   co,nc,best,trials=tune(train);fits.append({'heldout':held,'best':best,'trials':trials})
   for s in test.validate().values():
    r=s['host'];ps=dict(old[r.shape]['predictions']['predicted_ms']);ps['host']=pred(r,best['t'],co,nc)
    choice=min(ps,key=lambda n:(ps[n],n!='host',n));times={n:r.measured_ms for n,r in s.items()}
    rows.append({'shape':r.shape,'choice':choice,'regret':times[choice]/min(times.values())-1,'host_loss':times[choice]/times['host']-1,'predictions':ps})
  out[proto]={'mean':float(np.mean([r['regret'] for r in rows])),'max':max(r['regret'] for r in rows),'host_losses':sum(r['host_loss']>1e-12 for r in rows),'rows':rows,'fits':fits}
  print(proto,out[proto]['mean'],out[proto]['max'],out[proto]['host_losses'],flush=True)
  for r in rows:
   if r['shape'][1]==512 or r['shape']==(4,64,4096,2048):print(r,flush=True)
 out['elapsed_seconds']=time.monotonic()-start
 Path(__file__).with_name('paired_exploration.json').write_text(json.dumps(out,indent=2)+'\n')
