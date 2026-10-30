"""Traffic-coupled host and separated fused fixed/partition work, nested fitting."""
import explore as e
import numpy as np
from scipy.optimize import nnls
from dataclasses import replace
import math,json,time
from pathlib import Path

T=(1,2,4,8);B=(.25,.5,1,2);R=(.001,.01,.1);C=(16,32,64,108)

def hf(r,t,b):
 k,n=r.k/4096,r.n/512;a=r.rows_per_peer*r.k/(64*4096)
 return [1,k,k*k,max(k-b,0)*n,max(math.log2(a/t),0),a/16]

def ff(r,c,mode):
 x=list(e.fused_features(r,c))
 if mode==0:return x
 k,q,g=r.k/4096,r.split_k_slices,r.n_tiles_per_cta
 G=math.ceil(r.rows_per_peer/64)*math.ceil(r.n/(128*g))
 x[3]=G*(q-1)/108
 x[6]=max(k-1,0)*G/108
 if mode==2:x.append(G/108)
 return x

def solve(surfs,host,p,ridge,mode):
 rows=[r for s in surfs for r in s.values() if r.is_host==host]
 counts={s['host'].shape:len(s)-1 for s in surfs}
 x=np.array([hf(r,*p) if host else ff(r,p,mode) for r in rows]);y=np.array([r.measured_ms for r in rows]);scale=np.maximum(np.sqrt(np.mean(x*x,0)),1e-9)
 w=np.array([1 if host else 1/math.sqrt(counts[r.shape]) for r in rows]);x=x/scale*w[:,None];y=y*w
 pen=np.eye(len(scale))*math.sqrt(ridge);pen[0,0]=0
 coef=nnls(np.vstack((x,pen)),np.r_[y,np.zeros(len(scale))],maxiter=1000)[0]/scale
 return coef

def tune(data,host,mode):
 ss=list(data.validate().values());groups=sorted({r.group_id for r in data.rows});scores=[]
 params=[(t,b) for t in T for b in B] if host else C
 for p in params:
  for ridge in R:
   errors=[]
   for g in groups:
    train=[s for s in ss if s['host'].group_id!=g];test=[s for s in ss if s['host'].group_id==g]
    co=solve(train,host,p,ridge,mode)
    for s in test:
     ys=[(float(np.dot(hf(r,*p) if host else ff(r,p,mode),co))-r.measured_ms)**2/s['host'].measured_ms**2 for r in s.values() if r.is_host==host]
     errors.append(np.mean(ys))
   scores.append({'p':p,'ridge':ridge,'loss':float(np.mean(errors))})
 best=min(scores,key=lambda a:(a['loss'],a['p'],a['ridge']))
 return solve(ss,host,best['p'],best['ridge'],mode),best,scores

start=time.monotonic();out={'status':'adaptive engineering; external folds not independent','variants':[]}
for mode in [0,1,2]:
 result={'fused_mode':mode}
 for proto in ['primary','stress']:
  key=lambda r:r.group_id if proto=='primary' else r.k
  rows=[];fits=[];old={tuple(r['shape']):r for r in e.PREVIOUS[proto]['decisions']}
  for held in sorted({key(r) for r in e.D.rows}):
   train=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)!=held));test=replace(e.D,rows=tuple(r for r in e.D.rows if key(r)==held))
   h,hp,hs=tune(train,True,mode);f,fp,fs=tune(train,False,mode)
   fits.append({'heldout':held,'host':hp,'fused':fp,'host_trials':hs,'fused_trials':fs})
   for s in test.validate().values():
    r=s['host'];pred={n:float(np.dot(hf(r,*hp['p']),h) if n=='host' else np.dot(ff(s[n],fp['p'],mode),f)) for n in old[r.shape]['predictions']['predicted_ms']}
    choice=min(pred,key=lambda n:(pred[n],n!='host',n));times={n:r.measured_ms for n,r in s.items()}
    rows.append({'shape':r.shape,'choice':choice,'regret':times[choice]/min(times.values())-1,'host_loss':times[choice]/times['host']-1,'scores':pred})
  result[proto]={'mean':float(np.mean([r['regret'] for r in rows])),'max':max(r['regret'] for r in rows),'host_losses':sum(r['host_loss']>1e-12 for r in rows),'rows':rows,'fits':fits}
 out['variants'].append(result)
 print('mode',mode,[(p,result[p]['mean'],result[p]['max'],result[p]['host_losses']) for p in ['primary','stress']],flush=True)
 for p in ['primary','stress']:
  for r in result[p]['rows']:
   if r['shape'][1]==512 or r['shape']==(4,64,4096,2048):print(p,r,flush=True)
out['elapsed_seconds']=time.monotonic()-start
Path(__file__).with_name('exploration_v2.json').write_text(json.dumps(out,indent=2)+'\n')
