#!/usr/bin/env python3
"""Independent sampled integer-dot checks of the prepared FP32 references."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'bundle/inputs.json').read_text())
records=[]
for wi,info in manifest['workloads'].items():
    p,m,k,n=info['shape'];path=ROOT/'bundle/data'/wi
    raw=np.memmap(path/'inputs.bin',dtype='<f2',mode='r')
    a=raw[:p*p*m*k].reshape(p,p,m,k);w=raw[p*p*m*k:].reshape(p,k,n)
    c=np.memmap(path/'reference.bin',dtype='<f4',mode='r').reshape(p,p,m,n)
    if not np.isfinite(c).all():raise ValueError('nonfinite reference')
    max_error=0.;checked=0
    for source in range(p):
        for dest in range(p):
            for row,col in ((0,0),(m//2,n//3),(m-1,n-1)):
                av=a[source,dest,row,:].astype(np.float64)*32
                wv=w[dest,:,col].astype(np.float64)*32
                if not (np.equal(av,np.round(av)).all() and np.equal(wv,np.round(wv)).all()):raise ValueError('non-dyadic input')
                exact=int(av.astype(np.int64)@wv.astype(np.int64))/1024
                error=abs(float(c[source,dest,row,col])-exact)
                if error!=0:raise ValueError('integer dot/reference mismatch')
                max_error=max(max_error,error);checked+=1
    records.append({'workload':wi,'shape':info['shape'],'checked_elements':checked,'max_absolute_error':max_error})
result={'scope':'Exact sampled CPU integer dots on identical inputs; not full GPU validation','records':records,'checked_elements':sum(r['checked_elements'] for r in records)}
(ROOT/'qa/input-reference-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'workloads':len(records),'checked_elements':result['checked_elements']}))
