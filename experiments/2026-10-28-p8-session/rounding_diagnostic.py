#!/usr/bin/env python3
"""Bounded CPU sample of ideal exact partition sums with FP16 serial epilogues.
This is a numerical mechanism check, not a replay of device synchronization.
"""
import mmap,struct,time,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;data=root.parent/'2026-10-25-p8-gpu-collection/bundle/data/w005/inputs.bin';P,M,K,N=8,64,4096,2048
half=lambda x:struct.unpack('<e',struct.pack('<e',x))[0]
lookup={struct.unpack('<H',struct.pack('<e',v/32))[0]:v for v in range(-16,16)}
start=time.monotonic();records={q:{'max_abs':0.0,'max_rel':0.0,'witness':None} for q in (1,2,4,8,16)};count=0
with data.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
 values=memoryview(mm).cast('H')
 for slot in range(P*P):
  if time.monotonic()-start>20:break
  dest=slot%P;weight=P*P*M*K+dest*K*N
  for row in (0,31,63):
   aa=[lookup[values[slot*M*K+row*K+k]] for k in range(K)]
   for col in range(0,N,37):
    parts=[sum(aa[k]*lookup[values[weight+k*N+col]] for k in range(i,i+256))/1024 for i in range(0,K,256)]
    exact=sum(parts);count+=1
    for q in records:
     out=0.0;step=16//q
     for i in range(0,16,step):out=half(out+sum(parts[i:i+step]))
     error=abs(out-exact);rel=error/max(abs(exact),1.0);r=records[q]
     r['max_abs']=max(r['max_abs'],error)
     if rel>r['max_rel']:r.update(max_rel=rel,witness={'slot':slot,'row':row,'column':col,'expected':exact,'simulated':out,'absolute_error':error})
 del values
result={'shape':[P,M,K,N],'sampled_outputs':count,'elapsed_seconds':time.monotonic()-start,'model':'Exact integer products/sums on original FP16 inputs; each serial split epilogue rounded to FP16. q1 rounds once. Matches source arithmetic structure, does not test device ordering or semaphore correctness. Sample only, no timing or general correctness claim.','input_path':str(data),'records':records}
(root/'rounding-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
