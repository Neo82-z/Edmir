#!/usr/bin/env python3
"""Replay a GPU error witness with exact partition sums and FP16 epilogues."""
import argparse,struct,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent.parent/'2026-10-25-p8-gpu-collection/bundle';data=root/'data/w005';P,M,K,N=8,64,4096,2048
h=lambda x:struct.unpack('<e',struct.pack('<e',x))[0]
results=[]
for f in sorted(a.diagnostic.glob('stderr_*.txt')):
 lines=f.read_text().splitlines();w=[x for x in lines if x.startswith('WITNESS ')]
 if not w:continue
 _,slot,index,expected,actual=w[0].split();slot,index=int(slot),int(index);expected,actual=float(expected),float(actual)
 row,col=divmod(index,N);destination=slot%P
 with (data/'reference.bin').open('rb') as ref:
  ref.seek(4*(slot*M*N+index));reference=struct.unpack('<f',ref.read(4))[0]
 assert reference==expected
 with (data/'inputs.bin').open('rb') as inp:
  inp.seek(2*(slot*M*K+row*K));aa=struct.unpack('<'+str(K)+'e',inp.read(2*K));bb=[]
  for k in range(K):
   inp.seek(2*(P*P*M*K+destination*K*N+k*N+col));bb.append(struct.unpack('<e',inp.read(2))[0])
 parts=[sum(int(aa[k]*32)*int(bb[k]*32) for k in range(i,i+256))/1024 for i in range(0,K,256)]
 assert sum(parts)==reference
 values={}
 for q in (1,2,4,8,16):
  value=0.0;step=16//q
  for i in range(0,16,step):value=h(value+sum(parts[i:i+step]))
  values[q]=value
 results.append({'stderr_file':f.name,'stderr_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'slot':slot,'row':row,'column':col,'reference':reference,'gpu_actual':actual,'simulated_by_q':values,'q16_exactly_matches_gpu_witness':values[16]==actual,'absolute_error':abs(actual-reference),'normalized_error':abs(actual-reference)/max(abs(reference),1.0),'exact_partition_sums':parts})
result={'shape':[P,M,K,N],'candidate':'splitk16@prepared_immutable','input_sha256':json.loads((root/'inputs.json').read_text())['workloads']['w005']['input_sha256'],'witnesses':results,'limit':'Only observed error witnesses are reproduced. This can identify a sufficient FP16 serial-epilogue rounding mechanism; it does not certify all outputs or rule out additional synchronization errors. No tolerance changed.'}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
