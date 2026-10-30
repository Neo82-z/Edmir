#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,time
root=Path('/root/2026-10-25-p8-gpu-collection');out=root/'work/correctness-diagnostic';exe=out/'diagnostic';receipt={'purpose':'Repeat same diagnostic executable with required minimum warmup/iterations. Earlier 0/1 invocations were CLI rejections and are not GPU numerical failures.','tests':[]}
for i in range(3):
 if time.time()>1793165400-70:break
 args=[str(exe),'splitk16',str(root/'bundle/data/w005/inputs.bin'),str(root/'bundle/data/w005/reference.bin'),'20','100','prepared']
 start=time.time()
 with (out/f'stdout_replay_{i}.txt').open('w') as so,(out/f'stderr_replay_{i}.txt').open('w') as se:
  try:r=subprocess.run(args,stdout=so,stderr=se,timeout=60);result={'returncode':r.returncode}
  except subprocess.TimeoutExpired:result={'timeout':True}
 receipt['tests'].append({'repetition':i,'args':args,'elapsed_seconds':time.time()-start,**result})
 (out/'replay-valid-args.json').write_text(json.dumps(receipt,indent=2)+'\n')
receipt['complete_epoch']=time.time();(out/'replay-valid-args.json').write_text(json.dumps(receipt,indent=2)+'\n')
