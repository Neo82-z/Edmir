#!/usr/bin/env python3
import concurrent.futures,json,os,pathlib,subprocess,time
root=pathlib.Path('/root/2026-10-25-p8-gpu-collection')
subprocess.run(['tar','-xzf','/root/code-only.tar.gz','-C','/root'],check=True,timeout=60)
plan=json.loads((root/'bundle/plan.json').read_text())
(root/'dependencies').mkdir(exist_ok=True)
def fetch(name,url,revision):
 path=root/'dependencies'/name;path.mkdir()
 for cmd in (['git','init',str(path)],['git','-C',str(path),'fetch','--depth','1',url,revision],['git','-C',str(path),'checkout','--detach','FETCH_HEAD']):
  subprocess.run(cmd,check=True,timeout=180)
 print('DEPENDENCY_PINNED',name,revision,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 tasks=[pool.submit(fetch,'cutlass','https://github.com/NVIDIA/cutlass.git',plan['cutlass_revision']),pool.submit(fetch,'nccl','https://github.com/NVIDIA/nccl.git',plan['nccl_revision'])]
 for task in tasks:task.result()
subprocess.run(['python3','/root/hydrate_inputs.py','--root',str(root)],check=True,timeout=300)
subprocess.run(['python3','/root/run_priority.py','--root',str(root),'--deadline','1793165400','--jobs','16'],check=True,timeout=max(1,1793165400-time.time()+10))
