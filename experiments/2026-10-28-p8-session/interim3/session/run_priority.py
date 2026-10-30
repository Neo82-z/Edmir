#!/usr/bin/env python3
"""External ordering adapter; frozen source, candidates and repeats unchanged."""
import argparse, json, runpy, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--deadline',type=float,required=True)
p.add_argument('--jobs',type=int,default=12)
p.add_argument('--start-at',choices=('verify','build'),default='verify')
a=p.parse_args(); root=a.root.resolve()
sys.path[:0]=[str(root/'bundle/source/EDM-IR'),str(root/'bundle/source/EDM-IR/tools')]
import prepare_p8_gpu
import run_p8_gpu as runner
order=('shape5','shape6','shape3','shape7','shape1','shape0','shape2','shape4')
assert set(order)==set(prepare_p8_gpu.STAGES) and len(order)==len(prepare_p8_gpu.STAGES)
(root/'stage-order.json').write_text(json.dumps({'order':order,'reason':'Existing four mechanism priorities first; full candidates and three process blocks per shape retained. No independent frozen shapes.'},indent=2)+'\n')
# Change only the invocation order. Mutating prepare_p8_gpu.STAGES would also
# change plan() and invalidate the original frozen measurement contract.
original=runner.base.command
def bounded(args,**kw):
    remaining=a.deadline-time.time()-5
    if remaining<=0: raise TimeoutError('rental collection deadline reached')
    kw['timeout']=min(kw.get('timeout',60),remaining)
    return original(args,**kw)
runner.base.command=bounded
work=root/'work';work.mkdir(exist_ok=True)
runner.base.write_json(work/'session.json',{'deadline_epoch':a.deadline,'started_epoch':time.time(),'state':'running','stage_order':order,'gpu_stop_managed_externally':True})
def call(action,*extra):
    sys.argv=[str(root/'server.py'),action,'--jobs',str(a.jobs),*map(str,extra)]
    print('STAGE',action,*extra,flush=True)
    runpy.run_path(str(root/'server.py'),run_name='__main__')
try:
    call('verify')
    if a.start_at=='verify': call('inventory');call('build-deps')
    call('build');call('preflight')
    for stage in order:
        remaining=int(a.deadline-time.time())
        if remaining<180: raise TimeoutError('insufficient bounded stage budget')
        call('run','--stage',stage,'--budget-seconds',remaining)
    runner.base.write_json(work/'session-final.json',{'state':'complete','finished_epoch':time.time()})
except BaseException as error:
    runner.base.write_json(work/'session-final.json',{'state':'stopped','finished_epoch':time.time(),'reason':str(error),'type':type(error).__name__})
    raise
