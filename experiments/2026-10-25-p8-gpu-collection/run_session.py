#!/usr/bin/env python3
"""Bound all build/measurement subprocesses by an externally supplied rental deadline."""
import argparse,json,os,runpy,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);p.add_argument('--jobs',type=int,default=12);a=p.parse_args()
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'bundle/source/EDM-IR'),str(root/'bundle/source/EDM-IR/tools')]
import run_p8_gpu as runner
from prepare_p8_gpu import STAGES
original=runner.base.command
def bounded(args,**kw):
    remaining=a.deadline-time.time()-5
    if remaining<=0: raise TimeoutError('rental collection deadline reached')
    kw['timeout']=min(kw.get('timeout',60),remaining)
    return original(args,**kw)
runner.base.command=bounded
work=root/'work';work.mkdir(exist_ok=True)
runner.base.write_json(work/'session.json',{'deadline_epoch':a.deadline,'started_epoch':time.time(),'state':'running','gpu_stop_managed_externally':True})
def call(action,*extra):
    sys.argv=[str(root/'server.py'),action,'--jobs',str(a.jobs),*map(str,extra)]
    print('STAGE',action,*extra,flush=True)
    runpy.run_path(str(root/'server.py'),run_name='__main__')
try:
    call('verify');call('inventory');call('build-deps');call('build');call('preflight')
    for stage in STAGES:
        remaining=int(a.deadline-time.time())
        if remaining<180: raise TimeoutError('insufficient bounded stage budget')
        call('run','--stage',stage,'--budget-seconds',remaining)
    runner.base.write_json(work/'session-final.json',{'state':'complete','finished_epoch':time.time()})
except BaseException as error:
    runner.base.write_json(work/'session-final.json',{'state':'stopped','finished_epoch':time.time(),'reason':str(error),'type':type(error).__name__})
    raise
