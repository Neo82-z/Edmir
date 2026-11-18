#!/usr/bin/env python3
"""Run detached inside the named pod to request provider Stop at an absolute deadline."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id",required=True)
    p.add_argument("--billed-from",required=True,type=float)
    p.add_argument("--hourly-total",required=True,type=float)
    p.add_argument("--budget-usd",required=True,type=float)
    p.add_argument("--log",required=True,type=Path)
    args=p.parse_args()
    if not args.pod_id or args.pod_id!=os.environ.get("RUNPOD_POD_ID"):
        p.error("current pod identity mismatch")
    if not all(math.isfinite(x) and x>0 for x in (args.hourly_total,args.budget_usd,args.billed_from)):
        p.error("invalid billing parameters")
    if args.billed_from>time.time(): p.error("billing start must not be in the future")
    stop=args.billed_from+min(3300,3600*0.9*args.budget_usd/args.hourly_total)
    cli=shutil.which("runpodctl")
    if not cli:p.error("runpodctl unavailable")
    subprocess.run([cli,"stop","pod","--help"],check=True,capture_output=True,timeout=5)
    with args.log.open("x",buffering=1) as log:
        def record(**kwargs):
            print(json.dumps(dict(epoch=time.time(),pod=args.pod_id,**kwargs)),file=log)
            log.flush();os.fsync(log.fileno())
        record(state="armed",pid=os.getpid(),stop_epoch=stop,collection_cutoff=stop-600)
        while time.time()<stop:time.sleep(max(0,min(10,stop-time.time())))
        for _ in range(5):
            try:
                r=subprocess.run([cli,"stop","pod",args.pod_id],timeout=10,capture_output=True)
                record(state="stop_requested",returncode=r.returncode)
                if r.returncode==0:return
            except subprocess.TimeoutExpired:record(state="request_timeout")
            time.sleep(2)
        raise SystemExit("provider Stop unconfirmed; verify in control plane")

if __name__=="__main__":main()
