#!/usr/bin/env python3
"""Rebuild original inputs; accept only exact matches to the pre-existing seal."""
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();package=root/'bundle'
sys.path[:0]=[str(package/'source/EDM-IR'),str(package/'source/EDM-IR/tools')]
from prepare_p8_gpu import verify_bundle
from prepare_selector_gpu import write_inputs
frozen=verify_bundle(package,require_inputs=False)
expected=json.loads((package/'inputs.json').read_text())['workloads']
records=[]
for w in frozen['workloads']:
 start=time.time();actual=write_inputs(w['shape'],package/'data'/w['id'])
 for key in ('shape','input_seed','input_sha256','reference_sha256','input_bytes','reference_bytes'):
  if actual[key]!=expected[w['id']][key]:raise ValueError('Original input identity differs: '+w['id']+' '+key)
 records.append(dict(workload_id=w['id'],elapsed_seconds=time.time()-start,actual=actual))
 print('EXACT_INPUT_MATCH',w['id'],flush=True)
(root/'input-reconstruction.json').write_text(json.dumps(records,indent=2)+'\n')
verify_bundle(package)
print('Full sealed bundle verified',flush=True)
