#!/usr/bin/env python3
"""Snapshot only experiment evidence; never package the environment or credentials."""
from pathlib import Path
import argparse,hashlib,json,tarfile,time
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
assert a.label.isalnum()
root=Path('/root/2026-10-25-p8-gpu-collection'); files=[]
work=root/'work'
if work.exists():
 for f in work.rglob('*'):
  if not f.is_file():continue
  rel=f.relative_to(work)
  if rel.parts[0]!='build' or str(rel) in ('build/selector_build.json','build/build.stdout.txt','build/build.stderr.txt','build/CMakeCache.txt','build/Testing/Temporary/LastTest.log'):
   files.append((f,Path('work')/rel))
for n in ('input-reconstruction.json','stage-order.json'):
 f=root/n
 if f.exists():files.append((f,Path(n)))
for n in ('collection.log','collection-resume.log','collection-build2.log','cmake-install.log','numpy-install.log','edmir-stop-guard.jsonl','stop_guard.stdout','bootstrap.py','hydrate_inputs.py','run_priority.py','stop_guard_v2.py','run_subset.py','subset.log','diagnose_correctness.py','correctness-diagnostic.log','run_diagnostic_replay.py','replay-correctness.log'):
 f=Path('/root')/n
 if f.exists():files.append((f,Path('session')/n))
out=Path('/root')/('results-'+a.label+'.tar.gz')
with tarfile.open(out,'w:gz') as t:
 for f,rel in files:t.add(f,arcname=str(rel),recursive=False)
sha=hashlib.sha256(out.read_bytes()).hexdigest()
print(json.dumps({'path':str(out),'sha256':sha,'bytes':out.stat().st_size,'files':len(files),'snapshot_epoch':time.time()}),flush=True)
