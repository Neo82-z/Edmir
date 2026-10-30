"""Run one bounded server phase, recording completion even if SSH disconnects."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'bundle-r01/source/EDM-IR'),str(root/'bundle-r01/source/EDM-IR/tools')]
from run_selector_gpu import command,write_json
phase=sys.argv[1];timeout=int(sys.argv[2]);argv=sys.argv[3:]
work=root/'work';work.mkdir(exist_ok=True);status=work/(phase+'.status.json')
if status.exists():raise RuntimeError('phase status already exists')
start=time.time();write_json(status,{'state':'running','pid':os.getpid(),'argv':argv,'started_unix':start,'timeout_seconds':timeout})
try:
 with (work/(phase+'.stdout.txt')).open('w') as stdout,(work/(phase+'.stderr.txt')).open('w') as stderr:
  result=command(argv,timeout=timeout,check=False,stdout=stdout,stderr=stderr)
 write_json(status,{'state':'complete' if result.returncode==0 else 'failed','returncode':result.returncode,
   'argv':argv,'started_unix':start,'elapsed_seconds':time.time()-start})
except BaseException as e:
 write_json(status,{'state':'failed','error':repr(e),'argv':argv,'started_unix':start,'elapsed_seconds':time.time()-start});raise
