#!/usr/bin/env python3
"""Pod-scoped fixed-deadline provider Stop; survives SSH disconnect."""
import datetime,json,os,pathlib,shutil,subprocess,time
pod='ktgaqsajla51i0'
assert os.environ.get('RUNPOD_POD_ID')==pod,'Wrong pod identity'
cli=shutil.which('runpodctl');assert cli,'runpodctl missing'
deadline=datetime.datetime.fromisoformat('2026-10-28T04:12:00+00:00').timestamp()
assert time.time()<deadline,'Deadline already passed'
log=pathlib.Path('/root/edmir-stop-guard.jsonl')
def emit(**record):
 with log.open('a') as f:
  f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**record))+'\n');f.flush();os.fsync(f.fileno())
emit(state='armed',pid=os.getpid(),pod_id=pod,deadline_utc='2026-10-28T04:12:00Z',command=[cli,'stop','pod',pod])
while time.time()<deadline: time.sleep(min(20,max(0,deadline-time.time())))
for attempt in range(8):
 emit(state='requesting_provider_stop',attempt=attempt)
 try:
  result=subprocess.run([cli,'stop','pod',pod],capture_output=True,text=True,timeout=30)
  emit(state='stop_command_result',attempt=attempt,returncode=result.returncode,stdout=result.stdout[-2000:],stderr=result.stderr[-2000:])
  if result.returncode==0: break
 except subprocess.TimeoutExpired: emit(state='stop_request_timeout',attempt=attempt)
 time.sleep(10)
