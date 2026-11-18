#!/usr/bin/env python3
"""Pull existing text logs over SSH; never launch/stop a remote experiment.

The worker runs on the Mac. It reads only configured paths remotely, emits
redacted local journal events, and persists byte offsets for resume.
"""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from ops import ROOT, emit, now, redact

REMOTE_READER = r'''
import base64,glob,json,os,sys
request=json.load(sys.stdin)
result=[]
paths=sorted(glob.glob(request['glob']))
if len(paths)>100:raise ValueError('more than 100 logs: narrow the configured glob')
for path in paths:
    if not os.path.isfile(path):continue
    previous=request['files'].get(path,{})
    with open(path,'rb') as f:
        info=os.fstat(f.fileno());identity=str(info.st_dev)+':'+str(info.st_ino)
        reset=bool(previous) and (previous['identity']!=identity or info.st_size<previous['offset'])
        start=0 if reset else previous.get('offset',0)
        f.seek(start);chunk=f.read(262144)
        end=chunk.rfind(b'\n')+1
        if not end and len(chunk)==262144:raise ValueError('log line exceeds 256 KiB: '+path)
        chunk=chunk[:end]
        result.append(dict(path=path,identity=identity,start=start,end=start+len(chunk),size=info.st_size,
            generation=previous.get('generation',0)+int(reset),reset=reset,
            content=base64.b64encode(chunk).decode()))
print(json.dumps(result))
'''


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def load(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def paths(name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}',name):
        raise ValueError('invalid source name')
    root=ROOT/'.local/remotes'/name
    root.mkdir(parents=True,exist_ok=True)
    return root,ROOT/'.local/remotes'/(name+'.json')


@contextmanager
def lock(root):
    with (root/'worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def running(root):
    try:
        with lock(root):return False
    except BlockingIOError:return True


def ingest(name,config,state,batch,spool):
    """Consume complete lines only. Never advance beyond successfully journaled data."""
    count=0
    for item in batch:
        source=item['path'];previous=state['files'].get(source,{})
        content=base64.b64decode(item['content'],validate=True)
        if item['end']-item['start']!=len(content) or (content and not content.endswith(b'\n')):
            raise ValueError('remote reader returned incomplete data')
        expected=0 if item['reset'] else previous.get('offset',0)
        if item['start']!=expected:raise ValueError('unexpected log offset')
        private=False if item['reset'] else previous.get('private_key_block',False)
        offset=item['start']
        run=config['run_prefix']+'-'+Path(source).parent.name
        if item['reset']:
            emit(spool,run,'remote_log_rotated','远端日志身份改变或文件缩短，从当前文件起点继续。',
                 kind='remote_log_observation',source=source,metadata={'remote':name,'generation':item['generation']})
        for line in content.splitlines(keepends=True):
            # splitlines also splits CR progress updates; all bytes are accounted for.
            text=line.decode('utf-8','replace').rstrip('\r\n')
            if '-----BEGIN ' in text and 'PRIVATE KEY' in text:private=True
            safe='[REDACTED PRIVATE KEY CONTENT]' if private else redact(text)
            if '-----END ' in text and 'PRIVATE KEY' in text:private=False
            if text.strip():
                key=json.dumps([name,source,item['identity'],item['generation'],offset,
                                hashlib.sha256(line).hexdigest()],ensure_ascii=False)
                emit(spool,run,'remote_log',safe,kind='remote_log_observation',
                     basis='received_at_original_line_time_unparsed',stage='remote-log',source=source,key=key,
                     metadata={'remote':name,'byte_offset':offset,'generation':item['generation'],
                               'remote_file_size':item['size']})
                count+=1
            offset+=len(line)
        state['files'][source]={'identity':item['identity'],'offset':item['end'],
                               'generation':item['generation'],'private_key_block':private,'size':item['size']}
    return count


def poll(name,root,config):
    state=load(root/'state.json',{'files':{},'total_forwarded_lines':0})
    command=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
             '-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',
             '-p',str(config['port']),'-i',str(Path(config['identity_file']).expanduser()),
             config['user']+'@'+config['host'],'python3 -c '+shlex.quote(REMOTE_READER)]
    result=subprocess.run(command,input=json.dumps({'glob':config['remote_glob'],'files':state['files']}),
                          capture_output=True,text=True,timeout=22)
    if result.returncode:raise RuntimeError(redact(result.stderr[-1500:]))
    batch=json.loads(result.stdout)
    count=ingest(name,config,state,batch,ROOT/'.local/events')
    state.update(last_success_at=now(),total_forwarded_lines=state['total_forwarded_lines']+count,
                 last_forwarded_lines=count,matched_files=len(batch))
    save(root/'state.json',state)
    return count


def watch(name,root,config):
    with lock(root):
        save(root/'worker.json',{'pid':os.getpid(),'started_at':now(),'mode':'read_only_remote_log_pull'})
        previous_error=None
        try:
            while not (root/'stop').exists():
                try:
                    count=poll(name,root,config)
                    if previous_error is not None:
                        emit(ROOT/'.local/events',config['run_prefix'],'remote_log_connection_restored',
                             'SSH 日志连接恢复；从本地检查点补入。',outcome='success',kind='log_transport',metadata={'remote':name})
                    previous_error=None
                    save(root/'health.json',{'checked_at':now(),'connected':True,'forwarded_lines':count})
                except Exception as e:
                    error=redact(str(e))
                    if error!=previous_error:
                        emit(ROOT/'.local/events',config['run_prefix'],'remote_log_connection_failed',error,
                             outcome='failure',kind='log_transport',stage='remote-log',metadata={'remote':name})
                    previous_error=error
                    save(root/'health.json',{'checked_at':now(),'connected':False,'error':error})
                until=time.monotonic()+max(5,float(config.get('interval_seconds',15)))
                while time.monotonic()<until and not (root/'stop').exists():time.sleep(.5)
        finally:
            save(root/'worker.json',{'pid':os.getpid(),'stopped_at':now(),'mode':'stopped'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop','status','once','watch'])
    parser.add_argument('name')
    args=parser.parse_args();root,config_path=paths(args.name)
    config=load(config_path,None)
    if config is None:raise SystemExit('Missing local source config: '+str(config_path))
    if args.action=='start':
        if running(root):print('Remote log collector already running');return
        (root/'stop').unlink(missing_ok=True)
        with (root/'worker.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch',args.name],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        for _ in range(40):
            if running(root):print('Started read-only log collector: '+args.name);return
            if proc.poll() is not None:raise SystemExit('Collector failed; inspect '+str(root/'worker.log'))
            time.sleep(.1)
        raise SystemExit('Collector startup not confirmed; inspect status before retrying')
    elif args.action=='stop':
        (root/'stop').touch()
        for _ in range(60):
            if not running(root):print('Log collector stopped; remote experiment untouched');return
            time.sleep(.5)
        raise SystemExit('Stop requested; collector has not exited yet')
    elif args.action=='watch':watch(args.name,root,config)
    elif args.action=='once':
        with lock(root):print('Forwarded lines:',poll(args.name,root,config))
    else:
        print(json.dumps({'running':running(root),'config':{k:v for k,v in config.items() if k!='identity_file'},
                          'worker':load(root/'worker.json',{}),'health':load(root/'health.json',{}),
                          'progress':load(root/'state.json',{})},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
