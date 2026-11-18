#!/usr/bin/env python3
"""Local append-only experiment events and bounded command recording (stdlib only)."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
SENSITIVE=re.compile(r'(?i)(password|passwd|token|secret|api[_-]?key|authorization|credential)')
ASSIGN=re.compile(r'''(?ix)((?:password|passwd|token|secret|api[_-]?key|authorization|credential)["']?\s*[:=]\s*)("[^"\n]*"|'[^'\n]*'|[^\s,;]+)''')

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def redact(value):
    if isinstance(value,dict):return {k:'[REDACTED]' if SENSITIVE.search(k) else redact(v) for k,v in value.items()}
    if isinstance(value,list):return [redact(v) for v in value]
    if not isinstance(value,str):return value
    if 'PRIVATE KEY' in value:return '[REDACTED PRIVATE KEY CONTENT]'
    value=re.sub(r'(?i)\b(Bearer|Basic)\s+[A-Za-z0-9_./+=-]+',r'\1 [REDACTED]',value)
    return ASSIGN.sub(lambda m:m[1]+'[REDACTED]',value)
def redact_argv(args):
    result=[];hide=False
    for a in args:
        if hide:result.append('[REDACTED]');hide=False;continue
        result.append(redact(a))
        if a.startswith('-') and '=' not in a and SENSITIVE.search(a):hide=True
    return shlex.join(result)

def emit(spool,run,action,message,*,outcome='unknown',ts=None,basis='recorded_at',kind='operator_note',key=None,metadata=None,stage='',shape='',candidate='',source='',source_hash='',record='',process=None):
    if outcome not in ('unknown','success','failure'):raise ValueError('invalid outcome')
    stamp=ts or now();dt=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('timestamp must include timezone')
    stamp=dt.astimezone(timezone.utc).isoformat()
    identity=hashlib.sha256(key.encode()).hexdigest() if key else uuid.uuid4().hex
    event={'@timestamp':stamp,'event':{'id':identity,'action':action,'outcome':outcome,'category':['process' if action.startswith('command_') else 'configuration'],'type':['info'],'dataset':'edmir.operations','provider':'edmir','created':now()},'message':redact(message),'experiment':{'id':run,'stage':stage,'shape':shape,'candidate':candidate,'kind':kind},'evidence':{'kind':kind,'timestamp_basis':basis,'source':source,'sha256':source_hash,'record':record},'metadata':redact(metadata or {}),'log':{'level':'error' if outcome=='failure' else 'info'}}
    if action=='command_finished' and metadata is not None:
        event['event']['duration']=round(metadata['duration_seconds']*1e9)
    if process:event['process']=redact(process)
    spool=Path(spool);spool.mkdir(parents=True,exist_ok=True)
    path=spool/(datetime.now(timezone.utc).strftime('%Y-%m-%d')+'.ndjson')
    data=(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        offset=0
        while offset<len(data):offset+=os.write(fd,data[offset:])
        os.fsync(fd)
    finally:os.close(fd)
    return identity

def run_command(spool,run,argv,timeout,stage=''):
    if not argv or timeout<=0:raise ValueError('command and positive timeout required')
    display=redact_argv(argv);operation=uuid.uuid4().hex;start=time.monotonic();proc=None
    meta={'operation_id':operation,'timeout_seconds':timeout,'cwd':os.getcwd()}
    emit(spool,run,'command_started',display,kind='live_command',stage=stage,metadata=meta,process={'command_line':display})
    result=1;reason='spawn_error'
    try:
        proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        sel=selectors.DefaultSelector();buffers={};private={}
        for stream,label in ((proc.stdout,'stdout'),(proc.stderr,'stderr')):
            os.set_blocking(stream.fileno(),False);sel.register(stream,selectors.EVENT_READ,label);buffers[label]=b'';private[label]=False
        deadline=start+timeout;kill_at=None
        while sel.get_map() or proc.poll() is None:
            current=time.monotonic()
            if current>=deadline and kill_at is None:
                reason='timeout';kill_at=current+3
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if kill_at is not None and current>=kill_at:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            for k,_ in sel.select(0.2):
                chunk=os.read(k.fileobj.fileno(),16384);label=k.data
                buffers[label]+=chunk
                while b'\n' in buffers[label] or len(buffers[label])>=16384 or (not chunk and buffers[label]):
                    if b'\n' in buffers[label]:line,buffers[label]=buffers[label].split(b'\n',1)
                    else:line,buffers[label]=buffers[label][:16384],buffers[label][16384:]
                    text=line.decode('utf-8','replace')
                    if '-----BEGIN ' in text and 'PRIVATE KEY' in text:private[label]=True
                    safe='[REDACTED PRIVATE KEY CONTENT]' if private[label] else redact(text)
                    if '-----END ' in text and 'PRIVATE KEY' in text:private[label]=False
                    emit(spool,run,'command_output',safe,kind='live_command',stage=stage,metadata={**meta,'stream':label},process={'pid':proc.pid})
                    print(safe,file=sys.stderr if label=='stderr' else sys.stdout,flush=True)
                if not chunk:sel.unregister(k.fileobj);k.fileobj.close()
            # A descendant retaining a pipe is still bounded by the process-group timeout.
            if kill_at is not None and current>kill_at+2:
                for k in list(sel.get_map().values()):sel.unregister(k.fileobj);k.fileobj.close()
                break
        result=proc.wait(timeout=5)
        if reason=='timeout':result=124
        else:reason='exit'
    except KeyboardInterrupt:
        reason='interrupted';result=130
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=3)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
    except Exception as e:
        meta['error']=str(e)
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=3)
            except (ProcessLookupError,subprocess.TimeoutExpired):pass
    finally:
        emit(spool,run,'command_finished',f'{reason}; exit={result}: {display}',outcome='success' if result==0 else 'failure',kind='live_command',stage=stage,metadata={**meta,'reason':reason,'duration_seconds':time.monotonic()-start},process={'exit_code':result,'command_line':display})
    return result

def import_w1(spool,session,run):
    session=Path(session).resolve();recovered=session/'recovered';count=0
    state_path=Path(spool).parent/'import-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    seen=set(state.get(str(session),[]))
    def item(path,record,action,message,**kwargs):
        nonlocal count
        digest=sha(path);key=str(path)+':'+digest+':'+str(record)
        if key in seen:return
        emit(spool,run,action,message,key=key,source=str(path),source_hash=digest,record=str(record),**kwargs)
        seen.add(key);count+=1
    plan_path=session/'session-plan.json';plan=json.loads(plan_path.read_text())
    active=plan.get('active_at_utc')
    item(plan_path,'active','instance_state','实例进入 Active（来自本地平台操作回执）',ts=active,kind='historical_operator_receipt',basis='platform_time_recorded_locally',metadata={'instance_id':plan.get('instance_id'),'hourly_usd':plan.get('observed_catalog_hourly_usd')})
    item(plan_path,'budget','budget_recorded','原窗口费率、预算及回收规则（回执记录；不是实时账单）',kind='historical_operator_receipt',basis='import_time_actual_time_unknown',metadata={k:plan.get(k) for k in ('budget_usd','observed_catalog_hourly_usd','platform_auto_delete_utc','platform_auto_delete_spend_usd','collection_cutoff_utc','observed_balance_decrease_usd','platform_balance_after_usd')})
    for i,op in enumerate(plan.get('operational_events',[])):
        stamp=op.get('at_utc') or op.get('initial_stop_utc')
        item(plan_path,'operation-'+str(i),'platform_operation',json.dumps(op,ensure_ascii=False),ts=stamp,kind='historical_operator_receipt',basis='recorded_operation_time' if stamp else 'import_time_actual_time_unknown',metadata=op)
    deleted=plan.get('platform_deleted_observed_utc')
    if deleted:item(plan_path,'deleted','instance_state','平台页面已确认实例删除；此时间为观察时间',ts=deleted,kind='historical_operator_receipt',basis='platform_observed_at',outcome='success',metadata={'instance_id':plan.get('instance_id'),'state':'deleted','automatic_rule_utc':plan.get('platform_auto_delete_utc')})
    for path in sorted((recovered/'results').glob('collect-[0-9][0-9][0-9].json')):
        data=json.loads(path.read_text());job=data['job'];stdout=recovered/'results'/data['stdout']
        valid=stdout.is_file() and sha(stdout)==data['stdout_sha256']
        if valid:
            valid=[json.loads(line) for line in stdout.read_text().splitlines()]==data['rows']
        match=re.search(r'-(\d{19})\.stdout$',data['stdout'])
        stamp=datetime.fromtimestamp(int(match[1])/1e9,timezone.utc).isoformat() if match else None
        rows=data['rows'];bad=[r['family'] for r in rows if not r['correctness_passed']]
        common={'ts':stamp,'basis':'process_start_from_raw_filename' if stamp else 'import_time_actual_time_unknown','kind':'historical_machine_record','stage':job['phase'],'shape':job['shape']}
        item(path,'group','collection_group',f"{job['shape']} / {job['phase']} / seed {job['seed']}: {len(rows)} 行，数值失败 {len(bad)}",outcome='success' if valid else 'failure',metadata={'seed':job['seed'],'repetition':job['repetition'],'process_seconds':data.get('process_seconds'),'rows':len(rows),'raw_stdout_hash_verified':valid,'numerical_failures':bad,'excluded':data.get('excluded_by_numerical_check',[])},**common)
        for j,row in enumerate(rows):
            if not row['correctness_passed']:item(path,'failure-'+str(j),'numerical_check_failed',row.get('failure','numerical correctness failed'),outcome='failure',candidate=row['family'],metadata=row,**common)
    completion=recovered/'results/collect-complete.json'
    if completion.exists():
        d=json.loads(completion.read_text());item(completion,0,'collection_complete','原定采集流程完成；不表示全部候选数值正确或论文实验全部完成',outcome='success',ts=d['complete_at'],basis='collector_receipt',kind='historical_machine_record',metadata=d)
    for path in sorted((recovered/'session').glob('*')):
        if path.is_file() and path.suffix in ('.log','.tsv') and path.stat().st_size<2_000_000:
            for i,line in enumerate(path.read_text(errors='replace').splitlines()):
                if not line.strip():continue
                item(path,i,'historical_output',line,kind='historical_text',basis='import_time_actual_line_time_unknown',metadata={'line':i+1})
    manifest_path=recovered/'recovery-manifest.json';manifest=json.loads(manifest_path.read_text());errors=[]
    for name,expected in manifest['files'].items():
        path=(recovered/name).resolve()
        actual=sha(path) if path.is_relative_to(recovered.resolve()) and path.is_file() else 'missing_or_unsafe'
        valid=actual==expected['sha256'] and path.stat().st_size==expected['bytes']
        if not valid:errors.append(name)
        item(manifest_path,name+':'+actual,'artifact_verified',name,outcome='success' if valid else 'failure',kind='local_verification',basis='verified_at',metadata={'relative_path':name,'bytes':expected['bytes'],'expected_sha256':expected['sha256'],'actual_sha256':actual,'match':valid})
    item(manifest_path,'summary:'+hashlib.sha256(json.dumps(errors).encode()).hexdigest(),'recovery_verified',f"回收清单 {len(manifest['files'])} 个文件，校验异常 {len(errors)}",outcome='failure' if errors else 'success',kind='local_verification',basis='verified_at',metadata={'count':len(manifest['files']),'errors':errors})
    state[str(session)]=sorted(seen);state_path.parent.mkdir(parents=True,exist_ok=True)
    tmp=state_path.with_suffix('.partial');tmp.write_text(json.dumps(state));tmp.replace(state_path)
    return count

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spool',type=Path,default=ROOT/'.local/events');sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('run');a.add_argument('--run',required=True);a.add_argument('--stage',default='');a.add_argument('--timeout',type=float,default=300);a.add_argument('argv',nargs=argparse.REMAINDER)
    a=sub.add_parser('record');a.add_argument('--run',required=True);a.add_argument('--action',required=True);a.add_argument('--message',required=True);a.add_argument('--outcome',choices=['unknown','success','failure'],default='unknown');a.add_argument('--metadata',default='{}');a.add_argument('--source',default='');a.add_argument('--ts')
    a=sub.add_parser('import-w1');a.add_argument('--session',type=Path,required=True);a.add_argument('--run',default='shadeform-w1-20261111')
    a=p.parse_args()
    if a.cmd=='run':sys.exit(run_command(a.spool,a.run,a.argv[1:] if a.argv[:1]==['--'] else a.argv,a.timeout,a.stage))
    elif a.cmd=='record':print(emit(a.spool,a.run,a.action,a.message,outcome=a.outcome,metadata=json.loads(a.metadata),source=a.source,source_hash=sha(a.source) if a.source and Path(a.source).is_file() else '',ts=a.ts))
    else:print('Imported events:',import_w1(a.spool,a.session,a.run))
if __name__=='__main__':main()
