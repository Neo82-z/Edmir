"""Sequential frozen collector runs; no fitting, downloads, or asset edits."""
import argparse, datetime, json, os, signal, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['baseline','routes','trace'],required=True);p.add_argument('--attempt',default='01');a=p.parse_args()
root=a.assets.resolve();run=a.run.resolve();mode=a.mode;tag=f'{mode}-attempt{a.attempt}'
guard=json.loads((run/'guard.json').read_text());end=guard['platform_delete_epoch']-180
if end-time.time()<240:raise SystemExit('Too little time before recovery cutoff')
name='edmir-'+tag
base=['docker','run','--name',name,'--pull=never','--gpus','all','--ipc=host','--network=none','--cpus=160','--memory=850g',
      '-v',f'{root}/code:/code:ro','-v',f'{root}/model:/model:ro','-v',f'{root}/records:/cpu-records:ro',
      '-v',f'{root}/tools/nsight-systems/2025.1.3:/nsight:ro','-v',f'{run}:/work']
env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','HF_HOME':'/work/cache/hf','XDG_CACHE_HOME':'/work/cache','VLLM_CACHE_ROOT':'/work/cache/vllm','TRITON_CACHE_DIR':'/work/cache/triton','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':'/code/EDM-IR','PYTHONUNBUFFERED':'1','EDMIR_INSTANCE_ID':guard['instance_id'],'NCCL_DEBUG':'INFO','VLLM_HOST_IP':'127.0.0.1'}
for k,v in env.items():base+=['-e',f'{k}={v}']
cmd=['python3','/code/EDM-IR/tools/run_real_moe.py','--execute','--mode',mode,'--corpus','/code/experiments/2026-11-17-real-moe-local/corpus/requests-source.json','--tokens','/cpu-records/cpu-check-attempt02/requests-tokenized.json','--cpu-receipt','/cpu-records/cpu-check-attempt02/cpu-receipt.json','--manifest','/code/experiments/2026-11-16-next-window/model-manifest.json','--model-dir','/model','--source-lock','/code/experiments/2026-11-17-real-moe-local/vllm-source-lock.json','--guard','/work/guard.json','--out',f'/work/{tag}']
if mode=='trace':
    cmd=['/nsight/target-linux-x64/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-event-trace=false','--capture-range=cudaProfilerApi','--capture-range-end=repeat:4','--cuda-graph-trace=node','--output',f'/work/{tag}-timeline',*cmd]
base+=['--entrypoint',cmd[0],'edmir/vllm:0.20.1-locked',*cmd[1:]]
record={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':base,'status':'running','predictor_used':False,'timeout_epoch':end}
(run/f'{tag}-process.json').write_text(json.dumps(record,indent=2)+'\n')
with (run/f'{tag}.log').open('xb') as log:
    proc=subprocess.Popen(base,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=proc.wait(timeout=max(1,end-time.time()))
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','--time','20',name],timeout=30)
        rc=proc.wait(timeout=30);record['timed_out']=True
record.update(status='finished' if rc==0 else 'failed',exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/f'{tag}-process.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(rc)
