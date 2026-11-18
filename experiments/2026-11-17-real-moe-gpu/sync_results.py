"""Recover this experiment's outputs only; never copy model/image/cache assets."""
import argparse,datetime,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--until',type=float,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
out=Path(__file__).resolve().parent/'recovered';out.mkdir(exist_ok=True)
key='/Users/neo/Downloads/private_key.pem'
cmd=['rsync','-rt','--partial','--timeout=30','--exclude=cache/','--exclude=__pycache__/','--exclude=*.jsonl.gz','--exclude=*.nsys-rep','--exclude=*.sqlite*','-e',f'ssh -i {key} -o BatchMode=yes -o ConnectTimeout=10 -o HostKeyAlias=shadeform-6711de91-f052-42c3-857e-222cb3b49bec',f'shadeform@{a.host}:/mnt/vdb/edmir-assets/runs/20261117/',str(out)+'/']
count=0
while time.time()<a.until:
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr[-2000:]}
    except Exception as exc:result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(exc)}
    (out.parent/'sync-last.json').write_text(json.dumps(result,indent=2)+'\n');count+=1
    if a.once:break
    time.sleep(30)
print('Sync iterations:',count)
