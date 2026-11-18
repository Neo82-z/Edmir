"""Copy different rank journals on independent SSH streams; preserve partial files."""
import concurrent.futures, datetime, json, subprocess, time
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'recovered/routes-attempt01/workers'
out.mkdir(parents=True,exist_ok=True)
ssh='ssh -i /Users/neo/Downloads/private_key.pem -o BatchMode=yes -o ConnectTimeout=10 -o HostKeyAlias=shadeform-6711de91-f052-42c3-857e-222cb3b49bec'
def copy(rank):
    while time.time()<1794903900:
        source=f'shadeform@86.38.238.92:/mnt/vdb/edmir-assets/runs/20261117/routes-attempt01/workers/*-rank{rank}.jsonl.gz'
        try:
            p=subprocess.run(['rsync','-rt','--partial','--timeout=45','-e',ssh,source,str(out)+'/'],capture_output=True,text=True,timeout=600)
            r=dict(exit_code=p.returncode,stderr=p.stderr[-1500:])
        except Exception as exc:r=dict(error=repr(exc))
        r['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (root/f'journal-sync-rank{rank}.json').write_text(json.dumps(r,indent=2)+'\n')
        time.sleep(30)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(copy,range(8)))
