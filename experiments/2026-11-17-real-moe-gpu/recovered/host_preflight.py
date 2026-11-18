"""Bounded inventory and offline image load on the authorized GPU VM."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/vdb/edmir-assets');out=root/'runs/20261117';out.mkdir(parents=True,exist_ok=True);pre=out/'preflight';pre.mkdir(exist_ok=True)
report={'status':'failed','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_files_modified':False}
def run(label,args,timeout=30):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    (pre/f'{label}.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f'{label}: exit {result.returncode}')
    return result.stdout
try:
    uuid=run('mount-uuid',['findmnt','-n','-o','UUID','--target',str(root)]).strip()
    assert uuid=='34464844-e985-4623-a0e2-3e7cdf43624c',uuid
    run('mount',['findmnt','--target',str(root)])
    run('disk',['df','-h',str(root)])
    gpu=run('gpus',['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,mig.mode.current','--format=csv,noheader'])
    rows=gpu.strip().splitlines();assert len(rows)==8 and all('A100' in s and 'SXM' in s and 'Disabled' in s for s in rows),gpu
    topo=run('topology',['nvidia-smi','topo','-m'])
    sys.path.insert(0,str(root/'code/EDM-IR'))
    from edmir.benchmark.real_moe_config import validate_topology
    validate_topology(topo)
    run('nvidia-details',['nvidia-smi','-q']);run('docker-version',['docker','version'])
    hashes=json.loads((root/'code-files-final-sha256.json').read_text())
    assert all(hashlib.sha256((root/'code'/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    archive=root/'vllm-0.20.1-amd64.docker.tar'
    with archive.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha=='3e52284234d31e1e94c3a8f42ccf913b13f3beb1287d0e1be782fda9688f0725'
    run('image-load',['docker','load','--input',str(archive)],timeout=600)
    image=json.loads(run('image-inspect',['docker','image','inspect','edmir/vllm:0.20.1-locked']))[0]
    assert image['Architecture']=='amd64' and image['Os']=='linux'
    assert image['Id'] in ('sha256:bf0486e51765766a069f6eb26a63a06748859a568b24200acd3aa4edb58f3d35','sha256:aef4ebc906574fa2e6e52079937b7d3a8735e218f6353b58667b215e4257a34b')
    report.update(status='passed',volume_uuid=uuid,gpu_count=8,all_pair_nvlink=True,archive_sha256=sha,source_hashes_verified=len(hashes),image_id=image['Id'])
except BaseException as exc:
    report['error']=repr(exc);raise
finally:
    report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (pre/'host-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
