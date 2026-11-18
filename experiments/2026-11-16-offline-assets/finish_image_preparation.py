"""Finish the current local preparation job; never contacts a cloud provider."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

REPO=Path('/Users/neo/code/llm-serving')
RECORDS=REPO/'experiments/2026-11-16-offline-assets'
ASSETS=Path('/Users/neo/edmir-assets/2026-11-16')
DOCKER=['docker','-H','unix:///Users/neo/.colima/edmir-build/docker.sock']


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(command,log,timeout=900):
    with (RECORDS/log).open('w') as output:
        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def main():
    while True:
        state=json.loads((RECORDS/'image-download-status.json').read_text())
        if state['status']=='verified':break
        if state['status']=='failed':raise RuntimeError('image transfer failed; inspect receipts')
        time.sleep(10)
    layout=ASSETS/'vllm-oci';archive=ASSETS/'vllm-0.20.1-amd64.oci.tar'
    temp=archive.with_suffix('.partial')
    with tarfile.open(temp,'w',format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(layout.rglob('*')):
            if not path.is_file():continue
            if path.name.endswith('.partial'):raise ValueError('incomplete OCI blob present')
            info=tar.gettarinfo(str(path),str(path.relative_to(layout)))
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o444
            with path.open('rb') as stream:tar.addfile(info,stream)
    temp.replace(archive)
    with tarfile.open(archive,'r:') as tar:
        checked=0
        for entry in tar:
            if entry.name.startswith('blobs/sha256/'):
                with tar.extractfile(entry) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=Path(entry.name).name:
                        raise ValueError('OCI archive blob mismatch')
                checked+=1
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'verified_blob_count':checked,'manifest_digest':state['manifest'],
        'config_digest':state['config_digest'],'platform':'linux/amd64','GPU_used':False}
    (RECORDS/'image-archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('OCI archive verified; importing locally',flush=True)
    run([*DOCKER,'load','--input',str(archive)],'docker-load.log')
    run([*DOCKER,'image','inspect',state['config_digest']],'loaded-image-inspect.json')
    inspection=json.loads((RECORDS/'loaded-image-inspect.json').read_text())[0]
    if inspection['Id']!=state['config_digest'] or inspection['Architecture']!='amd64':
        raise ValueError('loaded image identity differs')
    out=RECORDS/'cpu-preflight-attempt01';out.mkdir(exist_ok=False)
    cmd=[*DOCKER,'run','--rm','--pull=never','--network=none','--platform=linux/amd64',
        '--read-only','--cpus=6','--memory=8g','--shm-size=1g',
        '--tmpfs','/tmp:rw,size=3g','--entrypoint','python3','--workdir','/work']
    for value in ['NVIDIA_VISIBLE_DEVICES=void','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                  'VLLM_NO_USAGE_STATS=1','DO_NOT_TRACK=1','HF_HOME=/tmp/hf',
                  'XDG_CACHE_HOME=/tmp/cache','VLLM_CACHE_ROOT=/tmp/vllm',
                  'TRITON_CACHE_DIR=/tmp/triton','OMP_NUM_THREADS=2','TOKENIZERS_PARALLELISM=false']:
        cmd+=['-e',value]
    for host,guest in [(ASSETS/'model','/model:ro'),(REPO/'EDM-IR/tools','/work:ro'),
                       (REPO/'experiments/2026-11-16-next-window','/packet:ro'),(out,'/out')]:
        cmd+=['-v',f'{host}:{guest}']
    cmd+=[state['config_digest'],'/work/prepare_native_inputs.py','--model-dir','/model',
        '--manifest','/packet/model-manifest.json','--requests','/packet/native-requests.json','--out','/out']
    (RECORDS/'cpu-preflight-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    print('Running actual pinned image offline, CPU only',flush=True)
    run(cmd,'cpu-preflight-attempt01.log',timeout=600)
    print('CPU image/tokenizer/API preflight passed',flush=True)


if __name__=='__main__':main()
