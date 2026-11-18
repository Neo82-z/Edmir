"""Download a digest-pinned public amd64 image into a verified OCI layout."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
from bounded_download import download

REPOSITORY = 'vllm/vllm-openai'
DIGEST = 'sha256:bf0486e51765766a069f6eb26a63a06748859a568b24200acd3aa4edb58f3d35'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--mirror', action='store_true', help='Use byte-identical DaoCloud registry mirror')
    a=p.parse_args();blobs=a.root/'blobs/sha256';blobs.mkdir(parents=True,exist_ok=True)
    token={};lock=threading.Lock()
    def headers():
        with lock:
            if time.monotonic()>=token.get('until',0):
                r=requests.get('https://m.daocloud.io/auth/token' if a.mirror else 'https://auth.docker.io/token', params={
                    'service':'docker.m.daocloud.io' if a.mirror else 'registry.docker.io',
                    'scope':f'repository:{REPOSITORY}:pull'},timeout=30)
                r.raise_for_status();data=r.json()
                token.update(value=data['token'],until=time.monotonic()+min(data.get('expires_in',300)-60,240))
            return {'Authorization':'Bearer '+token['value']}
    root=f'https://{"docker.m.daocloud.io" if a.mirror else "registry-1.docker.io"}/v2/{REPOSITORY}'
    r=requests.get(f'{root}/manifests/{DIGEST}',headers={**headers(),
        'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'},timeout=30)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest()!=DIGEST.split(':')[1]:raise ValueError('manifest digest mismatch')
    manifest=r.json();(blobs/DIGEST.split(':')[1]).write_bytes(r.content)
    (a.root/'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}\n')
    descriptor={'mediaType':manifest['mediaType'],'digest':DIGEST,'size':len(r.content),
        'platform':{'architecture':'amd64','os':'linux'},
        'annotations':{'org.opencontainers.image.ref.name':'vllm/vllm-openai:edmir-0.20.1-offline'}}
    (a.root/'index.json').write_text(json.dumps({'schemaVersion':2,'manifests':[descriptor]},indent=2))
    state={'manifest':DIGEST,'repository':REPOSITORY,'GPU_used':False,'status':'downloading','transport':root,
           'verified':{},'failures':{}}
    def save():
        t=a.records/'image-download-status.partial'
        t.write_text(json.dumps(state,indent=2));t.replace(a.records/'image-download-status.json')
    def fetch(row):
        name=row['digest'].split(':')[1];path=blobs/name
        if path.exists() and path.stat().st_size==row['size'] and sha(path)==name:return row
        partial=blobs/(name+'.partial')
        def progress(value):
            with (a.records/'image-ranges.jsonl').open('a') as log:
                log.write(json.dumps({'blob':name,**value})+'\n')
        download(f'{root}/blobs/{row["digest"]}',partial,row['size'],headers=headers,workers=4,
                 chunk_bytes=1024**2,progress=progress)
        if sha(partial)!=name:raise ValueError('blob hash mismatch: '+name)
        partial.replace(path);return row
    save()
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(fetch,row):row['digest'] for row in [manifest['config'],*manifest['layers']]}
        for job in as_completed(jobs):
            digest=jobs[job]
            try:state['verified'][digest]=job.result()
            except Exception as exc:state['failures'][digest]={'type':type(exc).__name__}
            save();print(json.dumps({'completed':len(state['verified']),'failed':len(state['failures']),
                'bytes':sum(x['size'] for x in state['verified'].values())}),flush=True)
    state['status']='failed' if state['failures'] else 'verified'
    if not state['failures']:
        cfg=json.loads((blobs/manifest['config']['digest'].split(':')[1]).read_text())
        if cfg['architecture']!='amd64' or cfg['os']!='linux':raise ValueError('wrong image platform')
        state['config_digest']=manifest['config']['digest']
        state['platform']='linux/amd64'
        (a.records/'image-config.json').write_text(json.dumps(cfg,indent=2))
    save()
    if state['failures']:raise SystemExit(1)


if __name__=='__main__':main()
