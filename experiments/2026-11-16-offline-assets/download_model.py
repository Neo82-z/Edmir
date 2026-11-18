"""Stage exactly one pinned public model on local disk, then verify all files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--mirror-inventory',type=Path)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);a.records.mkdir(parents=True,exist_ok=True)
    if list(Path('/dev').glob('nvidia[0-9]*')):raise RuntimeError('CPU-only preparation required')
    manifest=json.loads(a.manifest.read_text())
    if manifest['id']!='zai-org/GLM-4.7-Flash' or manifest['revision']!='7dd20894a642a0aa287e9827cb1a1f7f91386b67':
        raise ValueError('only the declared model and fixed revision are allowed')
    expected={r['rfilename']:dict(sha256=r['lfs']['sha256'],bytes=r['size']) for r in manifest['weight_files']}
    expected.update({name:dict(sha256=value) for name,value in manifest['auxiliary_hashes'].items()})
    destination=a.root/'model'
    destination.mkdir(exist_ok=True)
    remaining=sum(v.get('bytes',0) for k,v in expected.items() if not (destination/k).exists())
    if shutil.disk_usage(a.root).free < remaining+40*1024**3:raise RuntimeError('insufficient disk space')
    os.environ['HF_HOME']=str(a.root/'hf-home')
    os.environ['HF_XET_CACHE']=str(a.root/'hf-xet')
    os.environ['HF_XET_HIGH_PERFORMANCE']='1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    mirror={}
    if a.mirror_inventory:
        mirror={r['Path']:r for r in json.loads(a.mirror_inventory.read_text())['Data']['Files']}
        for row in manifest['weight_files']:
            r=mirror[row['rfilename']]
            if r['Sha256']!=row['lfs']['sha256'] or r['Size']!=row['size']:
                raise ValueError('mirror is not byte-identical to pinned HF weights')
    state=dict(model_id=manifest['id'],revision=manifest['revision'],manifest_sha256=sha(a.manifest),
        started_utc=datetime.now(timezone.utc).isoformat(),huggingface_hub=huggingface_hub.__version__,
        GPU_used=False,files={},failures={},status='downloading',
        transport='ModelScope official mirror, SHA-bound to HF revision' if mirror else 'HF Xet')
    def fetch(name,meta):
        target=destination/name
        if target.is_file() and ('bytes' not in meta or target.stat().st_size==meta['bytes']) and sha(target)==meta['sha256']:
            return dict(**meta,bytes=target.stat().st_size) if 'bytes' not in meta else meta
        for attempt in range(3):
            try:
                if mirror and name.endswith('.safetensors'):
                    from bounded_download import download
                    partial=destination/(name+'.http.partial')
                    offset=partial.stat().st_size if partial.exists() else 0
                    if offset>meta['bytes']:raise ValueError('partial is larger than the pinned shard')
                    if offset<meta['bytes']:
                        url=f"https://modelscope.cn/models/ZhipuAI/GLM-4.7-Flash/resolve/{mirror[name]['Revision']}/{name}"
                        download(url,partial,meta['bytes'],workers=4)
                    local=partial
                else:
                    local=Path(hf_hub_download(repo_id=manifest['id'],revision=manifest['revision'],filename=name,
                               local_dir=str(destination)))
                if ('bytes' in meta and local.stat().st_size!=meta['bytes']) or sha(local)!=meta['sha256']:
                    raise ValueError('downloaded file digest/size differs from pinned manifest: '+name)
                if local!=target:local.replace(target)
                return dict(sha256=meta['sha256'],bytes=target.stat().st_size)
            except Exception:
                if attempt==2:raise
                time.sleep(2*(attempt+1))
    with (a.root/'.model-download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(a.records/'model-download-status.json',state)
        with ThreadPoolExecutor(max_workers=8 if mirror else 4) as pool:
            futures={pool.submit(fetch,name,meta):name for name,meta in expected.items()}
            for f in as_completed(futures):
                name=futures[f]
                try:state['files'][name]=f.result()
                except Exception as exc:state['failures'][name]=dict(type=type(exc).__name__,message=str(exc))
                state['verified_bytes']=sum(x['bytes'] for x in state['files'].values())
                save(a.records/'model-download-status.json',state)
                print(json.dumps(dict(completed=len(state['files']),total=len(expected),failed=len(state['failures']),
                    verified_GB=round(state['verified_bytes']/1e9,3),last_file=name)),flush=True)
        state['status']='failed' if state['failures'] else 'complete'
        state['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(a.records/'model-download-status.json',state)
        if state['failures']:raise SystemExit(1)


if __name__=='__main__':main()
