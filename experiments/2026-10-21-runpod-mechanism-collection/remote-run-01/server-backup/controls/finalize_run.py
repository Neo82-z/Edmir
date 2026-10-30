"""Seal and back up completed work without altering or stopping the RunPod."""
from pathlib import Path
import hashlib, json, subprocess, tarfile, time

root = Path(__file__).resolve().parent
work = root/'work'
for name in ('anchors','mechanisms','geometry','short-host-check','compiled-resources-02','profiles'):
    assert json.loads((work/f'{name}.status.json').read_text())['state']=='complete', name
verify = subprocess.run(['python3',str(root/'server-r01.py'),'verify'],
                        capture_output=True,text=True,timeout=120,check=True)
processes = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader'],capture_output=True,text=True,timeout=30,check=True)
receipt = {'verified_package':json.loads(verify.stdout),'completed_unix':time.time(),
           'gpu_compute_processes_at_completion':processes.stdout,
           'pod_stopped':False,'primary_processes':282,'primary_timed_samples':28200,
           'additional_unprofiled_diagnostic_processes':24,'profiled_diagnostic_processes':4,
           'no_model_training':True,'compiler_admission':False}
(work/'post-run-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')

files = [(p,Path('work')/p.relative_to(work)) for p in sorted(work.rglob('*')) if p.is_file()]
files += [(p,Path('controls')/p.name) for p in sorted(root.glob('*.py'))]
files += [(root/'revision-01.json',Path('controls/revision-01.json'))]
library=root/'dependencies/nccl/build/lib/libnccl.so.2.28.9'
files.append((library,Path('dependencies/libnccl.so.2.28.9')))
manifest = {str(name):{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
            for path,name in files}
manifest_path=root/'result-file-manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
archive=root/'completed-run-01.tar.gz'
if archive.exists():raise FileExistsError(archive)
with tarfile.open(archive,'w:gz',compresslevel=6,dereference=True) as tar:
    for path,name in files:tar.add(path,arcname=str(name),recursive=False)
    tar.add(manifest_path,arcname=manifest_path.name)
sealed={'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'bytes':archive.stat().st_size,'files':len(files), 'receipt':receipt}
(root/'completed-run-01.json').write_text(json.dumps(sealed,indent=2)+'\n')
print(json.dumps(sealed),flush=True)
