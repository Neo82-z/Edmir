#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace/2026-10-25-regime-gpu-validation
test -f work-recheck/all-primary-finished.txt
test -f /workspace/baseline-extension-finished.txt
python3 -S server.py verify > work/final-package-verification.txt
python3 -S server.py inventory --label final
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > work/final-gpu-processes.txt
python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
import tarfile
root=Path('/workspace/2026-10-25-regime-gpu-validation')
workspace=root.parent
items=[root/'work',root/'work-recheck',*sorted(root.glob('review-*.json')),
       root/'dependencies/nccl/build/lib/libnccl.so.2.28.9']
names=['baseline-extension-01','baseline-extension-review.json','baseline-extension-finished.txt',
       'baseline_extension.py','baseline-extension-plan.json',
       'bootstrap-hour-01.sh','build-hour-01.sh','run-stage-hour-01.sh','recheck-hour-01.sh',
       'recheck-crossings-hour-01.sh','baseline-run-hour-01.sh',
       'bootstrap-hour-01.log','build-hour-01.log','anchors-hour-01.log','crossings-hour-01.log',
       'library-hour-01.log','warmup-hour-01.log','recheck-hour-01.log','recheck-crossings-hour-01.log',
       'baseline-extension-hour-01.log']
items += [workspace/n for n in names]
def rel(p):
    return str(p.relative_to(root) if p.is_relative_to(root) else p.relative_to(workspace))
files={};links={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for item in items:
    if not item.exists():raise ValueError('missing recovery item: '+str(item))
    for p in ([item,*item.rglob('*')] if item.is_dir() else [item]):
        if p.is_symlink():links[rel(p)]=os.readlink(p)
        elif p.is_file():files[rel(p)]=digest(p)
manifest=workspace/'recovery-file-manifest.json'
manifest.write_text(json.dumps({'schema':'edmir.hour-recovery.v1','files':files,'symlinks':links,
    'includes_raw_samples_and_binaries':True,'package_inputs_already_preserved_locally':True},indent=2)+'\n')
archive=workspace/'hour-01-results.tar.gz'
if archive.exists():raise ValueError('fresh archive required')
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
    for item in items:tar.add(item,arcname=rel(item),recursive=True)
    tar.add(manifest,arcname=manifest.name)
sha=digest(archive)
(workspace/'hour-01-results.tar.gz.sha256').write_text(sha+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),'sha256':sha,'bytes':archive.stat().st_size,'files':len(files),'links':len(links)}))
PY
