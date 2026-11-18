#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/offline/debs /workspace/offline/wheels
cp /var/cache/apt/archives/*.deb /workspace/offline/debs/
cp /opt/edmir-wheels/*.whl /workspace/offline/wheels/
dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /workspace/offline/installed-packages.tsv
python3 - <<'PY'
import hashlib, json, platform, subprocess
from pathlib import Path
r = Path('/workspace/offline')
def output(*cmd): return subprocess.check_output(cmd, text=True).strip()
manifest = {str(p.relative_to(r)): {'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest(), 'bytes': p.stat().st_size}
            for folder in ('debs', 'wheels') for p in sorted((r/folder).glob('*')) if p.is_file()}
receipt = {'architecture': platform.machine(), 'gpu_requested': False, 'nvcc': output('nvcc', '--version'),
           'cmake': output('cmake', '--version'), 'compiler': output('c++', '--version'),
           'python': output('python3', '--version'), 'files': manifest}
(r/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
(r/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in manifest.items()))
print('Cached', len(manifest), 'offline packages, bytes', sum(x['bytes'] for x in manifest.values()))
PY
