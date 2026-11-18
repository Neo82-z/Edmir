from pathlib import Path
import hashlib
import json
import time
ROOT=Path(__file__).resolve().parent
files={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    name=str(p.relative_to(ROOT))
    if name=='recovery-manifest.json' or name.endswith('.partial') or '__pycache__' in p.parts:continue
    if not (name.startswith(('results/','session/')) or p.parent==ROOT or name.startswith('build/')):continue
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    files[name]={'bytes':p.stat().st_size,'sha256':h}
tmp=ROOT/'recovery-manifest.partial'
tmp.write_text(json.dumps({'created_epoch':time.time(),'scope':'P1/P2 complete or partial data; no raw failures removed','files':files},indent=2)+'\n')
tmp.replace(ROOT/'recovery-manifest.json')
print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in files.values())}))

