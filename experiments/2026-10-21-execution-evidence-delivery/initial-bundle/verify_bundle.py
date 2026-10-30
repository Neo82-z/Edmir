import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
expected=manifest['files']
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
if actual!=set(expected):raise SystemExit('file set differs from manifest')
for rel,digest in expected.items():
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise SystemExit('unsafe manifest path')
 if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest:raise SystemExit('hash mismatch: '+rel)
print(json.dumps({'verified_files':len(expected),'status':'snapshot_identity_matches','gpu_validation':False}))
