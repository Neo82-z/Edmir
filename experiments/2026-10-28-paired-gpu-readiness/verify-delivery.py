from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
manifest=json.loads((root/'delivery.json').read_text())
for rel,expected in manifest['files'].items():
    p=Path(rel)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe delivery path')
    with (root/p).open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('delivery changed: '+rel)
print('verified',len(manifest['files']),'files; GPU validation remains pending')
