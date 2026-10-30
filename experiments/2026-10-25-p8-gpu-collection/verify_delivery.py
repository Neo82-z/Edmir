from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
d=json.loads((r/'delivery.json').read_text())
for rel,h in d['files'].items():
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts: raise ValueError('unsafe path')
 with (r/p).open('rb') as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=h:raise ValueError('changed: '+rel)
print('delivery verified',len(d['files']))
