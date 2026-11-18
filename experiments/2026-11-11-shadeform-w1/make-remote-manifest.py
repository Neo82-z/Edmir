"""Run after the collector/diagnostic exits, before copying and terminating VM."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path('/workspace')
paths = []
for name in ('results', 'session', 'session-attempt1'):
    paths.extend(p for p in (root / name).rglob('*') if p.is_file())
paths.extend(root / 'build' / name for name in ('receipt.json', 'build.log'))
paths.extend((root / 'build').glob('w1_*_batch'))
files = {}
for path in sorted(set(paths)):
    if not path.is_file():
        continue
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    files[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest}
manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'files': files}
(root / 'recovery-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Manifest files:', len(files))
