"""Create the final recovery manifest after collection/trace processes have exited."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path('/workspace'))
args = p.parse_args()
r = args.root.resolve()
paths = []
for name in ('results-precision', 'session'):
    paths.extend(f for f in (r/name).rglob('*') if f.is_file())
for name in ('receipt.json', 'build.log', 'Testing/Temporary/LastTest.log'):
    paths.append(r/'build-precision'/name)
paths.extend((r/'build-precision').glob('w1_*_batch'))
paths.extend([r/'window/platform-guard.json', r/'window/queue.json', r/'window/run_window.py'])
paths.extend(r/name for name in ('remote-stage.sh', 'host-prepare.sh', 'make-recovery.py',
                                'verify-files.py', 'export-traces.py', 'PAYLOAD_SHA256.json'))
files = {}
for f in sorted(set(paths)):
    if f.is_file():
        files[str(f.relative_to(r))] = {'bytes': f.stat().st_size,
            'sha256': hashlib.file_digest(f.open('rb'), 'sha256').hexdigest()}
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'four_development_shapes_no_fitting_no_frozen_shapes', 'files': files}
temp = r/'recovery-manifest.partial'
temp.write_text(json.dumps(manifest, indent=2)+'\n')
temp.replace(r/'recovery-manifest.json')
print(json.dumps({'files': len(files), 'bytes': sum(f['bytes'] for f in files.values())}))
