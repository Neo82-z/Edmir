#!/usr/bin/env python3
"""Verify the delivered files before building; requires only Python stdlib."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
for rel, expected in manifest['files'].items():
    path = root / rel
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or unexpected file type: ' + rel)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError('changed delivery file: ' + rel)
for rel, target in manifest['symlinks'].items():
    path = root / rel
    if not path.is_symlink() or str(path.readlink()) != target:
        raise ValueError('changed delivery symlink: ' + rel)
print(json.dumps({'verified_files': len(manifest['files']),
                  'verified_symlinks': len(manifest['symlinks']),
                  'GPU_validated': False}))
