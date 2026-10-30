#!/usr/bin/env python3
"""Verify the server's archive digest, extract safely, then verify every file."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
EXPECTED='6247311c6b09b10a1b52f5767baf28c6d7b815a0fe6324fca848df9892d33dc9'
archive=HERE/'hour-01-results.tar.gz';out=HERE/'recovered'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(archive)==EXPECTED
assert (HERE/'hour-01-results.tar.gz.sha256').read_text().split()[0]==EXPECTED
if out.exists():raise ValueError('fresh extraction required')
out.mkdir()
with tarfile.open(archive,'r:gz') as tar:tar.extractall(out,filter='data')
manifest=json.loads((out/'recovery-file-manifest.json').read_text())
for rel,digest in manifest['files'].items():
    p=out/rel
    assert p.is_file() and not p.is_symlink() and sha(p)==digest,rel
for rel,target in manifest['symlinks'].items():
    p=out/rel
    assert p.is_symlink() and os.readlink(p)==target,rel
actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() or p.is_symlink()}
assert actual==set(manifest['files'])|set(manifest['symlinks'])|{'recovery-file-manifest.json'}
original=HERE.parent/'2026-10-25-regime-gpu-validation.tar.gz'
original_hash=sha(original)
assert original_hash=='d9739cb102401acf6d5950313bbadddf998cf92fe58f786b7146f7d407548007'
record={'passed':True,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'archive_sha256':EXPECTED,'bytes':archive.stat().st_size,'files':len(manifest['files']),
    'symlinks':len(manifest['symlinks']),'unexpected_files':0,'frozen_upload_archive_unchanged':True,
    'frozen_upload_archive_sha256':original_hash}
(HERE/'recovery-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
