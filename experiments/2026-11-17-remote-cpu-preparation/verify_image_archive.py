#!/usr/bin/env python3
"""Check every archived blob and its pinned manifest/config without extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

MANIFEST = 'sha256:bf0486e51765766a069f6eb26a63a06748859a568b24200acd3aa4edb58f3d35'
CONFIG = 'sha256:aef4ebc906574fa2e6e52079937b7d3a8735e218f6353b58667b215e4257a34b'


def verify(archive):
    metadata = {}
    blobs = {}
    names = set()
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            if not member.isfile():
                if not member.isdir():
                    raise ValueError('archive contains links or special files')
                continue
            if member.name in names:
                raise ValueError('duplicate archive member')
            names.add(member.name)
            with tar.extractfile(member) as stream:
                if member.name.startswith('blobs/sha256/'):
                    expected = Path(member.name).name
                    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                        raise ValueError('invalid digest filename')
                    h = hashlib.sha256()
                    retain = expected in (MANIFEST.split(':')[1], CONFIG.split(':')[1])
                    if retain and member.size > 1024 * 1024:
                        raise ValueError('oversized image metadata')
                    data = bytearray()
                    while chunk := stream.read(1024 * 1024):
                        h.update(chunk)
                        if retain:
                            data.extend(chunk)
                    if h.hexdigest() != expected:
                        raise ValueError('archive blob hash mismatch: ' + expected)
                    blobs['sha256:' + expected] = member.size
                    if retain:
                        metadata['sha256:' + expected] = json.loads(data)
                elif member.name in ('index.json', 'oci-layout', 'manifest.json'):
                    if member.size > 1024 * 1024:
                        raise ValueError('oversized archive metadata')
                    metadata[member.name] = json.load(stream)
                else:
                    raise ValueError('unexpected archive member: ' + member.name)
    index = metadata['index.json']
    if len(index['manifests']) != 1 or index['manifests'][0]['digest'] != MANIFEST:
        raise ValueError('wrong image index identity')
    manifest = metadata[MANIFEST]
    if manifest['config']['digest'] != CONFIG:
        raise ValueError('wrong image config identity')
    expected_blobs = {MANIFEST}
    for ref in [manifest['config'], *manifest['layers']]:
        if blobs.get(ref['digest']) != ref['size']:
            raise ValueError('missing or wrong-sized manifest dependency')
        expected_blobs.add(ref['digest'])
    if set(blobs) != expected_blobs:
        raise ValueError('unreferenced or missing image blobs')
    cfg = metadata[CONFIG]
    if (cfg['os'], cfg['architecture']) != ('linux', 'amd64'):
        raise ValueError('wrong image architecture')
    with archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(status='verified', archive=str(archive), bytes=archive.stat().st_size,
                sha256=digest, manifest_digest=MANIFEST, config_digest=CONFIG,
                platform='linux/amd64', verified_blob_count=len(blobs),
                all_blob_hashes_verified=True, manifest_dependencies_complete=True,
                GPU_used=False, reload_verified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.archive)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
