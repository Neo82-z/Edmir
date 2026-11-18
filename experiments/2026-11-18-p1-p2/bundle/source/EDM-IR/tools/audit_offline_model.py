#!/usr/bin/env python3
"""Verify a pinned sharded model without allocating tensors or loading a GPU."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def inspect_shard(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        header_bytes, = struct.unpack('<Q', prefix)
        if not 2 <= header_bytes <= min(16 * 1024**2, size - 8):
            raise ValueError('invalid safetensors header length')
        header = json.loads(stream.read(header_bytes), object_pairs_hook=unique_object)
    tensors = {}
    spans = []
    counts = Counter()
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        dtype = tensor['dtype']
        if dtype not in ('BF16', 'F32'):
            raise ValueError('unexpected dtype in pinned BF16 model: ' + dtype)
        shape = tensor['shape']
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError('invalid tensor shape')
        begin, end = tensor['data_offsets']
        if type(begin) is not int or type(end) is not int or not 0 <= begin <= end:
            raise ValueError('invalid tensor offsets')
        elements = math.prod(shape)
        if end - begin != elements * {'BF16': 2, 'F32': 4}[dtype]:
            raise ValueError('tensor shape/dtype byte count mismatch: ' + name)
        spans.append((begin, end))
        tensors[name] = {'dtype': dtype, 'shape': shape}
        counts[dtype] += elements
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError('overlap or unaccounted bytes in shard')
        cursor = end
    if not tensors or cursor + 8 + header_bytes != size:
        raise ValueError('truncated or trailing shard payload')
    return tensors, counts, cursor


def audit(manifest, model_dir):
    actual_map = {}
    records = []
    counts = Counter()
    payload_bytes = 0
    for name, expected in manifest['auxiliary_hashes'].items():
        if Path(name).name != name or digest(model_dir / name) != expected:
            raise ValueError('pinned metadata mismatch: ' + name)
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text(),
                       object_pairs_hook=unique_object)
    expected_names = {r['rfilename'] for r in manifest['weight_files']}
    if set(index['weight_map'].values()) != expected_names:
        raise ValueError('index and pinned shard inventory differ')
    for row in manifest['weight_files']:
        name = row['rfilename']
        if Path(name).name != name:
            raise ValueError('unsafe shard path')
        path = model_dir / name
        if path.stat().st_size != row['size'] or digest(path) != row['lfs']['sha256']:
            raise ValueError('pinned weight mismatch: ' + name)
        tensors, local_counts, data_bytes = inspect_shard(path)
        if actual_map.keys() & tensors.keys():
            raise ValueError('tensor appears in more than one shard')
        actual_map.update({key: name for key in tensors})
        counts.update(local_counts)
        payload_bytes += data_bytes
        records.append({'file': name, 'sha256': row['lfs']['sha256'],
                        'bytes': row['size'], 'tensors': len(tensors)})
    if actual_map != index['weight_map']:
        raise ValueError('tensor headers do not match the weight index')
    reported_size = index['metadata']['total_size']
    warnings = []
    if payload_bytes != reported_size:
        # Some pinned upstream indices put the element count in total_size.
        # Keep that inconsistency visible; file hashes, every tensor offset,
        # dtype byte count and the complete weight_map must still agree.
        if (type(reported_size) is int and reported_size == sum(counts.values())
                and manifest.get('index_reported_total_size') == reported_size
                and 'model.safetensors.index.json' in manifest['auxiliary_hashes']):
            warnings.append({'code': 'index_total_size_matches_parameter_count',
                             'index_reported_total_size': reported_size,
                             'actual_tensor_payload_bytes': payload_bytes,
                             'actual_parameter_count': sum(counts.values()),
                             'upstream_index_modified': False})
        else:
            raise ValueError('index total_size and tensor payload differ')
    return {'status': 'verified_with_metadata_warning' if warnings else 'verified',
            'warnings': warnings, 'index_byte_size_consistent': not warnings,
            'model_id': manifest['id'], 'revision': manifest['revision'],
            'GPU_used': False, 'full_model_loaded': False,
            'method': 'full SHA256 plus complete tensor header/index/dtype/offset audit',
            'shards': records, 'tensors': len(actual_map), 'parameters_by_dtype': dict(counts),
            'tensor_payload_bytes': payload_bytes,
            'file_bytes': sum(row['bytes'] for row in records),
            'inference_correctness_verified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        result = audit(json.loads(a.manifest.read_text()), a.model_dir)
        result['manifest_sha256'] = digest(a.manifest)
    except Exception as exc:
        a.out.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2) + '\n')
        raise
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'shards'}))


if __name__ == '__main__':
    main()
