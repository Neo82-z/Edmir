"""Cache Linux x86_64 GPU wheels for a separate Ubuntu 24.04/Python 3.12 runtime.

The download host may have older glibc. No GPU environment is installed or run.
"""
import email
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def main():
    root = Path('/workspace/edmir-assets')
    if sys.platform != 'linux' or list(Path('/dev').glob('nvidia[0-9]*')):
        raise SystemExit('A Linux zero-GPU preparation host is required')
    wheels = root / 'software' / 'vllm-0.20.1-cp312-linux-x86_64'
    wheels.mkdir(parents=True, exist_ok=True)
    manifests = root / 'manifests'
    status_path = manifests / 'software-status.json'
    status = {'stage': 'downloading_wheels', 'started_unix': time.time(),
              'target': 'Ubuntu 24.04 x86_64 / CPython 3.12 / NVIDIA GPU',
              'runtime_installed': False, 'GPU_validated': False}
    status_path.write_text(json.dumps(status, indent=2))
    platforms = [f'manylinux_2_{v}_x86_64' for v in range(39, 4, -1)]
    platforms += ['manylinux2014_x86_64', 'manylinux2010_x86_64',
                  'manylinux1_x86_64', 'linux_x86_64']
    options = ['--only-binary=:all:', '--python-version', '3.12',
               '--implementation', 'cp', '--abi', 'cp312', '--abi', 'abi3', '--abi', 'none']
    for item in platforms:
        options += ['--platform', item]
    env = dict(os.environ, PIP_CACHE_DIR=str(root / 'software' / 'pip-cache'),
               PIP_DISABLE_PIP_VERSION_CHECK='1')
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--dest', str(wheels),
                        *options, 'vllm==0.20.1', 'nvtx'], env=env, check=True, timeout=5400)
        records = []
        for path in sorted(wheels.glob('*.whl')):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            with zipfile.ZipFile(path) as archive:
                # Vendored libraries (e.g. setuptools/_vendor) can contain metadata too.
                metadata_paths = [n for n in archive.namelist()
                                  if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
                if len(metadata_paths) != 1:
                    raise ValueError(f'Ambiguous wheel metadata: {path.name}')
                metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
            records.append({'name': metadata['Name'], 'version': metadata['Version'],
                            'filename': path.name, 'bytes': path.stat().st_size,
                            'sha256': digest.hexdigest()})
        (manifests / 'gpu-wheel-manifest.json').write_text(json.dumps(records, indent=2))
        lock = manifests / 'gpu-requirements.lock'
        lock.write_text(''.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in records))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--ignore-installed',
                        '--no-index', '--find-links', str(wheels), '--require-hashes',
                        '--report', str(manifests / 'gpu-offline-resolution.json'),
                        *options, '-r', str(lock)], env=env, check=True, timeout=900)
        status.update(stage='wheel_cache_verified', completed_unix=time.time(),
                      wheels=len(records), total_bytes=sum(r['bytes'] for r in records),
                      offline_dependency_resolution_passed=True)
    except Exception as exc:
        status.update(stage='failed', error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
