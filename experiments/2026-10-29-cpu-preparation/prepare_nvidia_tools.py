"""Download fixed NVIDIA profiler packages and extract them to persistent storage."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


def main():
    root = Path('/workspace/edmir-assets')
    manifest = json.loads((root / 'nvidia-tools-manifest.json').read_text())
    packages = root / 'software' / 'nvidia-debs'
    extracted = root / 'tools' / 'nvidia'
    packages.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    records = []
    for row in manifest['packages']:
        name = Path(row['Filename']).name
        url = manifest['repository'] + row['Filename']
        path = packages / name
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as output:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                output.write(chunk)
                digest.update(chunk)
        if path.stat().st_size != int(row['Size']) or digest.hexdigest() != row['SHA256']:
            raise ValueError('NVIDIA package hash mismatch: ' + name)
        subprocess.run(['dpkg-deb', '-x', str(path), str(extracted)], check=True, timeout=180)
        records.append({'package': row['Package'], 'version': row['Version'],
                        'sha256_verified': True, 'url': url})
    (root / 'manifests' / 'nvidia-tools-download.json').write_text(json.dumps({
        'packages': records, 'extracted_to': str(extracted),
        'gpu_profiling_validated': False,
    }, indent=2))


if __name__ == '__main__':
    main()
