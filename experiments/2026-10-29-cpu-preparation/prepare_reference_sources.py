"""Archive reviewed upstream repositories at fixed commits; do not build them."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def main():
    root = Path('/workspace/edmir-assets')
    rows = json.loads((root / 'source-manifest.json').read_text())
    destination = root / 'sources'
    destination.mkdir(exist_ok=True)
    for row in rows:
        filename = row['repository'].replace('/', '--') + '-' + row['commit'] + '.tar.gz'
        path = destination / filename
        with urllib.request.urlopen(row['archive_url'], timeout=120) as response, path.open('wb') as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                output.write(chunk)
        row['archive_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        row['archive_bytes'] = path.stat().st_size
        with tarfile.open(path) as archive:
            archive.extractall(destination, filter='data')
        row['submodules_included'] = False
        print(row['repository'], row['commit'], 'saved', flush=True)
    (root / 'manifests' / 'source-downloads.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
