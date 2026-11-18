"""Complete the current model download with a CPU-only structural audit."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/neo/code/llm-serving')
RECORDS=ROOT/'experiments/2026-11-16-offline-assets'
MODEL=Path('/Users/neo/edmir-assets/2026-11-16/model')
PACKET=ROOT/'experiments/2026-11-16-next-window'


def main():
    while True:
        state=json.loads((RECORDS/'model-download-status.json').read_text())
        if state['status']=='complete':break
        if state['status']=='failed':raise RuntimeError('model download failed; inspect receipts')
        time.sleep(10)
    with (RECORDS/'model-audit.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'EDM-IR/tools/audit_offline_model.py'),
            '--model-dir',str(MODEL),'--manifest',str(PACKET/'model-manifest.json'),
            '--out',str(RECORDS/'model-audit.json')],check=True,timeout=1200,stdout=log,stderr=subprocess.STDOUT)
    print('All model bytes, tensor headers and weight index verified',flush=True)
    with (RECORDS/'native-asset-review.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'EDM-IR/tools/run_native_moe_smoke.py'),
            '--model-dir',str(MODEL),'--manifest',str(PACKET/'model-manifest.json'),
            '--requests',str(PACKET/'native-requests.json'),
            '--out',str(RECORDS/'native-asset-review'),'--verify-hashes'],
            check=True,timeout=1200,stdout=log,stderr=subprocess.STDOUT)
    print('Native entry-point asset verification passed (no GPU)',flush=True)


if __name__=='__main__':main()
