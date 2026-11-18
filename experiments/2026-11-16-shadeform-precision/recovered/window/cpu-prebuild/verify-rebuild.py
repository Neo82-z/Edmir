import json
import subprocess
import time
from pathlib import Path

r = Path('/workspace')
receipt = {'started_epoch': time.time(), 'gpu_requested': False, 'status': 'RUNNING'}
try:
    subprocess.run(['python3', '/assets/rebuild_inputs.py', '--out', '/workspace/rebuilt-inputs'],
                   check=True, timeout=900)
    subprocess.run(['python3', '/workspace/precision-bundle/source/EDM-IR/tools/run_w1_gpu.py',
                    'verify', '--bundle', '/workspace/precision-bundle', '--inputs', '/workspace/rebuilt-inputs'],
                   check=True, timeout=120)
    receipt.update(status='PASS', verified_shape_seed_pairs=12, matched_sealed_v2=True)
except BaseException as error:
    receipt.update(status='FAILED', error=str(error))
    raise
finally:
    receipt.update(finished_epoch=time.time(), elapsed_seconds=time.time()-receipt['started_epoch'])
    (r/'input-rebuild-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
