"""Four separately stored diagnostics after all unprofiled collections finish."""
from pathlib import Path
import json, sys

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root/'bundle-r01/source/EDM-IR'), str(root/'bundle-r01/source/EDM-IR/tools')]
from run_selector_gpu import command, write_json

for phase in ('anchors', 'mechanisms', 'geometry', 'short-host-check'):
    if json.loads((root/'work'/f'{phase}.status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Unprofiled collection not complete: '+phase)

p = json.loads((root/'bundle-r01/package.json').read_text())['package']
selected = [
    (78, 'm512q2', 'w004', 'splitk2@prepared'),
    (32, 'm512q4', 'w004', 'splitk4@prepared'),
    (37, 'largenhost', 'w001', 'host@prepared'),
    (34, 'largenq2', 'w001', 'splitk2@prepared'),
]
for index, label, workload, candidate in selected:
    j = p['jobs'][index]
    if (j['workload_id'], j['candidate_id']) != (workload, candidate):
        raise ValueError('Frozen plan index mismatch')
    print('START', label, index, flush=True)
    command([sys.executable, root/'server-r01.py', 'trace', '--job-index', str(index), '--label', label], timeout=240)
    diagnostic = json.loads((root/'work'/f'trace-{label}'/'diagnostic.json').read_text())
    if diagnostic['state'] != 'complete' or diagnostic['profiler'] is not True:
        raise ValueError('Incomplete or mislabelled profiler output')
    print('COMPLETE', label, flush=True)

write_json(root/'work/profile-batch.json', {'selected': selected, 'state': 'complete',
    'primary_dataset': False, 'compiler_admission': False})
