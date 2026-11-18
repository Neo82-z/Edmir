"""Audit recovered W1 records without changing candidates or fitting a model."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = Path('/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01/bundle')
sys.path.insert(0, str(BUNDLE / 'source/EDM-IR/tools'))
from run_w1_gpu import validate_rows, done_raw_changed

def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

out = ROOT / 'recovered/results'
plan = json.loads((BUNDLE / 'plan.json').read_text())
context = json.loads((out / 'context.json').read_text())
context_hash = sha(out / 'context.json')
receipt_path = ROOT / 'recovered/build/receipt.json'
receipt = json.loads(receipt_path.read_text())
assert sha(receipt_path) == context['build']
assert receipt['package'] == sha(BUNDLE / 'files.json') == context['package']
for tag, expected in receipt['binaries'].items():
    assert sha(ROOT / f'recovered/build/w1_{tag}_batch') == expected
complete, missing, failures = [], [], []
timing_rows = 0
raw_samples = 0
check_rows = 0
for index, job in enumerate(plan['jobs']):
    path = out / f'collect-{index:03d}.json'
    if not path.exists():
        missing.append({'index': index, 'job': job})
        continue
    record = json.loads(path.read_text())
    assert record['job'] == job
    assert record['context'] == context_hash
    assert not done_raw_changed(path, out)
    assert record['order'] == [r['family'] for r in record['rows']]
    validate_rows(record['rows'], job, record['order'], plan['shapes'][job['shape']], context['runtime'])
    complete.append({'index': index, 'shape': job['shape'], 'phase': job['phase'],
                     'rows': len(record['rows']), 'excluded': record['excluded_by_numerical_check']})
    for row in record['rows']:
        if job['phase'] == 'check':
            check_rows += 1
        elif row['correctness_passed']:
            timing_rows += 1
            raw_samples += len(row['samples_ms'])
        if not row['correctness_passed']:
            failures.append({'index': index, 'shape': job['shape'], 'seed': job['seed'],
                             'family': row['family'], 'failure': row['failure'],
                             'max_abs_error': row.get('max_abs_error'),
                             'max_rel_error': row.get('max_rel_error')})

completion_path = out / 'collect-complete.json'
completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
if completion is not None:
    assert completion['attempted_groups'] == len(plan['jobs'])
    assert completion['valid_set_only'] is True
    assert not missing
summary = dict(scope='development_data_only_no_fitting_no_frozen_shapes',
               identity_and_completed_raw_records_verified=True,
               collector_complete_receipt_exists=completion is not None,
               complete_protocol=completion is not None and not missing,
               completed_groups=len(complete), planned_groups=len(plan['jobs']),
               correctness_check_rows=check_rows, valid_timing_rows=timing_rows,
               raw_timing_samples=raw_samples, complete=complete, missing=missing,
               numerical_failures=failures)
(ROOT / 'recovery-audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('complete', 'missing', 'numerical_failures')}, ensure_ascii=False))
print('Numerical failure records:', len(failures))
