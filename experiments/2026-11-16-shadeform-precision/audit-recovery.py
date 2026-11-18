"""Audit completed stage records, while retaining incomplete/failed work."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BUNDLE = REPO / 'experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2'
sys.path.insert(0, str(BUNDLE / 'source/EDM-IR/tools'))
from run_w1_gpu import validate_rows, done_raw_changed, require_successful_pilot

def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

recovered = ROOT / 'recovered'
manifest = json.loads((recovered / 'recovery-manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = recovered / name
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
queue = json.loads((recovered / 'window/queue.json').read_text())
assert sha(recovered / 'window/queue.json') == sha(REPO / 'experiments/2026-11-14-next-window/queue.json')
assert sha(recovered / 'window/run_window.py') == sha(REPO / 'experiments/2026-11-14-next-window/run_window.py')
source = json.loads((BUNDLE / 'plan.json').read_text())
receipt_path = recovered / 'build-precision/receipt.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['package'] == sha(BUNDLE / 'files.json') == queue['source_manifest_sha256']
for tag, expected in receipt['binaries'].items():
    assert sha(recovered / f'build-precision/w1_{tag}_batch') == expected

stages, failures, timing_rows, check_rows, samples = [], [], 0, 0, 0
for stage in queue['stages']:
    out = recovered / 'results-precision' / stage['name']
    state = out / 'context.json'
    completed, missing, invalid = [], [], set()
    pilot_valid = False
    if state.exists():
        context = json.loads(state.read_text())
        assert context['build'] == sha(receipt_path)
        assert context['package'] == receipt['package']
        if (out / 'pilot-complete.json').exists():
            require_successful_pilot(out, sha(state), source, context)
            pilot_valid = True
    for index, job in enumerate(stage['jobs']):
        path = out / f'collect-{index:03d}.json'
        if not path.exists():
            missing.append({'index': index, 'phase': job['phase'], 'seed': job['seed']})
            continue
        record = json.loads(path.read_text())
        assert pilot_valid, 'Completed collection without valid pilot'
        assert record['job'] == job and record['context'] == sha(state)
        expected_order = [f for f in job['order'] if job['phase'] == 'check' or f not in invalid]
        assert record['order'] == expected_order
        assert record['excluded_by_numerical_check'] == [f for f in job['order'] if f not in expected_order]
        assert not done_raw_changed(path, out)
        validate_rows(record['rows'], job, record['order'], source['shapes'][job['shape']], context['runtime'])
        completed.append({'index': index, 'phase': job['phase'], 'rows': len(record['rows']),
                          'excluded': record['excluded_by_numerical_check']})
        for row in record['rows']:
            if job['phase'] == 'check':
                check_rows += 1
            elif row['correctness_passed']:
                timing_rows += 1
                samples += len(row['samples_ms'])
            if not row['correctness_passed']:
                invalid.add(row['family'])
                failures.append({'shape': job['shape'], 'seed': job['seed'], 'family': row['family'],
                                 'failure': row['failure'], 'max_abs_error': row.get('max_abs_error'),
                                 'max_rel_error': row.get('max_rel_error')})
    completion = out / 'collect-complete.json'
    if completion.exists():
        assert not missing
        assert json.loads(completion.read_text())['attempted_groups'] == len(stage['jobs'])
    traces = {}
    for case in stage['traces']:
        trace = out / 'diagnostic' / case
        trace_receipt = trace / 'receipt.json'
        if trace_receipt.exists():
            assert sha(trace / 'timeline.nsys-rep') == json.loads(trace_receipt.read_text())['report_sha256']
            traces[case] = 'hash_verified_contents_require_review'
        elif (out / f'{case}-failure.json').exists():
            traces[case] = json.loads((out / f'{case}-failure.json').read_text())
        else:
            traces[case] = 'not_completed'
    stages.append({'name': stage['name'], 'pilot_valid': pilot_valid, 'complete': completion.exists() and not missing,
                   'completed': completed, 'missing': missing, 'traces': traces})
report = {'scope': 'development_only_no_fitting_no_frozen_shapes',
          'verified_files': len(manifest['files']), 'stages': stages,
          'completed_groups': sum(len(s['completed']) for s in stages),
          'planned_groups': sum(len(s['jobs']) for s in queue['stages']),
          'correctness_check_rows': check_rows, 'valid_timing_rows': timing_rows,
          'raw_timing_samples': samples, 'numerical_failures': failures,
          'complete_protocol': all(s['complete'] for s in stages)}
(ROOT / 'recovery-audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('stages','numerical_failures')}))
print('Numerical failure records:', len(failures))
