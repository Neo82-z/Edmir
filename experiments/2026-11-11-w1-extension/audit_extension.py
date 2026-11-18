#!/usr/bin/env python3
"""Audit acquisition records locally; never fit a model or rank candidates."""
import argparse
import json
from pathlib import Path
import sys

from run_extension import sha, validate_plan, save


def audit(bundle, recovery, plan_path):
    plan = json.loads(plan_path.read_text())
    source = json.loads((bundle / 'plan.json').read_text())
    validate_plan(plan, source, bundle)
    sys.path.insert(0, str(bundle / 'source/EDM-IR/tools'))
    from run_w1_gpu import validate_rows, done_raw_changed
    root = recovery / 'results'
    intent = json.loads((root / 'intent.json').read_text())
    assert intent['plan'] == plan
    context = intent['context']
    assert context['extension_identity']['plan'] == sha(plan_path)
    assert context['extension_identity']['runner'] == sha(plan_path.parent / 'run_extension.py')
    receipt = recovery / 'build/receipt.json'
    assert sha(receipt) == context['build']
    build = json.loads(receipt.read_text())
    assert build['package'] == context['package'] == sha(bundle / 'files.json')
    for name, expected in build['binaries'].items():
        assert sha(recovery / f'build/w1_{name}_batch') == expected
    count = {'completed_groups': 0, 'check_rows': 0, 'numerical_failure_records': 0,
             'valid_timing_rows': 0, 'raw_timing_samples': 0}
    missing, diagnostics = [], {}
    for stage in plan['stages']:
        out = root / stage['name']
        invalid = set()
        for i, job in enumerate(stage['jobs']):
            path = out / f'collect-{i:03d}.json'
            if not path.exists():
                missing.append({'stage': stage['name'], 'job': i})
                continue
            record = json.loads(path.read_text())
            assert json.loads((out / 'context.json').read_text()) == context
            assert record['context'] == sha(out / 'context.json')
            assert record['job'] == job and not done_raw_changed(path, out)
            order = [f for f in job['order'] if f not in invalid or job['phase'] == 'check']
            assert record['order'] == order
            assert record['excluded_by_numerical_check'] == [f for f in job['order'] if f not in order]
            validate_rows(record['rows'], job, order, plan['shapes'][stage['name']], context['runtime'])
            count['completed_groups'] += 1
            for row in record['rows']:
                if job['phase'] == 'check':
                    count['check_rows'] += 1
                elif row['correctness_passed']:
                    count['valid_timing_rows'] += 1
                    count['raw_timing_samples'] += len(row['samples_ms'])
                if not row['correctness_passed']:
                    invalid.add(row['family'])
                    count['numerical_failure_records'] += 1
        for case in stage['traces']:
            dest = out / 'diagnostic' / case
            report = dest / 'timeline.nsys-rep'
            note = dest / 'receipt.json'
            if not note.is_file() or not report.is_file():
                diagnostics[case] = 'missing_or_failed'
                continue
            assert json.loads(note.read_text())['report_sha256'] == sha(report)
            assert json.loads((dest / 'intent.json').read_text())['context'] == sha(out / 'context.json')
            diagnostics[case] = 'file_verified_contents_require_review'
    return {**count, 'missing': missing, 'diagnostics': diagnostics,
            'all_30_groups_present': count['completed_groups'] == 30 and not missing,
            'trace_files_complete': len(diagnostics) == 4 and all(
                v == 'file_verified_contents_require_review' for v in diagnostics.values()),
            'pilot_samples_excluded': True, 'independent_generalization': False,
            'real_moe_serving': False, 'cross_session_pooling_performed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.bundle.resolve(), a.recovery.resolve(), Path(__file__).with_name('plan.json'))
    save(a.recovery / 'audit.json', result)
    print(json.dumps(result, indent=2))
