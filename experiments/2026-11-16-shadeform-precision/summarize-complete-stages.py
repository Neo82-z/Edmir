"""Summarize audited full stages; never promote incomplete data to a timing label."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
audit = json.loads((root / 'recovery-audit.json').read_text())
report = {'scope': audit['scope'], 'aggregation': 'median of three process medians; each has 100 raw executions',
          'selection_model_trained': False, 'frozen_shapes_evaluated': False, 'stages': {}}
for stage in audit['stages']:
    if not stage['complete']:
        report['stages'][stage['name']] = {'complete': False, 'missing': stage['missing']}
        continue
    folder = root / 'recovered/results-precision' / stage['name']
    checks, times = {}, {}
    for path in sorted(folder.glob('collect-[0-9][0-9][0-9].json')):
        block = json.loads(path.read_text())
        for row in block['rows']:
            family = row['family']
            if block['job']['phase'] == 'check':
                checks.setdefault(family, []).append({'seed': block['job']['seed'],
                                                       'passed': row['correctness_passed'],
                                                       'max_abs_error': row['max_abs_error'],
                                                       'max_rel_error': row['max_rel_error']})
            elif row['correctness_passed']:
                times.setdefault(family, []).append(row['median_ms'])
    assert all(len(v) == 3 for v in checks.values())
    assert all(len(v) == 3 for v in times.values())
    admitted = sorted(f for f, values in checks.items() if all(v['passed'] for v in values))
    assert sorted(times) == admitted
    timings = {f: {'process_medians_ms': values, 'median_ms': statistics.median(values),
                   'min_process_median_ms': min(values), 'max_process_median_ms': max(values)}
               for f, values in times.items()}
    host = min((f for f in times if f.startswith('host')), key=lambda f: timings[f]['median_ms'])
    fastest = min(times, key=lambda f: timings[f]['median_ms'])
    report['stages'][stage['name']] = {
        'complete': True, 'checks': checks, 'admitted': admitted,
        'rejected': sorted(set(checks) - set(admitted)), 'timings': timings,
        'fastest_measured_candidate': fastest, 'strongest_measured_host': host,
        'speedup_over_strongest_measured_host': timings[host]['median_ms'] / timings[fastest]['median_ms'],
        'note': 'Development operator measurements on this node only; not serving speedup or independent generalization.'}
(root / 'stage-summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
for name, stage in report['stages'].items():
    if stage['complete']:
        print(name, 'admitted', len(stage['admitted']), 'fastest', stage['fastest_measured_candidate'],
              'strongest_host', stage['strongest_measured_host'], 'speedup', stage['speedup_over_strongest_measured_host'])
    else:
        print(name, 'INCOMPLETE')
