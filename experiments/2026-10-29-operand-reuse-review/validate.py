"""Bounded local verification; frozen evidence is hashed, never parsed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    design = json.loads((HERE / 'design-before.json').read_text())
    guards = {**design['source_hashes'], **design['protected_hashes']}
    for path, digest in guards.items():
        if sha(ROOT / path) != digest:
            raise AssertionError('prior file changed: ' + path)
    executed = json.loads((HERE / 'run-01/execution-before.json').read_text())
    for path, digest in executed.items():
        if sha(ROOT / path) != digest:
            raise AssertionError('review implementation changed after run: ' + path)
    tests = ['operand_reuse_model', 'dependency_wave_model', 'resource_service_simulator', 'shape_transfer_selection']
    command = [sys.executable, '-m', 'unittest', *['tests/' + p + '_test.py' for p in tests]]
    proc = subprocess.run(command, cwd=ROOT / 'EDM-IR', capture_output=True, text=True, timeout=30)
    (HERE / 'test-output.txt').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise AssertionError('local tests failed; see test-output.txt')
    audit = subprocess.run([sys.executable, str(HERE / 'audit.py')], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
    if audit.returncode:
        raise AssertionError(audit.stdout + audit.stderr)
    result = json.loads((HERE / 'run-01/results.json').read_text())
    if len(result['scenarios']) != 320 or result['summary']['score_changes'] != 0:
        raise AssertionError('incomplete scenario or scoring control')
    for r in result['scenarios']:
        if r['prediction_us'] is not None or r['chosen'] is not None or r['training_eligible']:
            raise AssertionError('conditional model crossed evidence boundary')
        if r['references'] > 1_000_000 or r['peak_resident_panels'] > 100_000:
            raise AssertionError('resource budget violated')
        if r['peak_resident_bytes'] > r['scenario']['capacity_bytes']:
            raise AssertionError('cache capacity violated')
        for p in r['passes']:
            for s in p['operands'].values():
                if s['read_bytes'] != s['read_hit_bytes'] + s['read_miss_bytes']:
                    raise AssertionError('read conservation violated')
    for path, digest in guards.items():
        if sha(ROOT / path) != digest:
            raise AssertionError('prior file changed during verification: ' + path)
    files = [HERE / 'design-before.json', HERE / 'review.py', HERE / 'audit.py', Path(__file__),
             HERE / 'run-01/results.json', HERE / 'run-01/audit.json', HERE / 'test-output.txt',
             ROOT / 'EDM-IR/edmir/compiler/operand_reuse_model.py', ROOT / 'EDM-IR/tests/operand_reuse_model_test.py']
    status = {'passed': True, 'test_command': command, 'test_output': proc.stdout + proc.stderr,
              'prior_source_hashes_unchanged': len(design['source_hashes']),
              'protected_hashes_unchanged': len(design['protected_hashes']),
              'scenario_count': 320, 'score_changes': 0, 'GPU_started': False,
              'frozen_six_parsed_or_scored': False, 'training_calls': [],
              'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
              'cpu_seconds': time.monotonic() - start}
    (HERE / 'verification.json').write_text(json.dumps(status, indent=2) + '\n')
    print(proc.stdout + proc.stderr)
    print(json.dumps({k: v for k, v in status.items() if k not in ('files', 'test_output', 'test_command')}, indent=2))


if __name__ == '__main__':
    main()
