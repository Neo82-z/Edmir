"""Bounded local checks; no fitting, model selection sweep, or device access."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def run():
    checks=HERE/'final-checks'; checks.mkdir(exist_ok=False)
    script="""import unittest
suite=unittest.TestSuite()
for pattern in ('execution_evidence_test.py','resource_geometry_test.py','schedule_space_test.py','backend_test.py'):
 suite.addTests(unittest.defaultTestLoader.discover('EDM-IR/tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
    start=time.monotonic()
    tested=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=60)
    (checks/'tests.stdout.txt').write_text(tested.stdout)
    (checks/'tests.stderr.txt').write_text(tested.stderr)
    if tested.returncode:raise RuntimeError('local tests failed; logs retained')
    count=int(re.search(r'Ran (\d+) tests',tested.stderr).group(1))
    command=[sys.executable,str(ROOT/'EDM-IR/tools/prepare_execution_review.py'),'--out']
    result=subprocess.run(command+[str(HERE/'review')],cwd=ROOT,capture_output=True,text=True,timeout=60)
    (checks/'review.stdout.txt').write_text(result.stdout)
    (checks/'review.stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError('review generation failed')
    with tempfile.TemporaryDirectory(prefix='execution-review-') as tmp:
        other=Path(tmp)/'review'
        result=subprocess.run(command+[str(other)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        if result.returncode:raise RuntimeError(result.stderr)
        for path in (HERE/'review').glob('*.json'):
            a=json.loads(path.read_text());b=json.loads((other/path.name).read_text())
            if path.name=='summary.json':
                a.pop('elapsed_seconds'); b.pop('elapsed_seconds')
            if a!=b:raise AssertionError('review differs on independent process rerun: '+path.name)
    protected=json.loads((HERE/'protected-before.json').read_text())
    changed=[path for path,expected in protected.items() if sha(ROOT/path)!=expected]
    if changed:raise AssertionError('protected artifacts changed: '+repr(changed))
    summary=json.loads((HERE/'review/summary.json').read_text())
    result={'status':'local_checks_passed','tests':count,'old_shapes_checked':summary['old_shapes_checked'],
            'candidate_ledgers_checked':summary['candidate_ledgers_checked'],
            'ledger_traversal_seconds':summary['elapsed_seconds'],
            'review_reproduced_in_separate_process':True,'protected_files_unchanged':len(protected),
            'new_training_runs':0,'new_gpu_runs':0,'independent_performance_validation':False,
            'cli_inference_check':json.loads((HERE/'cli-verification.json').read_text()),
            'initial_issue_resolved':'builtin CLI missed json import; fixed and covered by its new end-to-end test',
            'elapsed_seconds':time.monotonic()-start,
            'verification_scope':'Python logic plus existing CPU C++ runtime/backend checks; no CUDA kernels',
            'check_files':{str(p.relative_to(HERE)):sha(p) for p in sorted(checks.glob('*'))}}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':run()
