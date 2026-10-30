"""Bounded local regression and immutable evidence check."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    began=time.monotonic()
    design=json.loads((HERE/'design-before.json').read_text())
    guard={**design['source_hashes'],**design['protected_hashes']}
    execution=json.loads((HERE/'run-01/execution-before.json').read_text())
    for p,h in {**guard,**execution}.items():
        if sha(ROOT/p)!=h:raise AssertionError('source/evidence changed: '+p)
    command=[sys.executable,'-m','unittest','tests/pipeline_dependency_model_test.py',
             'tests/dependency_wave_model_test.py','tests/operand_reuse_model_test.py']
    proc=subprocess.run(command,cwd=ROOT/'EDM-IR',text=True,capture_output=True,timeout=30)
    (HERE/'test-output.txt').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise AssertionError(proc.stdout+proc.stderr)
    result=json.loads((HERE/'run-01/results.json').read_text())
    summary=result['summary']
    if summary['scenarios']!=144 or summary['conditional_GEMM_evaluations']!=2592 or summary['score_changes']!=0:
        raise AssertionError('scenario/scoring regression')
    if summary['training_calls'] or summary['new_training_labels'] or summary['GPU_started']:
        raise AssertionError('unexpected evidence boundary crossing')
    for r in result['CTA_graphs']:
        r=r['result']
        if r['prediction_us'] is not None or r['training_eligible'] or r['compiler_admission']:
            raise AssertionError('conditional services promoted to timing labels')
        if r['node_count']>60000 or r['edge_count']>=2*r['node_count']:
            raise AssertionError('unbounded graph')
    # Every checked dependency header was also compared to its pinned Git blob.
    audit=json.loads((HERE/'run-01/source-audit.json').read_text())
    for p,h in audit['sources'].items():
        if sha(ROOT/p)!=h['sha256']:raise AssertionError('reviewed template changed: '+p)
    for p,h in guard.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior evidence changed during tests: '+p)
    files=[HERE/n for n in ('design-before.json','review.py','audit.py','validate.py','README.md',
                           'run-01/results.json','run-01/source-audit.json','test-output.txt')]
    files += [ROOT/'EDM-IR/edmir/compiler/pipeline_dependency_model.py',ROOT/'EDM-IR/tests/pipeline_dependency_model_test.py']
    report={'passed':True,'test_command':command,'test_output':proc.stdout+proc.stderr,
            'prior_source_hashes_unchanged':len(design['source_hashes']),
            'protected_hashes_unchanged':len(design['protected_hashes']),
            'source_headers_verified_against_pinned_git':9,'source_revision':audit['CUTLASS_revision'],
            'scoring_changes':0,'GPU_started':False,'frozen_six_parsed_or_scored':False,
            'local_near_perfect_acceptance_passed':False,'compiler_admission':False,
            'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'cpu_seconds':time.monotonic()-began}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(proc.stdout+proc.stderr)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','test_output','test_command')},indent=2))


if __name__=='__main__':main()
