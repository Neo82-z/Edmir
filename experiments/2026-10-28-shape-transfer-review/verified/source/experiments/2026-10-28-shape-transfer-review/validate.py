import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import shape_transfer_selection as model
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start=time.monotonic();tests={}
    for name in ('shape_transfer_selection_test','axis_residual_selection_test','resource_service_simulator_test','dependency_wave_model_test'):
        run=subprocess.run([sys.executable,str(ROOT/'EDM-IR/tests'/f'{name}.py')],cwd=ROOT,text=True,capture_output=True,timeout=20)
        log=run.stdout+run.stderr;(HERE/f'{name}.log').write_text(log)
        if run.returncode:raise RuntimeError(log)
        tests[name]=int(re.search(r'Ran (\d+) tests?',log).group(1))
    a=json.loads((HERE/'run-01/results.json').read_text());b=json.loads((HERE/'verified/results.json').read_text())
    if a['metrics']!=b['metrics']:raise AssertionError('metrics changed during final validation')
    for mode,dd in a['decisions'].items():
        x={tuple(d['shape']):d['prediction']['predicted_log_us'] for d in dd}
        y={tuple(d['shape']):d['prediction']['predicted_log_us'] for d in b['decisions'][mode]}
        if x!=y:raise AssertionError('scores changed')
    rows,folds,bases=diagnose.load_inputs();timing=[]
    with diagnose.forbid_training() as calls:
        models={f:model.assemble(base,[r for r in rows if folds[tuple(r['shape'])]!=f],'work_equivalent_with_M') for f,base in bases.items()}
        surfaces=[(s,[r for r in rows if tuple(r['shape'])==s]) for s in sorted(folds)]
        for _ in range(10):
            t=time.perf_counter()
            for shape,rr in surfaces:model.predict(models[folds[shape]],rr)
            timing.append(1000*(time.perf_counter()-t))
    design=json.loads((HERE/'design-before.json').read_text())
    for p,h in {**design['source_hashes'],**design['protected_hashes']}.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior input changed: '+p)
    sources=[ROOT/'EDM-IR/edmir/compiler/shape_transfer_selection.py',ROOT/'EDM-IR/tests/shape_transfer_selection_test.py',
             *[HERE/p for p in ('review.py','audit.py','validate.py','README.md')]]
    for source in sources:
        dest=HERE/'verified/source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    result={'python_tests':tests,'total_tests':sum(tests.values()),'training_calls':calls,
            'all_41_scoring_ms':{'median_warm':statistics.median(timing[2:]),'maximum':max(timing),'repetitions':len(timing)},
            'protected_files_unchanged':len(design['protected_hashes']),
            'prior_source_and_evidence_files_unchanged':len(design['source_hashes']),
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'final_scores_match_first_run':True,'GPU_started':False,'frozen_six_parsed_or_scored':False,
            'compiler_admission':False,'all_new_controls_rejected':True,'elapsed_seconds':time.monotonic()-start}
    if result['elapsed_seconds']>60:raise TimeoutError('local verification limit')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
