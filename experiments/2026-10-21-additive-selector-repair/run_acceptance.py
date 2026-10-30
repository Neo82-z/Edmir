#!/usr/bin/env python3
"""Bounded local acceptance; writes a fresh evidence directory, never uses GPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import platform

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from run_selector_gpu import command, write_json
from edmir.compiler.measurement_dataset import sha256, load_identified_selector_dataset
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector
from edmir.compiler.relative_selection import ScheduleCandidate
from predict_additive_schedule import static_candidates
from evaluate_additive_selector import METHOD, PRIORITY


def main(out):
    import numpy, scipy
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(ROOT)
    started=time.monotonic();stages=[]
    jobs=[('evaluate',[sys.executable,ROOT/'EDM-IR/tools/evaluate_additive_selector.py','--out',out/'results.json'],60),
          ('review',[sys.executable,HERE/'review.py','--results',out/'results.json','--out',out/'review_results.json'],120),
          ('evidence',[sys.executable,ROOT/'experiments/2026-10-20-selector-decision-audit/verify_evidence.py',out/'evidence.json'],60)]
    for name,path,pattern in (
        ('selection_tests',ROOT/'EDM-IR/tests','*selection_test.py'),
        ('splitk_tests',ROOT/'EDM-IR/tests','splitk_revision_test.py'),
        ('reliability_tests',ROOT/'EDM-IR/tests','predictor_reliability_test.py'),
        ('runner_tests',ROOT/'EDM-IR/tests','selector_runner_safety_test.py'),
        ('previous_review_tests',ROOT/'experiments/2026-10-21-execution-cost-revision','test_*.py'),
        ('review_tests',HERE,'test_*.py')):
        jobs.append((name,[sys.executable,'-m','unittest','discover','-s',path,'-p',pattern,'-v'],60))
    for name,args,limit in jobs:
        write_json(out/'progress.json',{'state':'running','stage':name,'timeout_seconds':limit})
        start=time.monotonic()
        try:
            with (out/f'{name}.stdout.txt').open('w') as stdout,(out/f'{name}.stderr.txt').open('w') as stderr:
                command(args,timeout=limit,stdout=stdout,stderr=stderr)
        except BaseException as error:
            write_json(out/'progress.json',{'state':'failed','stage':name,'reason':str(error)});raise
        stages.append({'stage':name,'exit_code':0,'timeout_seconds':limit,'elapsed_seconds':time.monotonic()-start})
    result=json.loads((out/'results.json').read_text());review=json.loads((out/'review_results.json').read_text())
    assert result['gate']['passed'] and review['results_sha256']==sha256(out/'results.json')
    for obj in (result,review):
        for path,digest in obj['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    initial=json.loads((HERE/'results.json').read_text());initial_review=json.loads((HERE/'review_results.json').read_text())
    for key in ('primary','stress','gate','fitted_model'):assert result[key]==initial[key],key
    for key in ('sensitivity','deletion'):assert review[key]==initial_review[key],key
    model=AdditiveLatencySelector.from_dict(result['fitted_model'])
    write_json(out/'selector.json',model.to_dict())
    predictions=[]
    for _,m,k,n in sorted(PRIORITY):
        rows=static_candidates(m,k,n,model.environment_id)
        from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
        predictions.append({'shape':[4,m,k,n],**model.select(rows,baseline=PER_SOURCE_CUTLASS)})
    write_json(out/'priority_predictions.json',{'note':'Full-data fit for future comparison; these are not held-out accuracy results. No GPU execution.',
                                               'model_sha256':sha256(out/'selector.json'),'predictions':predictions})
    # Distinct CLI invocation under -S proves measurement-free/site-package-free use.
    with (out/'prediction_cli.stdout.txt').open('w') as stdout,(out/'prediction_cli.stderr.txt').open('w') as stderr:
        command([sys.executable,'-S',ROOT/'EDM-IR/tools/predict_additive_schedule.py','--model',out/'selector.json',
                 '--m','64','--k','8192','--n','512','--environment',model.environment_id,'--baseline',model.baseline_id],
                timeout=10,stdout=stdout,stderr=stderr)
    assert json.loads((out/'prediction_cli.stdout.txt').read_text())['choice']=='splitk8'
    tests={}
    for name,_,_ in jobs:
        if name.endswith('_tests'):
            log=(out/f'{name}.stderr.txt').read_text();match=re.search(r'Ran (\d+) tests? in',log)
            assert match and log.rstrip().endswith('OK'),name
            tests[name]=int(match.group(1))
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces=[{n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items()} for s in data.validate().values()]
    elapsed=[]
    for _ in range(100):
        start=time.perf_counter()
        for s in surfaces:model.select(s,baseline=data.baseline)
        elapsed.append(time.perf_counter()-start)
    performance={'full_41_shape_sweeps':100,'decisions':4100,'mean_ms_per_41_shapes':1000*sum(elapsed)/len(elapsed),
                 'max_ms_per_41_shapes':1000*max(elapsed),'measured_environment':platform.platform(),
                 'interpretation':'Local Python selector latency, not GPU/operator latency.'}
    write_json(out/'performance.json',performance)
    sources={**result['source_sha256'],**review['source_sha256']}
    for p in (Path(__file__),HERE/'test_review.py',HERE/'design.json',HERE/'requirements-fit.txt',
              ROOT/'EDM-IR/tests/additive_latency_selection_test.py',ROOT/'EDM-IR/tools/predict_additive_schedule.py',ROOT/'EDM-IR/pyproject.toml'):
        sources[str(p.relative_to(ROOT))]=sha256(p)
    decision={'schema':'edmir.additive-selector-acceptance.v1','date':'2026-10-21','local_accepted':True,
              'gate':result['gate'],'numerical_replay_exact':True,'total_tests':sum(tests.values()),'tests':tests,
              'scope':'retrospective local engineering acceptance; archived A100 P4 per-source CUTLASS only',
              'new_independent_labels':0,'gpu_operations':0,'compiler_admission':False,
              'remaining':'New matched-environment measurements; largest-N decision remains sensitive; M512 extrapolation errors remain.',
              'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__},
              'source_sha256':sources,
              'evidence_sha256':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='progress.json'}}
    write_json(out/'decision.json',decision)
    write_json(out/'progress.json',{'state':'completed','elapsed_seconds':time.monotonic()-started,'stages':stages,'gpu_operations':0})
    print(json.dumps({'local_accepted':True,'total_tests':sum(tests.values()),'numerical_replay_exact':True,
                      'performance':performance,'elapsed_seconds':time.monotonic()-started},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
