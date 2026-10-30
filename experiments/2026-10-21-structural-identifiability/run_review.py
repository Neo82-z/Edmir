"""Bounded, CPU-only replay. Successful execution does not mean acceptance."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys
import time
from candidate import Candidate, ROOT
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from run_selector_gpu import command, write_json

HERE=Path(__file__).resolve().parent


def main(out):
    import numpy, scipy
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs=[('evaluate',[sys.executable,HERE/'evaluate.py','--out',out/'results.json'],180),
          ('diagnose',[sys.executable,HERE/'diagnose.py','--out',out/'diagnosis.json'],60),
          ('measurement_design',[sys.executable,HERE/'measurement_design.py','--out',out/'next_measurements.json'],10),
          ('evidence',[sys.executable,ROOT/'experiments/2026-10-20-selector-decision-audit/verify_evidence.py',out/'evidence.json'],60)]
    for name,folder,pattern in (
        ('selection_tests',ROOT/'EDM-IR/tests','*selection_test.py'),
        ('splitk_tests',ROOT/'EDM-IR/tests','splitk_revision_test.py'),
        ('reliability_tests',ROOT/'EDM-IR/tests','predictor_reliability_test.py'),
        ('runner_tests',ROOT/'EDM-IR/tests','selector_runner_safety_test.py'),
        ('previous_review_tests',ROOT/'experiments/2026-10-21-execution-cost-revision','test_*.py'),
        ('previous_boundary_tests',ROOT/'experiments/2026-10-21-host-boundary-repair','test_*.py'),
        ('resource_tests',ROOT/'EDM-IR/tests','resource_geometry_test.py'),
        ('audit_tests',HERE,'test_*.py')):
        jobs.append((name,[sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern,'-v'],60))
    stages=[];started=time.monotonic()
    for name,args,limit in jobs:
        write_json(out/'progress.json',{'state':'running','stage':name,'timeout_seconds':limit})
        start=time.monotonic()
        try:
            with (out/f'{name}.stdout.txt').open('w') as stdout,(out/f'{name}.stderr.txt').open('w') as stderr:
                command(args,timeout=limit,stdout=stdout,stderr=stderr)
        except BaseException as error:
            write_json(out/'progress.json',{'state':'failed','stage':name,'reason':str(error)});raise
        stages.append({'stage':name,'elapsed_seconds':time.monotonic()-start,'timeout_seconds':limit,'exit_code':0})
    result=json.loads((out/'results.json').read_text());diagnosis=json.loads((out/'diagnosis.json').read_text())
    initial=json.loads((HERE/'results.json').read_text());old_diagnosis=json.loads((HERE/'diagnosis_margin.json').read_text())
    assert result['variants']==initial['variants'],'numerical selection replay differs'
    for key in ('old_alias','axes_alias','records'):assert diagnosis[key]==old_diagnosis[key],key
    tests={}
    for name,_,_ in jobs:
        if name.endswith('_tests'):
            log=(out/f'{name}.stderr.txt').read_text();match=re.search(r'Ran (\d+) tests? in',log)
            assert match and log.rstrip().endswith('OK'),name
            tests[name]=int(match.group(1))
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces=[{n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items()} for s in data.validate().values()]
    perf=[]
    for average in (False,True):
        model=Candidate.fit(data,average=average);elapsed=[]
        for _ in range(100):
            start=time.perf_counter()
            for s in surfaces:model.select(s,baseline=data.baseline)
            elapsed.append(time.perf_counter()-start)
        perf.append({'method':'separate_resources_average' if average else 'separate_resources','decisions':4100,
                     'mean_ms_per_41_shapes':1000*sum(elapsed)/len(elapsed),'max_ms_per_41_shapes':1000*max(elapsed)})
    write_json(out/'performance.json',{'scope':'Local Python predictor timing, not GPU operator time','results':perf})
    sources={**result['source_sha256'],**diagnosis['source_sha256']}
    for p in [Path(__file__),HERE/'test_audit.py',HERE/'measurement_design.py',ROOT/'EDM-IR/tests/resource_geometry_test.py']:
        sources[str(p.relative_to(ROOT))]=sha256(p)
    for path,digest in sources.items():assert sha256(ROOT/path)==digest,path
    accepted=[v['method'] for v in result['variants'] if v['assessment']['candidate_accepted']]
    if accepted:raise AssertionError('unexpected acceptance requires explicit investigation, not silent promotion')
    decision={'schema':'edmir.structural-identifiability-review.v1','local_accepted':False,
              'completed':['resource count separation','old feature collision repair in experimental candidates','exact axes rank/null-space analysis',
                           'both fixed candidate replays','all previous stability gates','training-only feasible-model witnesses','independent numerical solver checks'],
              'unresolved':['M512 and large-N choices do not jointly pass stability gates','physical causes of cost changes are not identified by end-to-end totals'],
              'next_evidence':'Small matched P4 measurements are required for an evidence-based next revision; this is not a proof that all mathematical model classes have been exhausted.',
              'candidate_assessments':{v['method']:v['assessment'] for v in result['variants']},
              'prior_accepted_selector_unchanged':True,'compiler_admission':False,'gpu_operations':0,'new_independent_labels':0,
              'numerical_replay_exact':True,'total_tests':sum(tests.values()),'tests':tests,
              'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__},
              'source_sha256':sources,'evidence_sha256':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='progress.json'}}
    write_json(out/'decision.json',decision)
    write_json(out/'progress.json',{'state':'completed_model_unresolved','stages':stages,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'local_accepted':False,'tests':sum(tests.values()),'exact_replay':True,'performance':perf,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
