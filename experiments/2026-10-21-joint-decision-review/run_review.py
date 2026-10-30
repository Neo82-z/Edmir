"""Bounded replay, numerical checks, and preservation of earlier artifacts."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from candidate import Candidate, ROOT, VARIANTS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from run_selector_gpu import command,write_json

HERE=Path(__file__).resolve().parent


def main(out):
    import numpy,scipy
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['JOINT_REVIEW_RESULTS']=str(out/'replay/results.json')
    jobs=[('evaluate',[sys.executable,HERE/'evaluate.py','--out',out/'replay'],900),
          ('diagnose',[sys.executable,HERE/'diagnose.py','--out',out/'diagnosis.json'],60),
          ('audit_tests',[sys.executable,'-m','unittest','discover','-s',HERE,'-p','test_*.py','-v'],60),
          ('traffic_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','traffic_latency_selection_test.py','-v'],60),
          ('resource_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','resource_geometry_test.py','-v'],60)]
    stages=[];started=time.monotonic()
    for name,args,limit in jobs:
        write_json(out/'progress.json',{'state':'running','stage':name,'timeout_seconds':limit})
        before=time.monotonic()
        try:
            with (out/f'{name}.stdout.txt').open('w') as stdout,(out/f'{name}.stderr.txt').open('w') as stderr:
                command(args,timeout=limit,stdout=stdout,stderr=stderr)
        except BaseException as error:
            write_json(out/'progress.json',{'state':'failed','stage':name,'reason':str(error)});raise
        stages.append({'stage':name,'elapsed_seconds':time.monotonic()-before,'timeout_seconds':limit,'exit_code':0})
    result=json.loads((out/'replay/results.json').read_text())
    initial=json.loads((HERE/'initial/results.json').read_text())
    for old,new in zip(initial['variants'],result['variants']):
        old=copy.deepcopy(old);new=copy.deepcopy(new)
        old.pop('elapsed_seconds');new.pop('elapsed_seconds')
        assert old==new,'exact replay differs'
    assert json.loads((out/'diagnosis.json').read_text())==json.loads((HERE/'initial/diagnosis.json').read_text())
    tests={}
    for name,_,_ in jobs:
        if name.endswith('_tests'):
            log=(out/f'{name}.stderr.txt').read_text();match=re.search(r'Ran (\d+) tests? in',log)
            assert match and log.rstrip().endswith('OK'),name
            tests[name]=int(match.group(1))
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    static=[{n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items()} for s in data.validate().values()]
    performance=[]
    for name in VARIANTS:
        before=time.perf_counter();model=Candidate.fit(data,method=name,use_cache=False)
        fit_seconds=time.perf_counter()-before;elapsed=[]
        for _ in range(100):
            before=time.perf_counter()
            for surface in static:model.select(surface,baseline=data.baseline)
            elapsed.append(time.perf_counter()-before)
        performance.append({'method':name,'uncached_fit_seconds':fit_seconds,'decisions':4100,
                            'mean_ms_per_41_shapes':1000*sum(elapsed)/len(elapsed),
                            'max_ms_per_41_shapes':1000*max(elapsed)})
    write_json(out/'performance.json',{'scope':'Local CPU predictor time, not GPU execution latency','records':performance})
    sources={**result['source_sha256'],**json.loads((out/'diagnosis.json').read_text())['source_sha256']}
    for p in (HERE/'test_review.py',Path(__file__)):
        sources[str(p.relative_to(ROOT))]=sha256(p)
    for relative,value in sources.items():assert sha256(ROOT/relative)==value,relative
    accepted=[v['method'] for v in result['variants'] if v['assessment']['candidate_accepted']]
    decision={'schema':'edmir.joint-decision-review.v1','date':'2026-10-21',
              'local_accepted':bool(accepted),'accepted_variants':accepted,'compiler_admission':False,
              'gpu_operations':0,'new_independent_labels':0,'scope':'Retrospective engineering on inspected archive',
              'candidate_assessments':{v['method']:v['assessment'] for v in result['variants']},
              'exact_replay':True,'tests':tests,'total_tests':sum(tests.values()),
              'old_evidence_and_gpu_package_unchanged':True,'performance':performance,
              'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__},
              'source_sha256':sources,
              'evidence_sha256':{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*'))
                                 if p.is_file() and p.name!='progress.json'},
              'elapsed_seconds':time.monotonic()-started}
    write_json(out/'decision.json',decision)
    write_json(out/'progress.json',{'state':'complete','accepted':accepted,'stages':stages,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'local_accepted':bool(accepted),'accepted':accepted,'tests':sum(tests.values()),
                      'exact_replay':True,'performance':performance,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
