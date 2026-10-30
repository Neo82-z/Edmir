"""Bounded, independently repeated local replay and CPU inference benchmark."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.decision_loss_selection import DecisionLossSelector,VARIANTS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset,sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from run_selector_gpu import command,write_json


def main(out):
    import numpy,scipy
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['DECISION_REVIEW_RESULTS']=str(out/'replay/results.json')
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    jobs=[('evaluate',[sys.executable,HERE/'evaluate.py','--out',out/'replay'],1800),
          ('decision_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','decision_loss_selection_test.py','-v'],120),
          ('old_shared_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','shared_candidate_selection_test.py','-v'],120),
          ('integrity_tests',[sys.executable,'-m','unittest','discover','-s',HERE,'-p','test_review.py','-v'],60),
          ('additive_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','additive_latency_selection_test.py','-v'],120),
          ('traffic_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','traffic_latency_selection_test.py','-v'],120),
          ('resource_tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'EDM-IR/tests','-p','resource_geometry_test.py','-v'],60)]
    started=time.monotonic();stages=[]
    for name,args,limit in jobs:
        write_json(out/'progress.json',{'state':'running','stage':name,'timeout_seconds':limit})
        before=time.monotonic()
        try:
            with (out/(name+'.stdout.txt')).open('w') as stdout,(out/(name+'.stderr.txt')).open('w') as stderr:
                command(args,timeout=limit,stdout=stdout,stderr=stderr)
        except BaseException as e:
            write_json(out/'progress.json',{'state':'failed','stage':name,'reason':str(e)});raise
        stages.append({'stage':name,'elapsed_seconds':time.monotonic()-before,'exit_code':0,'timeout_seconds':limit})
    result=json.loads((out/'replay/results.json').read_text())
    initial=json.loads((HERE/'initial/results.json').read_text())
    for a,b in zip(initial['variants'],result['variants']):
        a=copy.deepcopy(a);b=copy.deepcopy(b);a.pop('elapsed_seconds');b.pop('elapsed_seconds')
        assert a==b,'repeated numerical replay differs'
    assert initial['source_sha256']==result['source_sha256']
    tests={}
    for name,_,_ in jobs:
        if name.endswith('_tests'):
            log=(out/(name+'.stderr.txt')).read_text();m=re.search(r'Ran (\d+) tests? in',log)
            assert m and log.rstrip().endswith('OK'),name
            tests[name]=int(m.group(1))
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    surfaces=[{n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in s.items()} for s in data.validate().values()]
    performance=[]
    (out/'models').mkdir()
    for name in VARIANTS:
        t=time.perf_counter();model=DecisionLossSelector.fit(data,method=name,use_cache=False)
        seconds=time.perf_counter()-t;elapsed=[]
        write_json(out/'models'/(name+'.json'),model.to_dict())
        for _ in range(100):
            t=time.perf_counter()
            for c in surfaces:model.select(c,baseline=data.baseline)
            elapsed.append(time.perf_counter()-t)
        performance.append({'method':name,'uncached_fit_seconds':seconds,'decisions':4100,
                            'mean_ms_per_41_shapes':1000*sum(elapsed)/len(elapsed),
                            'max_ms_per_41_shapes':1000*max(elapsed)})
    write_json(out/'performance.json',{'scope':'Local CPU selector time; NOT GPU execution latency','records':performance})
    sources=result['source_sha256']
    for p in (Path(__file__),HERE/'test_review.py',
              ROOT/'EDM-IR/tests/decision_loss_selection_test.py',ROOT/'EDM-IR/tools/predict_decision_schedule.py'):
        sources[str(p.relative_to(ROOT))]=sha256(p)
    for relative,value in sources.items():assert sha256(ROOT/relative)==value,relative
    accepted=[v['method'] for v in result['variants'] if v['assessment']['candidate_accepted']]
    decision={'schema':'edmir.decision-loss-review.v1','date':'2026-10-21','exact_replay':True,
        'local_accepted':bool(accepted),'accepted_variants':accepted,'compiler_admission':False,
        'gpu_operations':0,'new_independent_labels':0,'tests':tests,'total_tests':sum(tests.values()),
        'old_evidence_and_gpu_package_unchanged':True,'performance':performance,
        'candidate_assessments':{v['method']:v['assessment'] for v in result['variants']},
        'optimization':{v['method']:{k:x for k,x in v['optimization'].items() if k!='audits'} for v in result['variants']},
        'source_sha256':sources,'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__},
        'evidence_sha256':{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='progress.json'},
        'elapsed_seconds':time.monotonic()-started}
    write_json(out/'decision.json',decision)
    write_json(out/'progress.json',{'state':'complete','accepted':accepted,'stages':stages,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'accepted':accepted,'tests':sum(tests.values()),'exact_replay':True,
                      'performance':performance,'elapsed_seconds':time.monotonic()-started},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
