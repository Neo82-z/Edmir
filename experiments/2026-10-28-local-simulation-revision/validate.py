"""CPU-only acceptance record; deliberately no SSH or cloud operation."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import axis_residual_selection as axis
from edmir.compiler.dependency_wave_model import KernelCase
from edmir.compiler.resource_service_simulator import ResourceScenario,simulate


def main():
    began=time.monotonic();tests={}
    for name in ('axis_residual_selection_test','resource_service_simulator_test',
                 'dependency_wave_model_test','prediction_explanation_test'):
        p=subprocess.run([sys.executable,str(ROOT/'EDM-IR/tests'/f'{name}.py')],
                         cwd=ROOT,text=True,capture_output=True,timeout=20)
        log=p.stdout+p.stderr
        (HERE/f'{name}.log').write_text(log)
        if p.returncode:raise RuntimeError(log)
        tests[name]=int(re.search(r'Ran (\d+) tests?',log).group(1))
    rows,folds,bases=diagnose.load_inputs();models={}
    with diagnose.forbid_training() as calls:
        for fold,base in bases.items():
            models[fold]=axis.assemble(base,[r for r in rows if folds[tuple(r['shape'])]!=fold],'M_work_extension')
        surfaces=[(shape,[r for r in rows if tuple(r['shape'])==shape]) for shape in sorted(folds)]
        timings=[]
        for _ in range(12):
            start=time.perf_counter()
            for shape,surface in surfaces:axis.predict(models[folds[shape]],surface)
            timings.append((time.perf_counter()-start)*1000)
        c=KernelCase('synthetic-capacity-test',4,64,8192,4096,16,1)
        s=ResourceScenario(1,1,.5,1,4096,4096,1,1,'synthetic_units','maximum supported CTA count fixture')
        maxima=[simulate(c,s,slots=slots,remote_source=remote,order=order)
                for slots,remote,order in ((1,False,'partition_major'),(108,True,'partition_major'),(108,True,'tile_major'))]
    old=json.loads((HERE/'run-02/results.json').read_text())
    verified=json.loads((HERE/'verified/results.json').read_text())
    if old['metrics']!=verified['metrics']:raise AssertionError('final scores changed')
    for variant in axis.MODES:
        a={tuple(d['shape']):d for d in old['decisions'][variant]}
        b={tuple(d['shape']):d for d in verified['decisions'][variant]}
        for shape in a:
            if a[shape]['prediction']['predicted_log_us']!=b[shape]['prediction']['predicted_log_us']:
                raise AssertionError('score changed during explanation fix')
    design=json.loads((HERE/'design-before.json').read_text())
    for p,h in {**design['source_hashes'],**design['protected_hashes']}.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise AssertionError('prior evidence changed: '+p)
    source_paths=[ROOT/'EDM-IR/edmir/compiler'/f'{s}.py' for s in ('axis_residual_selection','resource_service_simulator')]
    source_paths += [ROOT/'EDM-IR/tests'/f'{s}.py' for s in tests if s in ('axis_residual_selection_test','resource_service_simulator_test')]
    source_paths += [HERE/f for f in ('review.py','simulate_review.py','validate.py','README.md')]
    result={'python_tests':tests,'test_count':sum(tests.values()),'training_calls':calls,
            'scoring_41_shapes_ms':{'median':statistics.median(timings[2:]),'max':max(timings),'repetitions':len(timings)},
            'maximum_simulation_size':512,'maximum_size_scenarios':maxima,
            'new_sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
            'frozen_and_protected_files_unchanged':len(design['protected_hashes']),
            'prior_source_and_evidence_files_unchanged':len(design['source_hashes']),
            'frozen_six_parsed_or_scored':False,'GPU_started':False,'new_training_labels':0,
            'local_near_perfect_acceptance_passed':False,'compiler_admission':False,
            'elapsed_seconds':time.monotonic()-began}
    if result['elapsed_seconds']>60:raise TimeoutError('validation deadline')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('maximum_size_scenarios','new_sources_sha256')},indent=2))


if __name__=='__main__':main()
