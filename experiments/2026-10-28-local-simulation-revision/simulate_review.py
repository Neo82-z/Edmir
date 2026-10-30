"""Fixed synthetic mechanism scenarios; no fitting to observed pair gaps."""
from dataclasses import replace
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler.dependency_wave_model import KernelCase
from edmir.compiler.resource_service_simulator import ResourceScenario,simulate

def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design={'bandwidth_bytes_per_unit':[4096,16384,65536],
            'later_epilogue_units':[.25,1.,4.],'A_delivery_fraction':[.25,1.],
            'remote_source':[False,True],'admission_orders':['partition_major','tile_major'],
            'compute_fixed_units':1.,'compute_per_iteration_units':1.,'first_epilogue_units':.5,
            'B_delivery_fraction':1.,'slots':108,'unit':'synthetic_units',
            'scope':'same 18 work/service scenarios x 2 locality x 2 admission orders; no parameter optimization or best-scenario selection',
            'desired_observed_signs_are_NOT_simulator_inputs':True,'max_seconds':60}
    (out/'design-before.json').write_text(json.dumps(design,indent=2)+'\n')
    protected=json.loads((HERE/'design-before.json').read_text())['protected_hashes']
    for p,h in protected.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise AssertionError('protected input changed')
    outputs=[]
    for bw,e,a,remote,order in product(design['bandwidth_bytes_per_unit'],design['later_epilogue_units'],design['A_delivery_fraction'],design['remote_source'],design['admission_orders']):
        s=ResourceScenario(1.,1.,.5,e,bw,bw,a,1.,'synthetic_units',
                           'fixed synthetic fair sharing; not calibrated A100 service')
        short=[]
        for n in (128,256,512,1024,2048):
            c=KernelCase('synthetic-not-A100-calibration',4,64,512,n,1,1)
            pair={q:simulate(replace(c,q=q),s,slots=108,remote_source=remote,order=order) for q in (1,2)}
            gap=pair[2]['scenario_makespan']-pair[1]['scenario_makespan']
            short.append({'N':n,'gap_q2_minus_direct':gap,'sign':1 if gap>1e-9 else -1 if gap< -1e-9 else 0,'pair':pair})
        c=KernelCase('synthetic-not-A100-calibration',4,64,8192,1024,4,1)
        grouped={f'q{q}g{g}':simulate(replace(c,q=q,group_n=g),s,slots=108,remote_source=remote,order=order) for q,g in product((4,8),(1,2))}
        outputs.append({'bandwidth':bw,'later_epilogue':e,'A_fraction':a,'remote':remote,'order':order,
                        'short':short,'grouped':grouped})
        if time.monotonic()-start>60:raise TimeoutError('scenario review deadline')
    # Observed endpoint labels are read only after every simulation is complete.
    rows,_,_=diagnose.load_inputs();observed=[]
    for n in (128,256,512,1024,2048):
        rr={r['candidate_id']:r['latency_us'] for r in rows if r['shape']==[4,64,512,n]}
        gap=rr['splitk2']-rr['direct'];observed.append(1 if gap>0 else -1 if gap<0 else 0)
    patterns={}
    for r in outputs:
        signs=','.join(str(s['sign']) for s in r['short']);patterns[signs]=patterns.get(signs,0)+1
    matches=sum([s['sign'] for s in r['short']]==observed for r in outputs)
    for p,h in protected.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise AssertionError('protected input changed')
    result={'scenarios':len(outputs),'simulations':len(outputs)*14,'patterns':patterns,'observed_development_signs':observed,
            'scenarios_reproducing_all_five_signs':matches,'scope_caveat':'single-GEMM scenario signs are compared as hypotheses, not predictions of measured whole-program wall-time',
            'absence_in_grid_is_global_impossibility_proof':False,'outputs':outputs,
            'new_training_labels':0,'GPU_started':False,'frozen_six_parsed_or_scored':False,'seconds':time.monotonic()-start}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
