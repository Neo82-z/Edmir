#!/usr/bin/env python3
"""Inspect an experimental traffic selector (not accepted) without loading measurements or fitting."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.traffic_latency_selection import TrafficLatencySelector
from edmir.compiler.execution_cost_selection import ceildiv
from edmir.compiler.host_baseline import host_baseline
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.two_stage_selection import parse_schedule_family


def static_candidates(m,k,n,environment):
    families=['host','direct',*(f'splitk{q}' for q in (2,4,8,16))]
    families.extend([f'{f}_groupedn2' for f in families if f!='host'])
    rows={}
    for name in families:
        q,g=parse_schedule_family(name)
        rho=1 if name=='host' else ceildiv(n,128*g)
        row=ScheduleCandidate(environment,name,4,m,k,n,24*m*k,rho,q,g,
                              0 if name=='host' else ceildiv(m,64)*rho*q)
        row.validate();rows[name]=row
    return rows


def main(args):
    model=TrafficLatencySelector.from_dict(json.loads(args.model.read_text()))
    rows=static_candidates(args.m,args.k,args.n,args.environment)
    result=model.select(rows,baseline=host_baseline(args.baseline),allow_extrapolation=args.allow_extrapolation)
    from explain_execution import explain
    from edmir.compiler.evidence_selection import assess_candidates
    review=explain((4,args.m,args.k,args.n),environment_id=args.environment,baseline_id=args.baseline)
    review['decision']=assess_candidates(review['ledgers'],model_predictions_us={
        name:value*1000 for name,value in result['predicted_ms'].items()})
    result['model_preference']=result.pop('choice')
    review['fitted_model_diagnostic']=result
    print(json.dumps(review,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--m',type=int,required=True);p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--environment',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--allow-extrapolation',action='store_true')
    main(p.parse_args())
