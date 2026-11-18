#!/usr/bin/env python3
"""Static inference from a saved experimental decision-loss scorer; never fits a model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler.decision_loss_selection import DecisionLossSelector
from edmir.compiler.host_baseline import host_baseline
from predict_traffic_schedule import static_candidates
from explain_execution import explain


def main(args):
    model=DecisionLossSelector.from_dict(json.loads(args.model.read_text()))
    rows=static_candidates(args.m,args.k,args.n,args.environment)
    result=model.select(rows,baseline=host_baseline(args.baseline),allow_extrapolation=args.allow_extrapolation)
    review=explain((4,args.m,args.k,args.n),environment_id=args.environment,baseline_id=args.baseline)
    from edmir.compiler.evidence_selection import assess_candidates
    review['decision']=assess_candidates(review['ledgers'],model_predictions_us={
        name:value*1000 for name,value in result['predicted_ms'].items()})
    # Preserve research diagnostics without presenting a point estimate's
    # argmin as an evidence-supported executable decision.
    result['model_preference']=result.pop('choice')
    review['fitted_model_diagnostic']=result
    print(json.dumps(review,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    for name in ('m','k','n'):p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--environment',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--allow-extrapolation',action='store_true')
    main(p.parse_args())
