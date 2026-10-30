#!/usr/bin/env python3
"""Inspect archival P4 predictions; no GPU operation or automatic admission."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.compiler import path_aware_selection as revised
from edmir.compiler.shape_decision_model import features
from edmir.compiler.tile_execution_model import describe_execution
from edmir.compiler.execution_ledger import FAMILIES,source_provenance
from edmir.compiler.shared_candidate_selection import canonical_choice


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--shape',type=int,nargs=4,required=True,metavar=('P','M','K','N'))
    p.add_argument('--policy',choices=('cost_prior','supported_residual'),default='cost_prior')
    p.add_argument('--candidates',nargs='+',choices=FAMILIES,default=list(FAMILIES[:6]))
    a=p.parse_args();obj=json.loads(a.model.read_text());model=obj.get('model',obj)
    if model['world_size']!=a.shape[0] or model.get('prior') is None:
        p.error('same-world model with an identified empirical cost prior required; no P4-to-P8 relabeling')
    if len(set(a.candidates))!=len(a.candidates) or 'host' not in a.candidates or len(a.candidates)<2:
        p.error('unique candidate set including host and fused required')
    provenance=source_provenance();rows=[]
    for name in a.candidates:
        s=describe_execution(a.shape,name,environment_id=model['execution_context'],metadata_mode='legacy',provenance=provenance)
        rows.append({'shape':a.shape,'execution_context':model['execution_context'],
                     'candidate_id':name,'is_host':name=='host','features':features(s)})
    prior=dict(zip((r['candidate_id'] for r in rows),revised.prior_scores(model['prior'],rows)))
    support=revised.in_training_hull(tuple(tuple(s) for s in model['training_shapes']),tuple(a.shape))
    scores=prior if a.policy=='cost_prior' else revised.predict({**model,'residual_support':'training_hull'},rows)
    print(json.dumps({'choice':canonical_choice(scores),'policy':a.policy,'shape':a.shape,
        'execution_context':model['execution_context'],'predicted_us':{n:math.exp(v) for n,v in scores.items()},
        'prior_predicted_us':{n:math.exp(v) for n,v in prior.items()},
        'residual_geometric_support':support,'residual_applied':a.policy=='supported_residual' and support,
        'shape_was_in_training':a.shape in model['training_shapes'],
        'predictions_are_measurements':False,'compiler_admission':False},indent=2))


if __name__=='__main__':main()
