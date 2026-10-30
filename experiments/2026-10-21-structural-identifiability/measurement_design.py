"""Generate an offline-only discriminating measurement plan, not a GPU job."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from candidate import ROOT
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256


def build():
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    archived={r.shape for r in data.rows}
    anchors=[(4,512,4096,512),(4,64,4096,2048)]
    crossed=[(4,m,3072,n) for m in (64,192) for n in (512,1536)]
    fresh=crossed+[(4,512,3072,512)]
    assert not archived&set(fresh)
    # Pairing two points on the old axes alone does NOT identify an M/N
    # interaction. A complete 2x2 contrast supplies the missing crossed point.
    matrix=lambda ss:np.array([[k/4096,k/4096*m/64,k/4096*n/512,k/4096*m/64*n/512] for p,m,k,n in ss])
    rank_before=int(np.linalg.matrix_rank(matrix(sorted(archived))))
    rank_after=int(np.linalg.matrix_rank(matrix(sorted(archived)+crossed)))
    assert (rank_before,rank_after)==(3,4)
    families=['host','direct','splitk2','splitk4','splitk8']
    configurations=[{'shape':s,'family':f,'purpose':'environment_anchor' if s in anchors else 'model_identification',
                     'within_current_shape_domain':s[1]==64 or s[3]==512} for s in anchors+fresh for f in families]
    order=[]
    for block in range(3):
        items=list(configurations);random.Random(20261021+block).shuffle(items)
        offset=len(order)
        order.extend({'block':block,'ordinal':offset+i,**v} for i,v in enumerate(items))
    # No runner, command, network connection or GPU allocation is performed.
    return {'schema':'edmir.discriminating-measurement-plan.v1','execution_ready':False,
            'purpose':'New observations for model identification. If used to revise the model, they become development data, not independent final validation.',
            'hardware':'Four A100 GPUs with a recorded peer topology for the P4 full-operator path; no hardware rented or selected here.',
            'baseline_id':data.baseline.baseline_id,'environment_id':None,
            'environment_rule':'Assign a new environment identity and record source/build/library/topology/timing metadata. Never relabel it as the old archive.',
            'frozen_before_observation':True,
            'new_shapes':fresh,'anchor_shapes':anchors,'families':families,
            'crossed_design_rank_before':rank_before,'crossed_design_rank_after':rank_after,
            'contrasts':[
                {'name':'same output grid, different A/B working sets','shapes':[(4,64,3072,1536),(4,192,3072,512)]},
                {'name':'M/N interaction at fixed K','formula':'t(192,1536)-t(192,512)-t(64,1536)+t(64,512), at K=3072, separately for each family','shapes':crossed},
                {'name':'new K near the M512 host/split boundary','shapes':[(4,512,3072,512),(4,512,4096,512)]}],
            'timing':{'warmup_iterations':20,'timed_iterations':100,'fresh_process_blocks':3,
                      'primary':'unprofiled host steady_clock around execute plus synchronize, consistent across families',
                      'diagnostic_pass':'Separately record CPU submission and device stream timelines, and available memory/communication counters. Profiling results never replace the primary timing labels; overlapping phases must not be added.',
                      'process_timeout_seconds':120,'stop_on':'correctness failure, baseline mismatch, timeout, incomplete topology or metadata'},
            'unresolved_before_execution':['server and budget','frozen build/package for this plan','CPU reference outputs and execution preflight'],
            'configurations':len(configurations),'process_runs':len(order),'run_order':order,
            'not_complete_candidate_oracle':True,
            'scope_limit':'The crossed M192/N1536 point is diagnostic only; this plan does not extend compiler admission or grouped-N support. q16/grouped candidates are not covered.',
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__)),
                             'experiments/2026-10-21-structural-identifiability/candidate.py':sha256(Path(__file__).with_name('candidate.py'))}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('fresh output required')
    args.out.write_text(json.dumps(build(),indent=2,allow_nan=False)+'\n')
