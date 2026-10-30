"""Second bounded round: same shared head, retained empirical prior and support.

The prior is fitted strictly inside each original fold. Two losses and two
support policies are fixed before this run; no fresh-shape labels are accessed.
"""
from pathlib import Path
import json
import sys
import time
from copy import deepcopy

from diagnose import load_rows,ROOT
from edmir.compiler import path_aware_selection as model


def main(out):
    out.mkdir(parents=True,exist_ok=False);rows,folds=load_rows()
    losses=[False,'frontier'];supports=['unrestricted','training_hull']
    design={'representation':'original 31 features, unchanged width8; no added host interactions or class reweighting',
            'losses':losses,'support_policies':supports,'epochs':1000,'seeds':[0,1],
            'prior':'existing train-fold NNLS cost model; empirical, not identified physical service',
            'settings':model.SETTINGS,'new_labels':0,'total_seconds_cap':120,
            'model_choice':'retrospective comparison only; final independent model seal remains pending'}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    results=[];start=time.monotonic()
    for loss in losses:
        all_decisions={s:[] for s in supports}
        for fold in range(4):
            if time.monotonic()-start>120:raise TimeoutError('anchored run deadline exceeded')
            prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
            train=[r for r in rows if folds[tuple(r['shape'])]!=fold]
            test=[{**r,'split':'evaluation'} for r in rows if folds[tuple(r['shape'])]==fold]
            fitted=model.fit(train,representation='original',cross_path=loss,balanced=False,width=8,prior=prior)
            tr=model.evaluate(fitted,train,training_diagnostic=True)
            evaluations={}
            for support in supports:
                version={**fitted,'residual_support':support}
                ev=model.evaluate(version,test);all_decisions[support]+=ev['decisions'];evaluations[support]=ev
            (out/f'loss-{loss}-fold-{fold}.json').write_text(json.dumps({'model':fitted,'train':tr,'evaluations':evaluations},indent=2)+'\n')
        for support,decisions in all_decisions.items():
            result={'loss':loss,'support':support,'mean_regret':sum(r['regret'] for r in decisions)/len(decisions),
                    'max_regret':max(r['regret'] for r in decisions),'host_errors':sum(r['host_error'] for r in decisions),
                    'fused_errors':sum(r['fused_error'] for r in decisions),
                    'host_winning_choices':[{'shape':r['shape'],'chosen':r['chosen'],'regret':r['regret']} for r in decisions if r['host_wins']],
                    'all_order_invariant':all(r['order_invariant'] for r in decisions),'decisions':decisions}
            results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='decisions'}),flush=True)
        (out/'results.json').write_text(json.dumps({'design':design,'variants':results,'seconds':time.monotonic()-start,'compiler_admission':False},indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]))
