"""Finite factorial interventions; preserve all variants, no fresh-set access."""
from pathlib import Path
import itertools
import json
import sys
import time

from diagnose import load_rows,ROOT
from edmir.compiler import path_aware_selection as model


def main(out):
    out.mkdir(parents=True,exist_ok=False);rows,folds=load_rows()
    configs=[{'representation':rep,'cross_path':cross,'balanced':balanced,'width':8}
             for rep,cross,balanced in itertools.product(('original','path_geometry'),(False,True),(False,True))]
    configs.append({'representation':'path_geometry','cross_path':True,'balanced':False,'width':0})
    design={'variants':configs,'settings':model.SETTINGS,'data':'same original P4 measurements and same four folds',
            'weighting':'balanced uses training-fold measured winner counts only; evaluation remains shape-uniform',
            'initialization_and_scaler':'same within a feature representation; not changed by loss or weighting',
            'total_seconds_cap':180,'new_GPU_data':0,'new_boundary_shapes_used':False}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    results=[];start=time.monotonic()
    for idx,config in enumerate(configs):
        decisions=[];train_errors=[];files=[]
        for fold in range(4):
            if time.monotonic()-start>180:raise TimeoutError('ablation run deadline')
            train=[r for r in rows if folds[tuple(r['shape'])]!=fold]
            test=[{**r,'split':'evaluation'} for r in rows if folds[tuple(r['shape'])]==fold]
            fitted=model.fit(train,**config);tr=model.evaluate(fitted,train,training_diagnostic=True)
            ev=model.evaluate(fitted,test);decisions+=ev['decisions'];train_errors.append(tr['host_errors'])
            name=f'variant-{idx}-fold-{fold}.json';files.append(name)
            (out/name).write_text(json.dumps({'model':fitted,'train':tr,'evaluation':ev},indent=2)+'\n')
        summary={'variant':idx,'config':config,'train_host_errors_per_fold':train_errors,
                 'mean_regret':sum(r['regret'] for r in decisions)/len(decisions),
                 'max_regret':max(r['regret'] for r in decisions),
                 'host_errors':sum(r['host_error'] for r in decisions),'fused_errors':sum(r['fused_error'] for r in decisions),
                 'host_winning_choices':[{'shape':r['shape'],'chosen':r['chosen'],'regret':r['regret']} for r in decisions if r['host_wins']],
                 'all_order_invariant':all(r['order_invariant'] for r in decisions),
                 'files':files}
        results.append(summary);(out/'results.json').write_text(json.dumps({'design':design,'variants':results,'seconds':time.monotonic()-start,'compiler_admission':False},indent=2)+'\n')
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
