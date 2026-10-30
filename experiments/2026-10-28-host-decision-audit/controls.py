"""Reproduce the epoch-only and historical data-support controls."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import time
from diagnose import load_rows,ROOT,PREVIOUS,old,choices,tensorize


def epochs(out):
    out.mkdir(parents=True,exist_ok=False);rows,folds=load_rows();saved=dict(old.SETTINGS)
    design={'changed_setting':'epochs 100 -> 1000','unchanged_settings':saved,'folds':[0,1,2,3,'full'],'new_labels':0}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    start=time.monotonic()
    try:
        old.SETTINGS['epochs']=1000
        for fold in design['folds']:
            if time.monotonic()-start>60:raise TimeoutError('epoch control deadline')
            train=[r for r in rows if fold=='full' or folds[tuple(r['shape'])]!=fold]
            test=[r for r in rows if fold!='full' and folds[tuple(r['shape'])]==fold]
            model=old.fit(train)
            result={'fold':fold,'model':model,'training':choices(model,train),'evaluation':choices(model,test) if test else None}
            (out/f'fold-{fold}.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:old.SETTINGS.clear();old.SETTINGS.update(saved)


def support(out):
    import numpy as np
    from scipy.optimize import linprog
    from edmir.compiler.measurement_dataset import load_identified_selector_dataset
    from edmir.compiler.additive_latency_selection import AdditiveLatencySelector,host_features,fused_features
    from edmir.compiler.shared_candidate_selection import canonical_choice
    out.mkdir(parents=True,exist_ok=False);rows,folds=load_rows()
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    design={'cases':'same four folds, raw additive scores on exactly the original measured candidate sets',
            'extra_diagnostics':'convex-hull support and leave-one-target-shape-out; not a replacement of original four-fold evaluation','new_labels':0}
    (out/'design-before-fit.json').write_text(json.dumps(design,indent=2)+'\n')
    results=[];start=time.monotonic()
    for fold in range(4):
        if time.monotonic()-start>90:raise TimeoutError('support control deadline')
        selected=tuple(r for r in data.rows if folds[(r.world_size,r.rows_per_peer,r.k,r.n)]!=fold)
        model=AdditiveLatencySelector.fit(replace(data,rows=selected),timeout_seconds=15);decisions=[]
        for key,surface in data.validate().items():
            shape=tuple(key[-4:])
            if folds[shape]!=fold:continue
            actual={n:r.measured_ms*1000 for n,r in surface.items()}
            pred={n:1000*(model.host.predict(host_features(r)) if r.is_host else model.fused.predict(fused_features(r,model.capacity))) for n,r in surface.items()}
            winner=canonical_choice(pred);host=actual['host']<min(v for n,v in actual.items() if n!='host')
            decisions.append({'shape':shape,'chosen':winner,'regret':actual[winner]/min(actual.values())-1,
                'host_wins':host,'host_error':host and winner!='host','fused_error':not host and winner=='host',
                'predicted_us':pred,'measured_us':actual})
        results+=decisions
        (out/f'prior-fold-{fold}.json').write_text(json.dumps({'model':model.to_dict(),'decisions':decisions},indent=2)+'\n')
    (out/'prior-summary.json').write_text(json.dumps({'decisions':results,'mean_regret':sum(r['regret'] for r in results)/len(results),
        'max_regret':max(r['regret'] for r in results)},indent=2)+'\n')
    records=[];saved=dict(old.SETTINGS)
    try:
        old.SETTINGS['epochs']=1000
        for target in ((4,64,4096,2048),(4,64,8192,2048),(4,512,8192,512)):
            fold=folds[target];train_shapes=sorted({tuple(r['shape']) for r in rows if folds[tuple(r['shape'])]!=fold})
            coords=np.log2(np.array([s[1:] for s in train_shapes],float));point=np.log2(np.array(target[1:],float))
            hull=linprog(np.zeros(len(coords)),A_eq=np.vstack((coords.T,np.ones(len(coords)))),b_eq=np.r_[point,1.],bounds=(0,None),method='highs')
            if hull.status not in (0,2):raise RuntimeError('hull diagnosis incomplete')
            train=[r for r in rows if tuple(r['shape'])!=target];test=[r for r in rows if tuple(r['shape'])==target]
            model=old.fit(train)
            records.append({'target':target,'original_fold':fold,'within_training_logMKN_convex_hull':bool(hull.success),
                            'hull_status':int(hull.status),'leave_one_shape_out':choices(model,test),'model':model})
    finally:old.SETTINGS.clear();old.SETTINGS.update(saved)
    (out/'support.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in ('epochs','support'):raise SystemExit('usage: controls.py epochs|support NEW_OUTPUT_DIR')
    {'epochs':epochs,'support':support}[sys.argv[1]](Path(sys.argv[2]))
