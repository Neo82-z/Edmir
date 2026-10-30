"""Predeclared historical support stress; no model tuning on these outcomes."""
from pathlib import Path
from dataclasses import replace
import json
import sys
import time
from diagnose import load_rows,ROOT,old,choices
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector
from edmir.compiler import path_aware_selection as revised
from edmir.compiler.shared_candidate_selection import canonical_choice


def main(out):
    out.mkdir(parents=True,exist_ok=False);rows,_=load_rows()
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    host_winners={tuple(s[0]['shape']) for s in old.validate_rows(rows,split='train') if
                  min(r['latency_us'] for r in s if r['is_host'])<min(r['latency_us'] for r in s if not r['is_host'])}
    cases={'all_M512':lambda r:r['shape'][1]==512,'all_N2048':lambda r:r['shape'][3]==2048,
           'all_host_winners':lambda r:tuple(r['shape']) in host_winners}
    (out/'design-before-fit.json').write_text(json.dumps({'cases':list(cases),'variants':['1000epoch_original','cost_prior','hull_limited_residual'],
        'settings':revised.SETTINGS,'purpose':'stronger historical development stress, not independent validation',
        'total_seconds_cap':90,'new_labels':0},indent=2)+'\n')
    saved=dict(old.SETTINGS);start=time.monotonic();results={}
    try:
        old.SETTINGS['epochs']=1000
        for name,test_fn in cases.items():
            if time.monotonic()-start>90:raise TimeoutError('stress deadline')
            train=[r for r in rows if not test_fn(r)];test=[r for r in rows if test_fn(r)]
            shapes={tuple(r['shape']) for r in train}
            prior=AdditiveLatencySelector.fit(replace(data,rows=tuple(r for r in data.rows if
                (r.world_size,r.rows_per_peer,r.k,r.n) in shapes)),timeout_seconds=15).to_dict()
            baseline=old.fit(train);base_result=choices(baseline,test)
            model=revised.fit(train,prior=prior,residual_support='training_hull')
            result=revised.evaluate(model,[{**r,'split':'evaluation'} for r in test])
            pure=[]
            for s in old.validate_rows(test,split='train'):
                score=dict(zip((r['candidate_id'] for r in s),revised.prior_scores(prior,s)))
                chosen=canonical_choice(score);actual={r['candidate_id']:r['latency_us'] for r in s}
                pure.append({'shape':s[0]['shape'],'chosen':chosen,'regret':actual[chosen]/min(actual.values())-1})
            results[name]={'original':base_result,'prior':{'decisions':pure,'mean_regret':sum(r['regret'] for r in pure)/len(pure),'max_regret':max(r['regret'] for r in pure)},
                           'revised':result,'model':model,'training_shapes':len(shapes),'training_host_winners':model['training_host_winners']}
            (out/f'{name}.json').write_text(json.dumps(results[name],indent=2)+'\n')
            print(json.dumps({'case':name,'train_host_winners':model['training_host_winners'],
                **{v:{'mean':results[name][v]['mean_regret'],'max':results[name][v]['max_regret']} for v in ('original','prior','revised')}}),flush=True)
    finally:old.SETTINGS.clear();old.SETTINGS.update(saved)


if __name__=='__main__':main(Path(sys.argv[1]))
