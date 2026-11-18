"""Save before-observation decisions from existing fits; never refit."""
import importlib.util
import json
from pathlib import Path
import sys
from prepare_selector_gpu import REPO
from edmir.compiler.additive_latency_selection import AdditiveLatencySelector
from edmir.compiler.traffic_latency_selection import TrafficLatencySelector, LatencyFit
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from predict_traffic_schedule import static_candidates


def freeze(root,shapes,families):
    evidence=REPO/'experiments'
    previous=[('additive',evidence/'2026-10-21-additive-selector-repair/verified',AdditiveLatencySelector),
              ('traffic',evidence/'2026-10-21-host-boundary-repair/verified',TrafficLatencySelector)]
    context=root/'model_context';context.mkdir()
    records=[];references={}
    def predict(name,model):
        forecasts=[]
        for s in shapes:
            if s[1]!=64 and s[3]!=512:
                forecasts.append({'shape':list(s),'status':'outside_model_domain','choice':None});continue
            rows={n:r for n,r in static_candidates(*s[1:], 'a100-sxm4-80gb-p4-2026-08-29').items() if n in families}
            result=model.select(rows,baseline=PER_SOURCE_CUTLASS)
            forecasts.append({'shape':list(s),'status':'conditional_forecast_in_old_environment',
                              'choice':result['choice'],'predicted_ms':result['predicted_ms']})
        return {'model':name,'compiler_admission':False,'forecasts':forecasts}
    for name,directory,cls in previous:
        state=json.loads((directory/'selector.json').read_text())
        model=cls.from_dict(state)
        records.append(predict(name,model))
        for file in ('selector.json','decision.json'):
            source=directory/file
            (context/(name+'_'+file)).write_bytes(source.read_bytes())
            references[str(source.relative_to(REPO))]=sha256(source)
    directory=evidence/'2026-10-21-structural-identifiability'
    source=directory/'candidate.py';name='boundary_frozen_candidate'
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    results=json.loads((directory/'verified/results.json').read_text())
    for path,value in results['source_sha256'].items():
        if sha256(REPO/path)!=value:raise ValueError('candidate source no longer matches reviewed fit')
    fit=lambda x:LatencyFit(tuple(x['names']),tuple(x['coefficients']),tuple(x['scales']),x['kkt_max_violation'])
    for variant in results['variants']:
        state=variant['full_fit']
        model=module.Candidate(tuple((tuple(p),r,fit(f)) for p,r,f in state['host_fits']),
                               tuple((p,r,fit(f)) for p,r,f in state['fused_fits']),
                               tuple(tuple(s) for s in state['training_shapes']),(),
                               tuple(state['host_cv']),tuple(state['fused_cv']),state['average'])
        records.append(predict(variant['method'],model))
        (context/(variant['method']+'.json')).write_text(json.dumps({'fit':state,'assessment':variant['assessment']},indent=2)+'\n')
    references[str((directory/'verified/results.json').relative_to(REPO))]=sha256(directory/'verified/results.json')
    payload={'purpose':'Frozen conditional forecasts from old-environment fits, not measured results or an admitted model',
             'old_environment_id':'a100-sxm4-80gb-p4-2026-08-29',
             'new_environment_prediction_supported':False,
             'note':'New hardware/software identity is recorded separately. Anchor drift must be reviewed before interpreting prediction errors. No inference-domain bypass for the crossed shape.',
             'models':records,'references_sha256':references,'fit_operations':0}
    (root/'frozen_predictions.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return [source]
