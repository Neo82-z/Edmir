"""Post-hoc explanation of selection errors; never selects a new model."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from candidate import (Candidate, ROOT, VARIANTS, HOST_GRID, FUSED_GRID, validate_data,
                       fit_component, host_features, fused_features, eligible)
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from run_selector_gpu import write_json

SHAPES=((4,512,4096,512),(4,64,4096,2048))


def diagnose(data):
    records=[]
    for shape in SHAPES:
        test=next(s for s in data.validate().values() if s['host'].shape==shape)
        train=replace(data,rows=tuple(r for r in data.rows if r.group_id!=test['host'].group_id))
        surfaces,_=validate_data(train)
        static={n:ScheduleCandidate.from_measurement(r,data.environment_id) for n,r in test.items()}
        for method in VARIANTS:
            model=Candidate.fit(train,method=method)
            chosen=model.select(static,baseline=data.baseline,allow_extrapolation=True)
            all_host={p:fit_component(surfaces,True,p,model.basis) for p in HOST_GRID}
            all_fused={p:fit_component(surfaces,False,p,model.basis) for p in FUSED_GRID}
            catalog=[]
            objective=model.objective
            minimum=min(r[objective] for r in model.cv_scores)
            for score in model.cv_scores:
                hp,fp=score['host_parameters'],score['fused_parameters']
                predictions={n:(all_host[hp].predict(host_features(r,*hp[:2])) if r.is_host
                                else all_fused[fp].predict(fused_features(r,fp[0],model.basis)))
                             for n,r in static.items() if eligible(r,model.grouped_support)}
                pick=min(predictions,key=lambda n:(predictions[n],n!='host',n))
                # Labels are used after every model/parameter prediction, for audit only.
                catalog.append({**score,'outer_choice_diagnostic_only':pick,
                    'outer_regret_diagnostic_only':test[pick].measured_ms/min(r.measured_ms for r in test.values())-1,
                    'primary_cv_tie':score[objective]==minimum})
            tied=[r for r in catalog if r['primary_cv_tie']]
            measured={n:r.measured_ms for n,r in test.items()}
            best=min(measured,key=lambda n:(measured[n],n!='host',n))
            focused=('host','splitk2','splitk4')
            components={}
            for n in focused:
                r=static[n];fit=model.host if r.is_host else model.fused
                x=host_features(r,*model.host_parameters[:2]) if r.is_host else fused_features(r,model.fused_parameters[0],model.basis)
                components[n]={'predicted_ms':chosen['predicted_ms'][n],'measured_ms':measured[n],
                               'error_ms':chosen['predicted_ms'][n]-measured[n],
                               'terms_ms':dict(zip(fit.names,(c*v for c,v in zip(fit.coefficients,x))))}
            records.append({'method':method,'shape':shape,'choice':chosen['choice'],'observed_best':best,
                'selected_host_parameters':model.host_parameters,'selected_fused_parameters':model.fused_parameters,
                'components':components,'primary_cv_minimum':minimum,'primary_cv_tied_pairs':len(tied),
                'tied_outer_choices':dict(Counter(r['outer_choice_diagnostic_only'] for r in tied)),
                'all_parameter_outer_choices':dict(Counter(r['outer_choice_diagnostic_only'] for r in catalog)),
                'parameter_catalog':catalog})
    return {'scope':'Post-hoc error decomposition on inspected development labels; does not choose or promote any parameter pair',
            'records':records,'gpu_operations':0,'new_independent_labels':0,
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__))}}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('fresh diagnosis path required')
    data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
    result=diagnose(data);write_json(args.out,result)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('parameter_catalog','components')} for r in result['records']],indent=2))
