#!/usr/bin/env python3
"""Audit the remaining mistakes, record one revision, and freeze both models."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clean_mechanism_data import clean_collection,sha
from evaluate_mechanism_prediction import DEFAULT_DATA,REPO,comparisons,metrics,prediction,write
from edmir.compiler.mechanism_prediction import HOSTS
from edmir.compiler.regime_prediction import RegimeModel
from edmir.compiler.residual_regime_prediction import ResidualRegimeModel

DESIGN=REPO/'experiments/2026-10-25-final-local-revision/design.json'
ACCEPTANCE={'macro_shape_mean_absolute_percentage_error_max':.1,'max_candidate_absolute_percentage_error_max':.25,
            'wrong_pair_decisions_with_over_10_percent_loss_max':0}


def fit(rows,variant,flagged=False):
    cls=RegimeModel if variant=='regime_max' else ResidualRegimeModel
    return cls.fit(rows,variant=variant,include_flagged=flagged)


def folds(primary,variant,axis='shape',include_flagged=False):
    rows=primary if axis=='shape' else [r for r in primary if r['family'] not in HOSTS]
    key=lambda r: tuple(r['shape']) if axis=='shape' else r['shape'][1]
    estimates,audits=[],[]
    for group in sorted({key(r) for r in rows}):
        train=[r for r in rows if key(r)!=group and (r['eligible_for_calibration'] or include_flagged)]
        test=[r for r in rows if key(r)==group]
        model=fit(train,variant,include_flagged)
        assert not set(model.training_config_ids)&{r['config_id'] for r in test}
        output=[prediction(model,r,'leave_'+axis+'_out') for r in test]
        estimates.extend(output)
        audits.append({'held_out':group,'model':model.to_dict(),'predictions':output,
                       'metrics':metrics(output,ACCEPTANCE)})
    return estimates,audits


def algorithm_audit(primary):
    rows=[]
    for row in primary:
        if row['family']!='host_merged_cublaslt' or not row['eligible_for_calibration']:continue
        train=[r for r in primary if r['eligible_for_calibration'] and r['family']==row['family'] and r['shape']!=row['shape']]
        configs={a for r in train for a in r['algorithm_ids']}
        rows.append({'shape':row['shape'],'target_recorded_algorithm':row['algorithm_ids'],
            'training_shape_count':len(train),'exact_configuration_seen_in_training':all(a in configs for a in row['algorithm_ids']),
            'observed_us':row['observed_us'],
            'scope':'Post-measurement support audit only. Target algorithm never provided to latency fit or used to discard its error.'})
    return {'rows':rows,'distinct_full_configurations':len({tuple(r['target_recorded_algorithm']) for r in rows}),
            'fitted_algorithm_latency_parameters':0,
            'conclusion':'All five complete algorithm configurations differ. Cannot identify algorithm-specific cost from these five one-shape configurations.'}


def quality_attribution(primary,variant):
    target=[4,64,3072,1536]
    stable=[r for r in primary if r['eligible_for_calibration'] and r['shape']!=target]
    flagged=[r for r in primary if r['quality_flags'] and r['shape']!=target]
    interventions=[('stable',[]),('host_flagged',[r for r in flagged if r['family'] in HOSTS]),
                   ('fused_flagged',[r for r in flagged if r['family'] not in HOSTS]),('all_flagged',flagged)]
    interventions += [(r['config_id'],[r]) for r in flagged]
    output=[]
    for label,added in interventions:
        model=fit(stable+added,variant,bool(added))
        rows=[prediction(model,r,'quality_intervention') for r in primary if r['shape']==target]
        _,pairs=comparisons(rows)
        output.append({'intervention':label,'added_configurations':[{'id':r['config_id'],'shape':r['shape'],'family':r['family'],'flags':r['quality_flags']} for r in added],
            'predictions':rows,'pairs':pairs,'model':model.to_dict()})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('fresh output directory required')
    start=time.monotonic()
    _,configs,manifest=clean_collection(DEFAULT_DATA)
    primary=[r for r in configs if r['primary_scope']];stable=[r for r in primary if r['eligible_for_calibration']]
    models,estimates,audits,results,stress,sensitivity={},{},{},{},{},{}
    for variant in ('regime_max','residual_depth'):
        models[variant]=fit(stable,variant).to_dict()
        estimates[variant],audits[variant]=folds(primary,variant)
        results[variant]=metrics(estimates[variant],ACCEPTANCE)
        _,stress[variant]=folds(primary,variant,axis='M')
        pred,fs=folds(primary,variant,include_flagged=True)
        sensitivity[variant]={'metrics':metrics(pred,ACCEPTANCE),'predictions':pred,'folds':fs,
                              'single_flag_attribution':quality_attribution(primary,variant)}
    pairs={v:comparisons(p)[1] for v,p in estimates.items()}
    for v,rows in pairs.items():
        for r in rows:
            r['all_process_signs_agree']=all(x>0 for x in r['paired_process_deltas_us']) or all(x<0 for x in r['paired_process_deltas_us'])
    a.out.mkdir(parents=True)
    decision={'primary_model':'regime_max','challenger':'residual_depth','challenger_promoted':False,
        'reason':'Mean error improves slightly, but wrong-pair count and all-candidate choice loss worsen; M256 ordering is not repaired.',
        'further_old_data_tuning':False,'new_GPU_data_required_for_next_revision':True,'compiler_admission':False}
    for name,val in {'design':json.loads(DESIGN.read_text()),'configurations':configs,'dataset-manifest':manifest,
        'models':models,'heldout-predictions':estimates,'fold-models':audits,'metrics':results,'leave-M-out':stress,
        'pair-audit':pairs,'algorithm-support':algorithm_audit(primary),'quality-sensitivity':sensitivity,'decision':decision}.items():
        write(a.out/(name+'.json'),val)
    files=[Path(__file__),DESIGN,*[REPO/'EDM-IR/edmir/compiler'/f for f in
        ('mechanism_prediction.py','regime_prediction.py','residual_regime_prediction.py')],
        *[Path(__file__).with_name(f) for f in ('evaluate_mechanism_prediction.py','clean_mechanism_data.py')]]
    write(a.out/'analysis-seal.json',{'sources':{str(f.relative_to(REPO)):sha(f) for f in files},
        'outputs':{f.name:sha(f) for f in a.out.iterdir()},'elapsed_seconds':time.monotonic()-start,
        'gpu_runs':0,'independent_validation':False})
    print(json.dumps({'metrics':results,'decision':decision,'elapsed_seconds':time.monotonic()-start},indent=2))


if __name__=='__main__':main()
