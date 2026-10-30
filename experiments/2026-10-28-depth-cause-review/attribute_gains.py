"""Exact change decomposition for the previous 7 -> 5 experiment.

Parameter effects explain model predictions, not physical GPU causality.
"""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import split_stage_selection as stage


def main():
    data=json.loads((ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json').read_text())['p4']
    folds={tuple(s['shape']):s['fold'] for s in data['folds']}
    source=ROOT/'experiments/2026-10-28-split-stage-review/final-validation'
    variants=json.loads((source/'results.json').read_text())['variants']
    previous={tuple(d['shape']):d for d in variants['previous']['decisions']}
    current={tuple(d['shape']):d for d in variants['depth']['decisions']}
    rows={(tuple(r['shape']),r['candidate_id']):r for r in data['rows']}
    results=[]
    for shape,old in previous.items():
        new=current[shape]
        if old['provisional_choice']==new['provisional_choice']:continue
        fold=folds[shape]
        prior=json.loads((ROOT/f'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-{fold}.json').read_text())['model']
        state=json.loads((source/f'depth-fold{fold}.json').read_text())['model']['stage_prior']
        co=dict(zip(prior['fused']['names'],prior['fused']['coefficients']))
        cn=dict(zip(state['fit']['names'],state['fit']['coefficients']))
        def raw(row):
            orig=dict(zip(stage.additive.FUSED_NAMES,stage.additive.fused_features(stage.geometry(row),prior['capacity'])))
            ns,xx=stage.features(row,'depth',state['knots'],state['capacity']);x=dict(zip(ns,xx))
            return {'depth_curve':1000*(sum(cn[n]*v for n,v in x.items() if n.startswith('depth_'))-co['partition_K']*orig['partition_K']),
                    'other_coefficients':1000*sum((cn[n]-co[n])*v for n,v in orig.items() if n!='partition_K')}
        parts={}
        for name in (old['provisional_choice'],new['provisional_choice']):
            delta=raw(rows[shape,name]);point=new['curve_details'][name]['curve']
            if point:
                for k,w in zip(point['anchor_k'],point['weights']):
                    anchor=(shape[0],shape[1],k,shape[3]);da=raw(rows[anchor,name])
                    for term in delta:delta[term]-=w*da[term]
            actual=math.exp(new['predicted_log_us'][name])-math.exp(old['predicted_log_us'][name])
            assert abs(sum(delta.values())-actual)<1e-8
            parts[name]={**delta,'total_prediction_change_us':actual,'local_curve':point is not None}
        a=old['provisional_choice'];b=new['provisional_choice']
        results.append({'shape':shape,'old_choice':a,'new_choice':b,'measured_best':min(new['measured_us'],key=new['measured_us'].get),
            'candidate_changes_us':parts,'gap_newchoice_minus_oldchoice':{
                'before':math.exp(old['predicted_log_us'][b])-math.exp(old['predicted_log_us'][a]),
                'after':math.exp(new['predicted_log_us'][b])-math.exp(new['predicted_log_us'][a]),
                'measured':new['measured_us'][b]-new['measured_us'][a],
                'depth_contribution':parts[b]['depth_curve']-parts[a]['depth_curve'],
                'other_contribution':parts[b]['other_coefficients']-parts[a]['other_coefficients']}})
    (HERE/'attribution.json').write_text(json.dumps({'scope':'exact model-delta accounting, not identified hardware timing','decisions':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
