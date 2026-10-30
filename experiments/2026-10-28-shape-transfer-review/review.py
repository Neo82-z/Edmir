"""Fixed work/reuse controls on the original development folds. CPU only."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('axis_review',ROOT/'experiments/2026-10-28-local-simulation-revision/review.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from edmir.compiler import shape_transfer_selection as shape_model
axis=shape_model.axis;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def controls(base,training):
    return {'retained':(base,axis.base.predict),
            'M_work_extension':(axis.assemble(base,training,'M_work_extension'),axis.predict),
            **{mode:(shape_model.assemble(base,training,mode),shape_model.predict) for mode in shape_model.MODES}}


def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design=json.loads((HERE/'design-before.json').read_text())
    for p,h in {**design['source_hashes'],**design['protected_hashes']}.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior input changed: '+p)
    sources=[Path(__file__),ROOT/'EDM-IR/edmir/compiler/shape_transfer_selection.py']
    (out/'execution-before.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sources},indent=2)+'\n')
    decisions={mode:[] for mode in ('retained','M_work_extension',*shape_model.MODES)};stress={}
    with prior.previous.no_fitting() as calls:
        rows,folds,models=prior.previous.diagnosis.load_inputs()
        for fold,base in models.items():
            train=[r for r in rows if folds[tuple(r['shape'])]!=fold]
            query=[r for r in rows if folds[tuple(r['shape'])]==fold]
            for mode,(model,predict) in controls(base,train).items():
                dd=prior.evaluate(model,query,predict);decisions[mode]+=dd
                (out/f'{mode}-fold{fold}.json').write_text(json.dumps({'model':model,'decisions':dd},indent=2,allow_nan=False)+'\n')
            if time.monotonic()-start>60:raise TimeoutError('bounded control deadline')
        for name in ('all_M512','all_N2048','all_host_winners'):
            base=json.loads((prior.previous.diagnosis.SAVED.parent/'audit'/f'{name}-intercept.json').read_text())['model']
            ts={tuple(s) for s in base['training_shapes']};train=[r for r in rows if tuple(r['shape']) in ts];query=[r for r in rows if tuple(r['shape']) not in ts]
            dd={mode:prior.evaluate(model,query,predict) for mode,(model,predict) in controls(base,train).items()}
            stress[name]={'metrics':{mode:prior.summary(d) for mode,d in dd.items()},'decisions':dd}
    metrics={mode:prior.summary(d) for mode,d in decisions.items()}
    admission={}
    for mode in shape_model.MODES:
        pairs=[(metrics[mode],metrics['retained'])]+[(s['metrics'][mode],s['metrics']['retained']) for s in stress.values()]
        passes=metrics[mode]['wrong']<metrics['retained']['wrong']
        passes &= all(n[k]<=o[k]+1e-12 for n,o in pairs for k in ('wrong','host_direction_wrong','mean_regret','max_regret','shape_mean_mape'))
        admission[mode]={'eligible_experimental_replacement':bool(passes),
            'near_perfect_local_gate':all(n['wrong']==0 for n,o in pairs),'compiler_admission':False}
    prior_result=json.loads((ROOT/'experiments/2026-10-28-local-simulation-revision/verified/results.json').read_text())
    for mode in ('retained','M_work_extension'):
        if metrics[mode]!=prior_result['metrics'][mode]:raise AssertionError('reference changed')
        a={tuple(d['shape']):d['prediction']['predicted_log_us'] for d in decisions[mode]}
        b={tuple(d['shape']):d['prediction']['predicted_log_us'] for d in prior_result['decisions'][mode]}
        if a!=b:raise AssertionError('reference scores changed')
    for p,h in {**design['source_hashes'],**design['protected_hashes']}.items():
        if sha(ROOT/p)!=h:raise AssertionError('prior input changed: '+p)
    result={'metrics':metrics,'stress':stress,'decisions':decisions,'admission':admission,'training_calls':calls,
            'prior_scores_identical':True,'frozen_six_parsed_or_scored':False,'GPU_started':False,'seconds':time.monotonic()-start}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':metrics,'stress':{k:v['metrics'] for k,v in stress.items()},'admission':admission,'seconds':result['seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
