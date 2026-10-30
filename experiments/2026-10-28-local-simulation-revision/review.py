"""Finite predeclared controls, original folds, no fitting or frozen labels."""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'experiments/2026-10-28-dependency-structure-review')]
import review as previous
from edmir.compiler import axis_residual_selection as axis

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(model, rows, predictor):
    if {tuple(r['shape']) for r in rows} & {tuple(s) for s in model['training_shapes']}:
        raise ValueError('whole-shape leakage')
    decisions=[]
    for shape in sorted({tuple(r['shape']) for r in rows}):
        surface=[r for r in rows if tuple(r['shape'])==shape]
        p=predictor(model,surface)
        if p != predictor(model,list(reversed(surface))): raise AssertionError('order bias')
        poison=deepcopy(surface)
        for r in poison: r.pop('latency_us'); r.pop('measurement_kind',None)
        if p != predictor(model,poison): raise AssertionError('query labels used')
        actual={r['candidate_id']:r['latency_us'] for r in surface}
        if set(actual)!=set(p['predicted_log_us']): raise AssertionError('candidate removed')
        chosen=p['provisional_choice'];best=min(actual.values())
        decisions.append({'shape':list(shape),'chosen':chosen,'best':min(actual,key=actual.get),
            'wrong':actual[chosen]>best,'regret':actual[chosen]/best-1,
            'host_direction_wrong':(chosen=='host')!=(actual['host']==best),
            'mape':sum(abs(math.exp(v)/actual[n]-1) for n,v in p['predicted_log_us'].items())/len(actual),
            'prediction':p,'observed_us':actual})
    return decisions


def summary(decisions):
    return {'shapes':len(decisions),'wrong':sum(d['wrong'] for d in decisions),
            'host_direction_wrong':sum(d['host_direction_wrong'] for d in decisions),
            'mean_regret':sum(d['regret'] for d in decisions)/len(decisions),
            'max_regret':max(d['regret'] for d in decisions),
            'shape_mean_mape':sum(d['mape'] for d in decisions)/len(decisions)}


def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design=json.loads((HERE/'design-before.json').read_text())
    hashes={**design['source_hashes'],**design['protected_hashes']}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h: raise AssertionError('prior input changed: '+p)
    sources=[Path(__file__),ROOT/'EDM-IR/edmir/compiler/axis_residual_selection.py']
    (out/'execution-before.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sources},indent=2)+'\n')
    all_decisions={mode:[] for mode in ('retained',*axis.MODES)};stress={}
    with previous.no_fitting() as calls:
        rows,folds,models=previous.diagnosis.load_inputs()
        for fold,base in models.items():
            training=[r for r in rows if folds[tuple(r['shape'])]!=fold]
            queries=[r for r in rows if folds[tuple(r['shape'])]==fold]
            all_decisions['retained']+=evaluate(base,queries,axis.base.predict)
            for mode in axis.MODES:
                model=axis.assemble(base,training,mode)
                result=evaluate(model,queries,axis.predict)
                all_decisions[mode]+=result
                (out/f'{mode}-fold{fold}.json').write_text(json.dumps({'model':model,'decisions':result},indent=2,allow_nan=False)+'\n')
            if time.monotonic()-start>60: raise TimeoutError('local review deadline')
        for name in ('all_M512','all_N2048','all_host_winners'):
            base=json.loads((previous.diagnosis.SAVED.parent/'audit'/f'{name}-intercept.json').read_text())['model']
            train_shapes={tuple(s) for s in base['training_shapes']}
            training=[r for r in rows if tuple(r['shape']) in train_shapes]
            queries=[r for r in rows if tuple(r['shape']) not in train_shapes]
            dd={'retained':evaluate(base,queries,axis.base.predict)}
            for mode in axis.MODES:
                model=axis.assemble(base,training,mode)
                dd[mode]=evaluate(model,queries,axis.predict)
            stress[name]={'metrics':{mode:summary(d) for mode,d in dd.items()},'decisions':dd}
        metrics={mode:summary(d) for mode,d in all_decisions.items()}
        admission={}
        for mode in axis.MODES:
            ok=metrics[mode]['wrong']<metrics['retained']['wrong']
            pairs=[(metrics[mode],metrics['retained'])]+[(s['metrics'][mode],s['metrics']['retained']) for s in stress.values()]
            for new,old in pairs:
                ok &= all(new[key]<=old[key]+1e-12 for key in ('wrong','host_direction_wrong','mean_regret','max_regret','shape_mean_mape'))
            admission[mode]={'eligible_experimental_replacement':bool(ok),
                            'near_perfect_local_gate':all(n['wrong']==0 for n,o in pairs),
                            'compiler_admission':False}
    for p,h in hashes.items():
        if sha(ROOT/p)!=h: raise AssertionError('prior input changed: '+p)
    result={'metrics':metrics,'decisions':all_decisions,'stress':stress,'admission':admission,
            'training_calls':calls,'seconds':time.monotonic()-start,'protected_unchanged':True,
            'frozen_six_parsed_or_scored':False,'GPU_started':False}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':metrics,'stress':{n:s['metrics'] for n,s in stress.items()},'admission':admission,'seconds':result['seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
