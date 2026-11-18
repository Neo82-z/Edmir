#!/usr/bin/env python3
"""Full-candidate historical P4/P8 replay; no GPU, no runtime promotion."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import signal
import statistics as st
import sys
import tarfile
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import load_selector_dataset, P4_ENV, P8_ENV
from edmir.compiler.scale_aware_prediction import ScaleAwareModel, geometry, legacy_graph
from edmir.compiler.two_stage_selection import TwoStageSelector
from replay_initial_experiments import summarize
from evaluate_relative_selector import fit_method

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'experiments/2026-10-25-initial-data-replay'
DATA = REPO/'experiments/2026-10-20-offline-predictor/data'
ARCHIVE = REPO/'experiments/2026-08-30-a100-8gpu/extracted/edmir-p8-src.tar.gz'
METHODS = ('original_full_log', 'scale_max', 'scale_retained_0.25', 'train_best_fixed', 'host')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def audit_archived_graph():
    """Execute only named modules from our preserved source, no tar extraction."""
    hashes, graphs = {}, {}
    prefix = '_edmir_archived_p8_replay'
    try:
        root = types.ModuleType(prefix); root.__path__ = []; sys.modules[prefix] = root
        with tarfile.open(ARCHIVE) as archive:
            for name in ('model', 'schedule', 'dependence'):
                path = 'EDM-IR/edmir/compiler/'+name+'.py'
                src = archive.extractfile(path).read()
                hashes[path] = hashlib.sha256(src).hexdigest()
                module = types.ModuleType(prefix+'.'+name)
                module.__package__ = prefix
                sys.modules[module.__name__] = module
                exec(compile(src, path, 'exec'), module.__dict__)
            model, schedule, dep = [sys.modules[prefix+'.'+x] for x in ('model','schedule','dependence')]
            for p in (4, 8):
                program = schedule.build_fused_program(model.A2AGemmProblem(p,64,4096,512), chunk_rows=64)
                dependencies = dep.derive_dependencies(program); dependencies.require_valid()
                prev = dependencies.predecessors(); depth = {}
                for op in dependencies.topological_order:
                    depth[op] = 1+max((depth[x] for x in prev[op]), default=0)
                counts = (len(program.ops), len(dependencies.dependencies), max(depth.values()))
                if counts != legacy_graph(p):
                    raise ValueError('scale geometry differs from archived dependency code')
                graphs[str(p)] = dict(zip(('ops','queue_edges','unit_duration_chain'), counts))
            for name in ('cutlass_sm80_gemm.cuh', 'nccl_device_fused_cutlass.cu'):
                path = 'EDM-IR/runtime/src/'+name
                hashes[path] = hashlib.sha256(archive.extractfile(path).read()).hexdigest()
    finally:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix+'.'):
                del sys.modules[key]
    return {'archive_sha256':sha(ARCHIVE), 'source_hashes':hashes, 'graphs':graphs,
            'status':'independent archived-code verification passed'}


def fit(method, rows, env):
    if method == 'original_full_log':
        return TwoStageSelector.fit(rows, ablation='full', ridge=1.)
    if method.startswith('scale_'):
        return ScaleAwareModel.fit(rows, environment_id=env,
            retained_fraction=.25 if method.endswith('0.25') else 0.)
    return fit_method(method, rows, env)


def forecast(model, rows, env):
    output=[]
    for row in rows:
        if isinstance(model, ScaleAwareModel):
            value=model.predict(row.shape, row.family, environment_id=env,
                                allow_transfer=env != model.environment_id)
        elif isinstance(model, str):
            # Fixed policies have ranks, not latency predictions.
            value={'predicted_us': 0. if row.family == model else 1., 'latency_prediction':False}
        else:
            regressor=model.host_model if row.is_host else model.fused_model
            value={'predicted_us':1000*regressor.predict_ms(row)}
        output.append({'shape':row.shape, 'family':row.family, 'group':row.group_id,
                       'observed_us':1000*row.measured_ms, **value})
    return output


def score(rows, *, fixed=False):
    result=summarize(rows)
    byshape=defaultdict(dict)
    for row in rows:
        byshape[tuple(row['shape'])][row['family']]=row
    top1=[d for d in result['decisions'] if d['budget']==1]
    for d in top1:
        group=byshape[tuple(d['shape'])]; chosen=group[d['winner']]
        host=group['host']; best=group[d['measured_subset_best']]
        d.update(host_us=host['observed_us'], selected_us=chosen['observed_us'],
                 oracle_us=best['observed_us'],
                 direction_correct=(d['winner']=='host') == (d['measured_subset_best']=='host'),
                 host_regression=chosen['observed_us'] > host['observed_us']*(1+1e-12))
    result['direction_accuracy']=st.mean(d['direction_correct'] for d in top1)
    result['host_regressions']=sum(d['host_regression'] for d in top1)
    result['historical_gates_pass']=(result['direction_accuracy']>=.99 and result['host_regressions']==0
        and result['selection']['1']['mean_loss']<=.005 and result['selection']['1']['max_loss']<=.01)
    if fixed:
        for key in ('macro_shape_MAPE','max_APE','wrong_pairs','pairs'):
            result.pop(key)
        result['selection']={'1':result['selection']['1']}
        result['decisions']=top1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    if args.out.exists(): parser.error('fresh output directory required')
    start=time.monotonic()
    def expired(*_): raise TimeoutError('120 second total replay bound exceeded')
    signal.signal(signal.SIGALRM,expired); signal.alarm(120)
    design=json.loads((BASE/'p8-design.json').read_text())
    source_audit=audit_archived_graph()
    p4,cal,evaluation,manifest=load_selector_dataset(DATA)
    if {r.shape for r in cal} != {tuple(x) for x in design['calibration_shapes']}:
        raise ValueError('original calibration split changed')
    if len(cal)!=44 or len(evaluation)!=209 or {r.shape for r in cal}&{r.shape for r in evaluation}:
        raise ValueError('original P8 coverage changed')
    args.out.mkdir(parents=True)
    write(args.out/'design.json',design); write(args.out/'source-audit.json',source_audit)
    forecasts={}; models={}; reports={}
    for method in METHODS:
        forecasts[method]={}; models[method]={}; reports[method]={}
        output=[];folds=[]
        for group in sorted({r.group_id for r in p4}):
            train=[r for r in p4 if r.group_id!=group]; test=[r for r in p4 if r.group_id==group]
            if {r.shape for r in train}&{r.shape for r in test}: raise ValueError('sweep leakage')
            fitted=fit(method,train,P4_ENV); output.extend(forecast(fitted,test,P4_ENV))
            folds.append({'excluded_group':group,'model':fitted if isinstance(fitted,str) else fitted.to_dict()})
        forecasts[method]['p4_group_heldout']=output; models[method]['p4_group_heldout']=folds
        for protocol,train,env in (('p4_to_p8_no_calibration',p4,P4_ENV),('p8_four_shape_refit',cal,P8_ENV)):
            fitted=fit(method,train,env)
            forecasts[method][protocol]=forecast(fitted,evaluation,P8_ENV)
            models[method][protocol]=fitted if isinstance(fitted,str) else fitted.to_dict()
        for protocol,rs in forecasts[method].items():
            reports[method][protocol]=score(rs,fixed=method in ('host','train_best_fixed'))
        write(args.out/'progress.json',{'completed_method':method,'elapsed_seconds':time.monotonic()-start})
    # Reference original-model choice metrics against a separate legacy runner.
    old=json.loads((BASE/'original-full-replay.json').read_text())['reports']['original']
    for protocol,rs in reports['original_full_log'].items():
        for a,b in (('mean_loss','mean_selection_regret'),('max_loss','max_selection_regret')):
            if abs(rs['selection']['1'][a]-old[protocol]['metrics'][b])>1e-10:
                raise ValueError('original reference metric changed')
    write(args.out/'predictions.json',forecasts); write(args.out/'models.json',models)
    write(args.out/'result.json',{'status':'retrospective development; no compiler admission',
        'GPU_runs':0,'counts':manifest['counts'],'reports':reports,'independent_validation':False})
    write(args.out/'geometry.json',[geometry(r.shape,r.family) for r in (*p4,*cal,*evaluation)])
    sources=[Path(__file__),BASE/'p8-design.json',BASE/'original-full-replay.json',
             DATA/'manifest.json',DATA/'selector_measurements.jsonl']
    sources.extend(REPO/'EDM-IR/edmir/compiler'/n for n in ('scale_aware_prediction.py',
        'measurement_dataset.py','regime_prediction.py','mechanism_prediction.py','two_stage_selection.py',
        'prediction_support.py','host_baseline.py','relative_selection.py'))
    sources.extend(Path(__file__).with_name(n) for n in ('replay_initial_experiments.py','evaluate_relative_selector.py','validate_p8_replay.py'))
    sources.append(REPO/'EDM-IR/tests/scale_aware_prediction_test.py')
    snapshot=args.out/'source-snapshot';snapshot.mkdir()
    for f in sources:
        if f.suffix=='.py': (snapshot/f.name).write_bytes(f.read_bytes())
    write(args.out/'seal.json',{'inputs':{str(f.relative_to(REPO)):sha(f) for f in sources},
        'outputs':{str(f.relative_to(args.out)):sha(f) for f in args.out.rglob('*') if f.is_file()},
        'elapsed_seconds':time.monotonic()-start,'GPU_runs':0})
    signal.alarm(0)
    print(json.dumps({m:{p:{'top1':r['selection']['1'],'MAPE':r.get('macro_shape_MAPE'),
        'gates':r['historical_gates_pass']} for p,r in ps.items()} for m,ps in reports.items()}))


if __name__=='__main__': main()
