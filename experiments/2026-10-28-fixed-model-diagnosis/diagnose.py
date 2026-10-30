"""Explain frozen scores and observed contrasts. No fitting or policy edits.

The decomposition is exact model arithmetic, not isolated GPU service times.
Every queried shape belongs to the existing P4 development set. The frozen
six-shape files are hashed for integrity, never parsed or scored here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'EDM-IR'))
from edmir.compiler import curvature_transfer_selection as scorer

DATA = ROOT / 'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json'
SAVED = ROOT / 'experiments/2026-10-28-depth-cause-review/final-validation'
ARCHIVE = ROOT / 'experiments/2026-10-20-offline-predictor/data/selector_measurements.jsonl'
PROTECTED = ROOT / 'experiments/2026-10-28-host-decision-audit/protected-before.json'
TARGETS = (
    ((4, 64, 512, 256), 'direct', 'splitk2'),
    ((4, 64, 512, 1024), 'direct', 'splitk2'),
    ((4, 128, 512, 512), 'direct', 'splitk2'),
    ((4, 64, 8192, 1024), 'splitk4', 'splitk8_groupedn2'),
)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


@contextmanager
def forbid_training():
    calls = []
    def forbidden(*args, **kwargs):
        calls.append('training_attempt')
        raise RuntimeError('fitting is forbidden in fixed-model diagnosis')
    with ExitStack() as stack:
        for module, name in (
            (scorer, 'fit'), (scorer.stage, 'fit'),
            (scorer.stage.curves, 'fit'), (scorer.stage.curves, 'fit_host_intervals'),
            (scorer.stage.additive, '_solve'),
        ):
            stack.enter_context(patch.object(module, name, forbidden))
        yield calls


def load_inputs():
    data = json.loads(DATA.read_text())['p4']
    folds = {tuple(r['shape']): r['fold'] for r in data['folds']}
    models = {f: json.loads((SAVED / f'intercept-fold{f}.json').read_text())['model'] for f in range(4)}
    if len(data['rows']) != 346 or len(folds) != 41:
        raise ValueError('unexpected development set')
    return data['rows'], folds, models


def components(model, row):
    if row['is_host']:
        raise ValueError('fused contrast only')
    state = model['stage_prior']
    names, values = scorer.stage.features(row, 'depth', state['knots'], state['capacity'])
    coefficients = dict(zip(state['fit']['names'], state['fit']['coefficients']))
    transfer = state['curvature_transfer']
    coefficients.update(transfer['non_depth_coefficients'])
    coefficients['fixed'] = transfer['intercept_ms']
    terms = {n: 1000 * coefficients[n] * v for n, v in zip(names, values)}
    if not math.isclose(math.fsum(terms.values()), scorer.transferred_value(state, row), abs_tol=1e-8, rel_tol=0):
        raise AssertionError('raw component reconstruction failed')
    return terms


def execution_counts(row, capacity):
    c = scorer.stage.geometry(row)
    g, q = c.n_tiles_per_cta, c.split_k_slices
    groups = ((c.rows_per_peer + 63)//64) * ((c.n + 128*g-1)//(128*g))
    warps = (64//32) * ((128*g)//64)
    return {
        'scope': 'one source GEMM; structural counts, not achieved occupancy or total program time',
        'cta_tile': [64, 128*g, 32], 'warp_tile': [32, 64, 32],
        'independent_output_chains': groups, 'split_ctas': groups*q,
        'depth_k_iterations': c.k//(32*q), 'warps_per_cta': warps,
        'threads_per_cta': 32*warps,
        'serial_interpartition_edges': groups*(q-1),
        'kernel_launches_per_gemm': 1,
        'padded_mnk_flops_per_gemm': 2*groups*64*(128*g)*c.k,
        'logical_mnk_flops_per_gemm': 2*c.rows_per_peer*c.n*c.k,
        'modeled_pressure': max(groups*q/capacity-1, 0),
        'capacity_is_measured_occupancy': False,
    }


def contrast(model, surface, first, second):
    p = scorer.predict(model, surface)
    rows = {r['candidate_id']: r for r in surface}
    a, b = rows[first], rows[second]
    ca, cb = components(model, a), components(model, b)
    delta = {n: cb[n]-ca[n] for n in ca}
    residual = {n: p['curve_details'][n]['prediction_us']-p['curve_details'][n]['prior_us'] for n in (first, second)}
    delta['local_residual'] = residual[second]-residual[first]
    predicted = math.exp(p['predicted_log_us'][second])-math.exp(p['predicted_log_us'][first])
    reconstruction_error = abs(math.fsum(delta.values())-predicted)
    if reconstruction_error > 1e-8:
        raise AssertionError('contrast decomposition failed')
    sensitivity = {
        n: {'term_difference_us': v,
            'relative_coefficient_change_to_zero_raw_gap': abs(predicted/v)}
        for n,v in delta.items() if abs(v)>1e-12 and n!='local_residual'
    }
    if any(residual.values()):
        raise ValueError('sensitivity certificate requires no local anchor correction')
    observed = b['latency_us']-a['latency_us']
    return {
        'shape': a['shape'], 'contrast': f'{second} - {first}',
        'predicted_gap_us': predicted, 'observed_median_gap_us': observed,
        'gap_error_us': predicted-observed,
        'term_differences_us': delta, 'reconstruction_error_us': reconstruction_error,
        'single_coefficient_sensitivity': sensitivity,
        'sensitivity_semantics': 'analytic break-even, not uncertainty estimate or a parameter update',
        'sum_abs_nonzero_terms_us': math.fsum(abs(v) for v in delta.values()),
        'cancellation_ratio': math.fsum(abs(v) for v in delta.values()) / abs(predicted) if predicted else None,
        'counts': {n: execution_counts(rows[n], model['stage_prior']['capacity']) for n in (first, second)},
        'local_curves': {n: p['curve_details'][n]['curve'] for n in (first, second)},
    }


def local_structure(rows, models):
    short = sorted({tuple(r['shape']) for r in rows if r['shape'][2]==512})
    reports = []
    for fold, model in models.items():
        state = model['stage_prior']; differences = []
        names = None
        for shape in short:
            rr = {r['candidate_id']:r for r in rows if tuple(r['shape'])==shape}
            names, a = scorer.stage.features(rr['direct'], 'depth', state['knots'], state['capacity'])
            _, b = scorer.stage.features(rr['splitk2'], 'depth', state['knots'], state['capacity'])
            if any(execution_counts(rr[n],state['capacity'])['modeled_pressure'] for n in ('direct','splitk2')):
                raise ValueError('short-K pressure branch changed')
            differences.append(np.asarray(b)-a)
        matrix = np.asarray(differences)
        active = [n for n,x in zip(names,matrix.T) if np.any(x)]
        # A change +1 to serial_stages and +16 to depth_8_16 cancels
        # in every q2-direct contrast at K512. This is an identity, not a fit.
        null = np.zeros(len(names)); null[names.index('serial_stages')]=1; null[names.index('depth_8_16')]=16
        residual = float(np.max(np.abs(matrix@null)))
        if residual>1e-12:
            raise AssertionError('local null direction changed')
        reports.append({'fold':fold,'local_contrast_count':len(short),'feature_count':len(names),
                        'local_contrast_rank':int(np.linalg.matrix_rank(matrix)),
                        'active_terms':active,'serial_depth_null_residual':residual})
    observed = []
    for shape in short:
        rr = {r['candidate_id']:r for r in rows if tuple(r['shape'])==shape}
        observed.append({'shape':shape,'q2_minus_direct_us':rr['splitk2']['latency_us']-rr['direct']['latency_us']})
    first = next(d['q2_minus_direct_us'] for d in observed if d['shape']==(4,64,512,128))
    middle = next(d['q2_minus_direct_us'] for d in observed if d['shape']==(4,64,512,512))
    return {
        'formula': 'raw gap = 1000*(serial_stages + ctas*G/108 - depth_8_16/16)',
        'conditions': 'K512; g1; q1/q2; both CTA counts <= capacity; no residual correction',
        'per_fold':reports, 'observed_short_k_contrasts':observed,
        'nondecreasing_gap_minimum_max_error_us':max(0,(first-middle)/2),
        'bound_scope':'two observed gap labels, one shared raw function; not a four-fold selection lower bound',
        'identification_limit':'these local contrasts cannot separately identify depth saving and serial cost; no global rank or causal claim',
    }


def grouped_factorial(model, surface):
    p = scorer.predict(model,surface)
    rr = {r['candidate_id']:r for r in surface}
    names = ('splitk4','splitk8','splitk4_groupedn2','splitk8_groupedn2')
    pred = {n:math.exp(p['predicted_log_us'][n]) for n in names}
    obs = {n:rr[n]['latency_us'] for n in names}
    def contrasts(t):
        return {
            'q8_minus_q4_at_g1':t[names[1]]-t[names[0]],
            'q8_minus_q4_at_g2':t[names[3]]-t[names[2]],
            'g2_minus_g1_at_q4':t[names[2]]-t[names[0]],
            'g2_minus_g1_at_q8':t[names[3]]-t[names[1]],
            'q_by_group_interaction':t[names[3]]-t[names[2]]-t[names[1]]+t[names[0]],
            'target_diagonal':t[names[3]]-t[names[0]],
        }
    pc,oc = contrasts(pred),contrasts(obs)
    return {'predicted_times_us':pred,'observed_median_times_us':obs,'predicted_contrasts_us':pc,
            'observed_contrasts_us':oc,'contrast_errors_us':{n:pc[n]-oc[n] for n in pc},
            'meaning':'a complete measured 2x2 contrast; not an isolated causal cost measurement'}


def archive_evidence(rows):
    allowed = {(tuple(r['shape']),r['candidate_id']) for r in rows}
    wanted = {(s,n) for s,a,b in TARGETS for n in (a,b)}
    wanted |= {((4,64,8192,1024),n) for n in ('splitk8','splitk4_groupedn2')}
    selected = []
    # Read the existing archive, select only the declared development rows.
    for line in ARCHIVE.read_text().splitlines():
        r=json.loads(line); x=r['measurement']
        shape=(x['world_size'],x['rows_per_peer'],x['k'],x['n']);key=(shape,x['family'])
        if key not in wanted or key not in allowed or r['environment_id']!='a100-sxm4-80gb-p4-2026-08-29':continue
        if r['historical_role']!='development':raise ValueError('non-development evidence selected')
        archived=next(a for a in rows if (tuple(a['shape']),a['candidate_id'])==key)
        if not math.isclose(archived['latency_us'],1000*r['reported_timing']['median_ms'],abs_tol=1e-9,rel_tol=0):
            raise AssertionError('archive median mismatch')
        selected.append({'shape':shape,'candidate':x['family'],'reported_timing':r['reported_timing'],
                         'repeat_observation':r['repeat_observation'],'source':r['source'],
                         'timing_scope':r['timing_scope']})
    if len(selected)!=10:raise ValueError('missing or duplicate target archive rows')
    return {'rows':selected,'candidate_statistics_are_not_paired_difference_CIs':True,
            'historical_median_objective_unchanged':True,'unrecorded_repeats_not_inferred':True}


def main(out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    protected=json.loads(PROTECTED.read_text())
    files=[DATA,ARCHIVE,SAVED/'results.json',Path(__file__)]
    files += [SAVED/f'intercept-fold{f}.json' for f in range(4)]
    files += sorted((ROOT/'EDM-IR/edmir/compiler').glob('*.py'))
    files += [ROOT/'EDM-IR/runtime/src'/n for n in ('nccl_device_fused_cutlass.cu','cutlass_sm80_gemm.cuh')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError('protected file changed: '+p)
    design={'targets':TARGETS,'max_seconds':30,'fit_allowed':False,'parameter_updates':False,
            'frozen_six_parsed_or_scored':False,'source_hashes':hashes,'protected_hashes':protected,
            'diagnostics':['exact score decomposition','local contrast rank identity','analytic coefficient sensitivity',
                           'fixed grouped 2x2 contrasts','existing timing provenance'],
            'gpu_started':False,'decision_rule_changed':False}
    (out/'design-before.json').write_text(json.dumps(design,indent=2)+'\n')
    with forbid_training() as calls:
        rows,folds,models=load_inputs()
        saved=json.loads((SAVED/'results.json').read_text())['variants']['intercept']['decisions']
        for d in saved:
            rr=[r for r in rows if r['shape']==d['shape']]
            p=scorer.predict(models[folds[tuple(d['shape'])]],rr)
            if p['provisional_choice']!=d['provisional_choice'] or p['predicted_log_us']!=d['predicted_log_us']:
                raise AssertionError('frozen scores changed')
        targets=[]
        for s,a,b in TARGETS:
            targets.append(contrast(models[folds[s]],[r for r in rows if tuple(r['shape'])==s],a,b))
        gs=(4,64,8192,1024)
        result={'targets':targets,'local_structure':local_structure(rows,models),
                'grouped_factorial':grouped_factorial(models[folds[gs]],[r for r in rows if tuple(r['shape'])==gs]),
                'archive':archive_evidence(rows),'training_calls':len(calls),'parameter_updates':0,
                'saved_41_shape_scores_identical':True,'frozen_six_parsed_or_scored':False,
                'gpu_started':False,'compiler_admission':False}
    if time.monotonic()-started>30:raise TimeoutError('diagnosis exceeded fixed budget')
    for p,h in {**protected,**hashes}.items():
        if sha(ROOT/p)!=h:raise AssertionError('input changed: '+p)
    result.update(protected_unchanged=True,source_and_model_hashes_unchanged=True,seconds=time.monotonic()-started)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'target_count':len(targets),'training_calls':len(calls),'scores_unchanged':True,
                      'seconds':result['seconds'],'output':str(out)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
