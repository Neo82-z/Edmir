"""One prespecified no-fit control and source-derived conditional model audit."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'EDM-IR/tools')]
from edmir.compiler import dependency_wave_model as structure
from edmir.compiler import curvature_transfer_selection as scorer
from review_calibrated_structure import aggregate

spec = importlib.util.spec_from_file_location('fixed_diagnosis', ROOT/'experiments/2026-10-28-fixed-model-diagnosis/diagnose.py')
diagnosis = importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnosis)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


@contextmanager
def no_fitting():
    with diagnosis.forbid_training() as calls, ExitStack() as stack:
        def fail(*a, **kw):
            calls.append('forbidden additional fit'); raise RuntimeError('no fitting allowed')
        stack.enter_context(patch.object(scorer.stage.curves.baseline, 'fit', fail))
        stack.enter_context(patch.object(scorer.stage.curves.baseline.old, 'fit', fail))
        yield calls


def capacity_control(model, static_training_rows):
    """Negative control, NOT occupancy-based physical prediction.

    Change only the pressure coordinate's denominator to 108. Coefficients,
    host model and observed anchor values remain fixed. Shift residuals by the
    change in base prediction, preserving exact training anchor interpolation.
    No query labels or new parameter estimation enter this transformation.
    """
    expected = {tuple(s) for s in model['training_shapes']}
    if {tuple(r['shape']) for r in static_training_rows} != expected:
        raise ValueError('training-only anchor geometry required')
    new = deepcopy(model)
    new['stage_prior']['capacity'] = 108
    touched = set()
    for row in static_training_rows:
        if row['is_host']: continue
        key = scorer.stage.curves.curve_key(row)
        anchor = next(a for a in new['curves'][key] if a['k'] == row['shape'][2])
        if row['source_id'] != anchor['source_id'] or row['execution_context'] != model['execution_context']:
            raise ValueError('anchor identity mismatch')
        identity = (key, anchor['k'])
        if identity in touched: raise ValueError('duplicate training geometry')
        touched.add(identity)
        delta = scorer.transferred_value(new['stage_prior'], row) - scorer.transferred_value(model['stage_prior'], row)
        anchor['residual_us'] -= delta
    new['capacity_control'] = {'status': 'negative_control_not_admitted', 'capacity':108,
                               'learned_coefficients_changed': False}
    return new


def summarize(decisions):
    result = aggregate(decisions)
    result['wrong_choices'] = sum(d['provisional_regret'] > 0 for d in decisions)
    return result


def make_case(row):
    return structure.KernelCase.from_candidate(scorer.stage.geometry(row), precision='fp16_fp32acc_fp16out')


def main(out):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    design = json.loads((HERE/'design-before.json').read_text())
    hashes = {**design['source_hashes'], **design['protected_hashes']}
    for path, value in hashes.items():
        if sha(ROOT/path) != value: raise AssertionError('protected input changed: '+path)
    sources = [Path(__file__), ROOT/'EDM-IR/edmir/compiler/dependency_wave_model.py',
               ROOT/'EDM-IR/tests/dependency_wave_model_test.py']
    (out/'execution-before.json').write_text(json.dumps({
        'design_sha256':sha(HERE/'design-before.json'),
        'new_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'fit_allowed':False, 'parameter_search':False}, indent=2)+'\n')
    with no_fitting() as fit_calls:
        rows, folds, models = diagnosis.load_inputs()
        old_saved = json.loads((diagnosis.SAVED/'results.json').read_text())['variants']['intercept']['decisions']
        originals, controls = [], []
        for fold, model in models.items():
            train = [r for r in rows if folds[tuple(r['shape'])] != fold]
            test = [{**r, 'split':'evaluation'} for r in rows if folds[tuple(r['shape'])] == fold]
            alternative = capacity_control(model, train)
            before = scorer.evaluate(model, test); after = scorer.evaluate(alternative, test)
            if alternative['stage_prior']['fit'] != model['stage_prior']['fit'] or alternative['stage_prior']['curvature_transfer'] != model['stage_prior']['curvature_transfer']:
                raise AssertionError('fitted constants changed')
            for a, b in zip(before, after):
                saved = next(d for d in old_saved if d['shape'] == a['shape'])
                if a['predicted_log_us'] != saved['predicted_log_us']: raise AssertionError('old predictions changed')
                if a['predicted_log_us']['host'] != b['predicted_log_us']['host']: raise AssertionError('host changed')
            originals.extend(before); controls.extend(after)
        stress = {}
        for cut in ('all_M512', 'all_N2048', 'all_host_winners'):
            stored = json.loads((diagnosis.SAVED.parent/f'audit/{cut}-intercept.json').read_text())
            model = stored['model']; shapes = {tuple(s) for s in model['training_shapes']}
            train = [r for r in rows if tuple(r['shape']) in shapes]
            test = [{**r, 'split':'evaluation'} for r in rows if tuple(r['shape']) not in shapes]
            original = scorer.evaluate(model, test)
            for d in original:
                saved = next(x for x in stored['evaluation']['decisions'] if x['shape']==d['shape'])
                if saved['predicted_log_us'] != d['predicted_log_us']: raise AssertionError('stress replay differs')
            control = scorer.evaluate(capacity_control(model, train), test)
            stress[cut] = {'original':summarize(original), 'capacity108_control':summarize(control)}

        # There are no GPU service values here: this fixed fixture checks algebra.
        inventories, traversal_ms = [], []
        for repeat in range(4):
            begin = time.perf_counter()
            for row in rows:
                if row['is_host']: continue
                case = make_case(row)
                service = structure.ServiceScenario(1+case.depth, 1, 1, 'synthetic_units',
                    'fixed fixture only: mainloop=1+d, first/later epilogue=1; not estimated GPU service')
                results = {order:structure.simulate(case, service, slots=108, order=order)
                           for order in structure.ADMISSION_ORDERS}
                if repeat == 0:
                    inventories.append({'shape':row['shape'], 'candidate':row['candidate_id'],
                        'counts':case.counts(), 'resources':structure.resource_envelope(case),
                        'conditional_fixture':results, 'prediction_us':None})
            traversal_ms.append((time.perf_counter()-begin)*1000)
            if time.monotonic()-started > 60: raise TimeoutError('fixed review budget')

        targets = []
        for shape in ((4,64,512,128), (4,64,512,256), (4,64,512,512), (4,64,512,1024), (4,128,512,512)):
            rr = {r['candidate_id']:r for r in rows if tuple(r['shape']) == shape}
            a, b = make_case(rr['direct']), make_case(rr['splitk2'])
            targets.append({'shape':shape, 'ctas_direct_q2':[a.ctas,b.ctas],
                'both_all_fit_at_108':max(a.ctas,b.ctas)<=108,
                'constant_service_gap_expression':'mainloop(depth8,g1,serial)-mainloop(depth16,g1,direct)+first_epilogue_serial-first_epilogue_direct+later_epilogue(g1)',
                'measured_gap_us':rr['splitk2']['latency_us']-rr['direct']['latency_us']})
        grouped_shape = (4,64,8192,1024)
        rr = [r for r in rows if tuple(r['shape'])==grouped_shape]
        grouped = diagnosis.grouped_factorial(models[folds[grouped_shape]], rr)
        grouped['all_four_fit_at_108'] = all(make_case(r).ctas <=108 for r in rr
            if r['candidate_id'] in ('splitk4','splitk8','splitk4_groupedn2','splitk8_groupedn2'))
        grouped['conditional_interaction_expression'] = (
            '[a_g2(32)-a_g2(64)]-[a_g1(32)-a_g1(64)]+4*(e_later_g2-e_later_g1)')
        grouped['old_empirical_pressure_contributions_us'] = {
            r['candidate_id']:{name:value for name,value in diagnosis.components(models[folds[grouped_shape]],r).items()
                               if 'pressure' in name}
            for r in rr if r['candidate_id'] in ('splitk4','splitk8','splitk4_groupedn2','splitk8_groupedn2')}
        # Signed prediction-error account, not a corrected model or a fit.
        m512 = []
        for d in stress['all_M512']['original']['decisions']:
            if d['provisional_regret'] <= 0: continue
            pred = {name:math.exp(value) for name,value in d['predicted_log_us'].items()}
            observed = d['measured_us']; winner = min(observed, key=observed.get)
            if winner == 'host': raise AssertionError('expected fused-winning M512 diagnostic')
            host_error = pred['host']-observed['host']
            fused_error = pred[winner]-observed[winner]
            # The oracle substitution is labelled and never passed to scoring.
            oracle = {**pred, 'host':observed['host']}
            oracle_choice = min(oracle, key=lambda n:(oracle[n],n))
            m512.append({'shape':d['shape'], 'predicted_us':pred, 'observed_us':observed,
                'actual_winner':winner, 'host_error_us':host_error, 'winner_error_us':fused_error,
                'predicted_winner_minus_host_us':pred[winner]-pred['host'],
                'observed_winner_minus_host_us':observed[winner]-observed['host'],
                'gap_error_us':fused_error-host_error,
                'host_share_of_signed_gap_error':-host_error/(fused_error-host_error),
                'oracle_host_substitution_not_a_prediction':{
                    'chosen':oracle_choice, 'regret':observed[oracle_choice]/min(observed.values())-1},
                'curve_details':d['curve_details']})
        changes = []
        for before in originals:
            after = next(x for x in controls if x['shape']==before['shape'])
            if before['provisional_choice'] != after['provisional_choice']:
                changes.append({'shape':before['shape'], 'before':before['provisional_choice'],
                    'after':after['provisional_choice'], 'before_regret':before['provisional_regret'],
                    'after_regret':after['provisional_regret']})
        result = {'original':summarize(originals), 'capacity108_control':summarize(controls),
            'changes':changes, 'stress':stress, 'short_k':targets, 'grouped':grouped,
            'M512_signed_error_account':m512,
            'structural_inventory':inventories,
            'conditional_traversal_ms':{'median':statistics.median(traversal_ms), 'max':max(traversal_ms), 'repetitions':4},
            'training_calls':len(fit_calls), 'coefficient_updates':0, 'new_gpu_measurements':0,
            'frozen_six_read_or_scored':False, 'compiler_admission':False,
            'conditional_model_calibrated':False, 'full_program_score_from_conditional_model':None,
            'old_scores_bitwise_identical':True, 'original_pressure_model_retained':True}
    for path,value in hashes.items():
        if sha(ROOT/path) != value: raise AssertionError('input changed: '+path)
    result['protected_unchanged'] = True
    result['seconds'] = time.monotonic()-started
    if result['seconds'] > 60: raise TimeoutError('fixed review budget')
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    compact = lambda d:{k:v for k,v in d.items() if k!='decisions'}
    print(json.dumps({'original':compact(result['original']), 'capacity108_control':compact(result['capacity108_control']),
        'changes':changes, 'stress':{s:{k:compact(v) for k,v in vals.items()} for s,vals in stress.items()},
        'training_calls':len(fit_calls), 'seconds':result['seconds'], 'traversal':result['conditional_traversal_ms']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True, type=Path)
    main(parser.parse_args().out)
