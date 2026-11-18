"""Trace the active path of the saved curvature-transfer development scorer.

This is an explanation API, not a new selector. It never changes routing,
coefficients, anchors, choices or support. Query timing labels are not read.
Model contributions are empirical accounting, not physical stage timings.
"""
from __future__ import annotations
import math

from . import curvature_transfer_selection as scorer
from .relative_selection import ScheduleCandidate


def host_base_terms(model, row):
    if not row['is_host'] or row['candidate_id'] != 'host':
        raise ValueError('per-source historical host required')
    prior = model['prior']
    if prior['host'] is not None:
        fit = prior['host']
        x = scorer.stage.curves.baseline.host_workload_features(row)
        name = 'workload_host_prior'
    else:
        p,m,k,n = row['shape']
        c = ScheduleCandidate(row['execution_context'], 'host', p,m,k,n,
                              2*p*(p-1)*m*k, 1,1,1,0)
        fit = prior['original']['host']
        x = scorer.stage.additive.host_features(c)
        name = 'original_log_M_quadratic_K_host_prior'
    if len(fit['names']) != len(x) or len(fit['coefficients']) != len(x):
        raise ValueError('host feature schema mismatch')
    return name, {n:1000*c*v for n,c,v in zip(fit['names'],fit['coefficients'],x)}


def fused_base_terms(model, row):
    state = model['stage_prior']
    transfer = state.get('curvature_transfer')
    if not transfer or transfer['schema'] != scorer.SCHEMA:
        raise ValueError('this explanation supports saved curvature-transfer models only')
    names,x = scorer.stage.features(row,'depth',state['knots'],state['capacity'])
    co = dict(zip(state['fit']['names'],state['fit']['coefficients']))
    co.update(transfer['non_depth_coefficients']); co['fixed'] = transfer['intercept_ms']
    return {n:1000*co[n]*v for n,v in zip(names,x)}


def explain(model, rows):
    prediction = scorer.predict(model, rows)  # owns identity/precision checks
    explanations = {}
    for row in sorted(rows, key=lambda r:r['candidate_id']):
        name = row['candidate_id']
        if name not in prediction['predicted_log_us']:
            continue
        detail = prediction['curve_details'][name]
        interval = model.get('host_interval') if row['is_host'] else None
        if row['is_host']:
            base_name, terms = host_base_terms(model, row)
        else:
            base_name, terms = 'curvature_transfer_fused_prior', fused_base_terms(model,row)
        base = math.fsum(terms.values())
        if not math.isclose(base, detail['prior_us'], abs_tol=1e-8, rel_tol=0):
            raise AssertionError('base trace differs from actual scoring path')
        used_name, applied_terms, residual = base_name, terms, 0.
        fallback = None
        inactive_interval = None
        if detail['host_interval_applied']:
            used_name = 'host_interval_'+interval.get('m_basis','log')
            values = scorer.stage.curves.host_interval_features(row,interval['knots'],interval.get('m_basis','log'))
            fit = interval['fit']
            applied_terms = {n:1000*c*v for n,c,v in zip(fit['names'],fit['coefficients'],values)}
        elif detail['curve_applied']:
            if model['mode'] == 'latency':
                used_name, applied_terms = 'absolute_K_interpolation', {}
            else:
                used_name = base_name+'_with_K_residual'
            residual = detail['curve']['value_us']
        if interval is not None and not detail['host_interval_applied']:
            inactive_interval = scorer.stage.curves.interval_host_prediction(interval,row)
            fallback = ('outside_host_interval_K_range' if inactive_interval is None
                        else 'no_same_M_N_candidate_precision_K_bracket')
        reconstructed = math.fsum(applied_terms.values())+residual
        if not math.isclose(reconstructed,detail['prediction_us'],abs_tol=1e-8,rel_tol=0):
            raise AssertionError('final trace differs from actual scoring path')
        if not math.isclose(math.exp(prediction['predicted_log_us'][name]),reconstructed,abs_tol=1e-8,rel_tol=0):
            raise AssertionError('log-time reconstruction differs')
        explanations[name] = {
            'active_path':used_name, 'nominal_model_mode':model['mode'],
            'fallback_reason':fallback, 'base_terms_us':terms,
            'active_terms_us':applied_terms, 'K_anchor_adjustment_us':residual,
            'prediction_us':detail['prediction_us'],
            'inactive_host_interval_us':inactive_interval,
            'inactive_value_is_used_for_selection':False,
            'same_identity_K_anchor':detail['curve'],
            'support':prediction['support'].get(name),
            'terms_are_measured_components':False,
        }
    return {'prediction':prediction, 'explanations':explanations,
            'scoring_rule_changed':False, 'compiler_admission':False}
