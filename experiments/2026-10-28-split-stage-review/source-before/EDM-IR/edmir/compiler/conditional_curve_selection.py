"""Bounded K-curve control for a fixed execution identity and M/N.

This is an empirical end-to-end predictor, not isolated communication/compute
service identification. It tests interpolation against the same-fold prior.
No GPU, optimizer search, extrapolated anchors, or candidate-specific thresholds.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict
import json
import math

from . import calibrated_candidate_selection as baseline

SCHEMA = 'edmir.conditional-k-curve.v1'
MODES = ('prior', 'latency', 'prior_residual', 'host_interval', 'fused_residual', 'host_interval_fused_residual',
         'host_tiles', 'host_tiles_fused_residual')


def convex_chord_diagnostic(points):
    """Necessary minimax error for any convex function through three labels.

    This concerns the observed labels, not a noise-free hardware law, and does
    not constrain the nonlinear residual network added to a convex prior.
    """
    if len(points)!=3 or any(type(k) is not int or k<=0 or not math.isfinite(t) or t<=0 for k,t in points):
        raise ValueError('three finite positive observations required')
    (a,ta),(b,tb),(c,tc)=points
    if not a<b<c:raise ValueError('strictly ordered K observations required')
    w=(b-a)/(c-a);chord=(1-w)*ta+w*tc;excess=max(tb-chord,0.)
    return {'middle_k':b,'middle_us':tb,'chord_us':chord,'above_chord_us':tb-chord,
            'convex_model_minimum_max_absolute_error_us':excess/2,
            'convex_model_minimum_max_relative_error':excess/(tb+chord),
            'noise_free_causal_claim':False}


def host_interval_features(row, knots, m_basis='log'):
    """Nonnegative interval slopes allow either convex or concave K curves.

    M/N interactions are shared across all host shapes. Knots come only from
    training K values; no M512 indicator or target-dependent breakpoint.
    """
    baseline.host_workload_features(row)  # historical host/precision guard
    if not 2 <= len(knots) <= 64 or knots != sorted(set(knots)) or any(type(k) is not int or k<=0 for k in knots):
        raise ValueError('bounded sorted training K knots required')
    if m_basis not in ('log','tiles'):raise ValueError('unknown M basis')
    _,m,k,n=row['shape']; mt=(m+63)//64
    lm=math.log2(mt) if m_basis=='log' else mt-1
    geometry=(1.,lm,n/512)
    return (*geometry, *(v*max(0,min(k-lo,hi-lo))/4096
                         for lo,hi in zip(knots,knots[1:]) for v in geometry))


def fit_host_intervals(rows, prior, m_basis='log'):
    hosts=sorted((r for r in rows if r['is_host']),key=lambda r:tuple(r['shape']))
    knots=sorted({r['shape'][2] for r in hosts})
    m_name='log_M_tiles' if m_basis=='log' else 'additional_M_tiles'
    names=('fixed',m_name,'N')+tuple(f'K_{lo}_{hi}_{term}'
        for lo,hi in zip(knots,knots[1:]) for term in ('base',m_name,'N'))
    fitted=baseline.additive._solve([host_interval_features(r,knots,m_basis) for r in hosts],
        [r['latency_us']/1000 for r in hosts],[1.]*len(hosts),names,prior['ridge'])
    return {'knots':knots,'m_basis':m_basis,'fit':asdict(fitted),'coefficient_meaning':'empirical end-to-end slopes; not identified bandwidth'}


def interval_host_prediction(state, row):
    if not state['knots'][0] <= row['shape'][2] <= state['knots'][-1]:
        return None
    fit=state['fit'];features=host_interval_features(row,state['knots'],state.get('m_basis','log'))
    if len(fit['coefficients'])!=len(features) or any(type(c) not in (int,float) or not math.isfinite(c) or c<0 for c in fit['coefficients']):
        raise ValueError('invalid nonnegative interval coefficients')
    v=1000*math.fsum(c*x for c,x in zip(fit['coefficients'],features))
    if not math.isfinite(v) or v<=0:raise ValueError('invalid interval prediction')
    return v


def curve_key(row):
    p, m, _, n = row['shape']
    return json.dumps([row['execution_context'], p, m, n, row['candidate_id'],
                       baseline.old.precision_key(row), row.get('implementation_id')],
                      separators=(',', ':'))


def interpolate(knots, k, field):
    """No extrapolation. Return exact convex weights and anchor provenance."""
    if type(k) is not int or k <= 0 or not 1 <= len(knots) <= 256:
        raise ValueError('bounded positive K and knot table required')
    if any(type(v['k']) is not int or v['k'] <= 0 or
           not math.isfinite(v[field]) for v in knots):
        raise ValueError('finite knot values required')
    ks = [v['k'] for v in knots]
    if ks != sorted(set(ks)):
        raise ValueError('knots must be unique and sorted')
    i = bisect_left(ks, k)
    if i < len(ks) and ks[i] == k:
        selected, weights = [knots[i]], [1.]
    elif i == 0 or i == len(ks):
        return None
    else:
        selected = knots[i-1:i+1]
        alpha = (k-ks[i-1])/(ks[i]-ks[i-1])
        weights = [1-alpha, alpha]
    return {'value_us': math.fsum(w*r[field] for w,r in zip(weights,selected)),
            'anchor_k': [r['k'] for r in selected], 'weights': weights,
            'anchor_sources': [r['source_id'] for r in selected],
            'exact_training_point': len(selected)==1}


def fit(rows, prior, *, mode='prior_residual', interval_support='same_M_N_bracket'):
    if mode not in MODES:
        raise ValueError('unknown curve mode')
    if interval_support not in ('same_M_N_bracket','global_K_range'):
        raise ValueError('unknown interval support rule')
    surfaces = baseline.old.valid_rows(rows, 'train')
    if len(surfaces) < 3 or prior is None:
        raise ValueError('three training shapes and same-fold empirical prior required')
    wrapped = baseline.make_prior(rows, prior, 'original')
    # make_prior/prior_scores enforce the archive-specific P4/FP16 contract.
    ordered = [r for s in surfaces for r in s]
    offsets = baseline.prior_scores(wrapped, ordered)
    curves = {}
    for row, offset in zip(ordered, offsets):
        value = row['latency_us']
        curves.setdefault(curve_key(row), []).append({
            'k': row['shape'][2], 'latency_us': value,
            'residual_us': value-math.exp(float(offset)), 'source_id': row['source_id']})
    for knots in curves.values():
        knots.sort(key=lambda r:r['k'])
        interpolate(knots, knots[0]['k'], 'latency_us')
    return {'schema':SCHEMA, 'mode':mode, 'prior':wrapped, 'curves':curves,
            'host_interval':fit_host_intervals(rows,prior,'tiles' if mode.startswith('host_tiles') else 'log') if mode.startswith('host_') else None,
            'interval_support':interval_support,
            'training_shapes':[s[0]['shape'] for s in surfaces],
            'training_sources':sorted({r['source_id'] for r in rows}),
            'execution_context':rows[0]['execution_context'], 'world_size':rows[0]['shape'][0],
            'implementation_contracts':sorted({json.dumps([r['candidate_id'],r.get('implementation_id')]) for r in rows}),
            'supported_precision_contracts':sorted({baseline.old.precision_key(r) for r in rows}),
            'candidate_support':{key:sorted([r['shape'] for r in rows if baseline.support_key(r)==key])
                                 for key in sorted({baseline.support_key(r) for r in rows})},
            'prediction_semantics':'empirical end-to-end latency, not identified component service',
            'optimizer':('nonnegative least squares for host intervals; fixed fused interpolation'
                         if mode.startswith('host_') else 'none; fixed piecewise affine interpolant'),
            'compiler_admission':False, 'calibration_verified_on_unseen_data':False}


def predict(model, rows):
    if model['schema'] != SCHEMA or model['mode'] not in MODES:
        raise ValueError('unknown curve schema')
    if not 1 <= len(rows) <= 14 or len({r['candidate_id'] for r in rows}) != len(rows) or len({tuple(r['shape']) for r in rows}) != 1:
        raise ValueError('one bounded surface required')
    retained, excluded = [], {}
    for r in sorted(rows, key=lambda r:r['candidate_id']):
        if r['execution_context'] != model['execution_context'] or r['shape'][0] != model['world_size']:
            raise ValueError('execution context mismatch')
        if baseline.old.precision_key(r) not in {tuple(p) for p in model['supported_precision_contracts']}:
            raise ValueError('unsupported precision')
        if 'implementation_contracts' in model and json.dumps([r['candidate_id'],r.get('implementation_id')]) not in model['implementation_contracts']:
            raise ValueError('unseen implementation cannot use archival prior')
        baseline.old.feature_vector(r)
        status = baseline.old.validity_status(r)
        if status != 'passed':
            excluded[r['candidate_id']] = status
        else:
            retained.append(r)
    offsets = baseline.prior_scores(model['prior'], retained) if retained else []
    scores, details, support = {}, {}, {}
    for r, log_prior in zip(retained, offsets):
        base = math.exp(float(log_prior)); key = curve_key(r)
        field = 'latency_us' if model['mode']=='latency' else 'residual_us'
        enabled=model['mode'] in ('latency','prior_residual') or (model['mode'].endswith('fused_residual') and not r['is_host'])
        point = interpolate(model['curves'][key], r['shape'][2], field) if enabled and key in model['curves'] else None
        value = (point['value_us'] + (base if field=='residual_us' else 0.)) if point else base
        local_bracket=interpolate(model['curves'][key],r['shape'][2],'latency_us') if key in model['curves'] else None
        interval_allowed=model.get('interval_support','global_K_range')=='global_K_range' or local_bracket is not None
        interval_value=interval_host_prediction(model['host_interval'],r) if model.get('host_interval') and r['is_host'] and interval_allowed else None
        if interval_value is not None:value=interval_value
        if not math.isfinite(value) or value <= 0:
            raise ValueError('nonpositive corrected latency; do not clip a failed prediction')
        name = r['candidate_id']; scores[name] = math.log(value)
        shapes = model['candidate_support'].get(baseline.support_key(r), [])
        geometric = bool(shapes) and (r['shape'] in shapes or
                    baseline.old.prior_model.in_training_hull(tuple(tuple(s) for s in shapes), tuple(r['shape'])))
        support[name] = {'supported':geometric, 'reason':None if geometric else 'outside_same_candidate_training_support'}
        details[name] = {'prior_us':base, 'prediction_us':value, 'curve_applied':point is not None,
                         'host_interval_applied':interval_value is not None,
                         'host_interval_anchor_k':local_bracket['anchor_k'] if interval_value is not None and local_bracket else None,
                         'curve':point, 'reason':None if point or interval_value is not None else
                         ('no_same_M_N_candidate_precision_K_bracket' if enabled else 'prior_control')}
    provisional = baseline.canonical_choice(scores) if scores else None
    needs = not scores or any(not d['supported'] for d in support.values()) or any(v!='failed' for v in excluded.values())
    return {'predicted_log_us':scores, 'curve_details':details, 'support':support, 'excluded':excluded,
            'provisional_choice':provisional, 'chosen':None if needs else provisional,
            'requires_validation':needs, 'compiler_admission':False,
            'support_is_accuracy_guarantee':False, 'all_measured_valid_candidates_retained':True}


def evaluate(model, rows):
    surfaces = baseline.old.valid_rows(rows, 'evaluation')
    if any(s[0]['shape'] in model['training_shapes'] for s in surfaces):
        raise ValueError('whole-shape leakage')
    decisions = []
    for surface in surfaces:
        p = predict(model, surface); actual = {r['candidate_id']:r['latency_us'] for r in surface}
        chosen = p['provisional_choice']; host = next(r['is_host'] for r in surface if r['candidate_id']==chosen)
        h = min(r['latency_us'] for r in surface if r['is_host']); f = min(r['latency_us'] for r in surface if not r['is_host'])
        errors = [abs(math.exp(p['predicted_log_us'][n])/v-1) for n,v in actual.items()]
        names=sorted(actual); logs=[p['predicted_log_us'][n] for n in names]; truth=[math.log(actual[n]) for n in names]
        def logsoft(v):
            z=[-(x-min(v))/baseline.SETTINGS['temperature'] for x in v]; norm=math.log(math.fsum(math.exp(x) for x in z))
            return [x-norm for x in z]
        lp,lq=logsoft(logs),logsoft(truth)
        decisions.append({'shape':surface[0]['shape'], **p, 'measured_us':actual,
            'provisional_regret':actual[chosen]/min(actual.values())-1,
            'admitted_regret':actual[p['chosen']]/min(actual.values())-1 if p['chosen'] else None,
            'host_error':h<f and not host, 'fused_error':f<h and host,
            'mape':math.fsum(errors)/len(errors), 'max_relative_error':max(errors),
            'cross_entropy':-math.fsum(math.exp(a)*b for a,b in zip(lq,lp)),
            'kl':math.fsum(math.exp(a)*(a-b) for a,b in zip(lq,lp)),
            'order_invariant':p==predict(model,list(reversed(surface)))})
    return decisions
