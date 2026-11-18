"""Preserve the successful depth curvature without drifting other costs.

The depth curve is estimated on the same training fold. Other coefficients
remain those of that fold's original prior. An optional single intercept is
calibrated from training times; no query labels or per-shape adjustments.
"""
from __future__ import annotations
import math
from . import split_stage_selection as stage

SCHEMA='edmir.curvature-transfer.v1'
MODES=('fixed','intercept')


def transferred_value(state,row):
    transfer=state['curvature_transfer']
    if state['variant']!='depth' or transfer['schema']!=SCHEMA or transfer['mode'] not in MODES:
        raise ValueError('unknown curvature transfer state')
    if transfer['mode']=='intercept':
        certificate=transfer.get('intercept_kkt_violation',0.)
        if type(certificate) not in (int,float) or not math.isfinite(certificate) or not 0<=certificate<=1e-9:
            raise ValueError('invalid intercept certificate')
    names,x=stage.features(row,'depth',state['knots'],state['capacity'])
    coefficients=dict(zip(state['fit']['names'],state['fit']['coefficients']))
    frozen=transfer['non_depth_coefficients']
    expected=set(stage.additive.FUSED_NAMES)-{'partition_K','fixed'}
    if set(frozen)!=expected:raise ValueError('frozen term schema mismatch')
    values=[*frozen.values(),transfer['intercept_ms']]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('invalid frozen costs')
    curve=math.fsum(coefficients[n]*v for n,v in zip(names,x) if n.startswith('depth_'))
    old=dict(zip(stage.additive.FUSED_NAMES,stage.additive.fused_features(stage.geometry(row),state['capacity'])))
    return 1000*(transfer['intercept_ms']+curve+math.fsum(frozen[n]*old[n] for n in frozen))


def fit(rows,prior,mode='intercept'):
    if mode not in MODES:raise ValueError('unknown transfer mode')
    base=stage.fit(rows,prior,'depth');state=base['stage_prior']
    co=dict(zip(prior['fused']['names'],prior['fused']['coefficients']))
    transfer={'schema':SCHEMA,'mode':mode,'intercept_ms':co['fixed'],
        'non_depth_coefficients':{n:v for n,v in co.items() if n not in ('partition_K','fixed')},
        'calibration':'original training-fold fixed term'}
    state['curvature_transfer']=transfer
    surfaces=stage.curves.baseline.old.valid_rows(rows,'train')
    fused=[];weighted=[]
    for surface in surfaces:
        rr=[r for r in surface if not r['is_host']];fused.extend(rr)
        for r in rr:
            rest=transferred_value(state,r)/1000-transfer['intercept_ms']
            weighted.append((1/len(rr),r['latency_us']/1000-rest))
    if mode=='intercept':
        denom=math.fsum(w for w,y in weighted)
        optimal=max(0.,math.fsum(w*y for w,y in weighted)/denom)
        gradient=math.fsum(w*(optimal-y) for w,y in weighted)
        violation=abs(gradient) if optimal>0 else max(0.,-gradient)
        if violation>1e-9:raise ValueError('intercept certificate failed')
        transfer.update(intercept_ms=optimal,calibration='one shape-balanced nonnegative least-squares intercept; training labels only',
                        intercept_kkt_violation=violation)
    for r in fused:
        value=transferred_value(state,r)
        if not math.isfinite(value) or value<=0:raise ValueError('invalid transferred time')
        knot=next(k for k in base['curves'][stage.curves.curve_key(r)] if k['k']==r['shape'][2])
        knot['residual_us']=r['latency_us']-value
    base['optimizer']='same-fold depth NNLS; original other coefficients fixed; optional exact one-dimensional intercept fit'
    base['prediction_semantics']='empirical depth-curve transfer; component costs are not identified physical times'
    return base


predict=stage.predict
evaluate=stage.evaluate
