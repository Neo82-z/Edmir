"""Shared split-K stage controls, preserving host and bounded K corrections.

Source distinguishes q=1 from split-K serial epilogues. Depth ramps are an
empirical alternative to a single K/q slope, not measured per-stage timings.
"""
from __future__ import annotations
from dataclasses import asdict
import math

from . import conditional_curve_selection as curves
from . import additive_latency_selection as additive
from .relative_selection import ScheduleCandidate
from .two_stage_selection import parse_schedule_family

SCHEMA='edmir.split-stage-prior.v1'
VARIANTS=('linear','activation','depth','depth_activation')
COUPLED_VARIANTS=('depth_parallel_offset','depth_parallel','depth_group','depth_parallel_group',
                 'depth_handoff','depth_handoff_group','depth_parallel_handoff','depth_parallel_handoff_group',
                 'depth_blend','depth_blend_group','depth_chain','depth_chain_group',
                 'depth_width','depth_width_chain','depth_footprint','depth_parallel_group_footprint')


def geometry(row):
    if row['is_host']:raise ValueError('split stage features require fused')
    p,m,k,n=row['shape'];q,g=parse_schedule_family(row['candidate_id'])
    nt=(n+128*g-1)//(128*g)
    c=ScheduleCandidate(row['execution_context'],row['candidate_id'],p,m,k,n,
        2*p*(p-1)*m*k,nt,q,g,((m+63)//64)*nt*q)
    c.validate()
    if not additive._domain(c):raise ValueError('outside historical P4 stage geometry')
    return c


def features(row, variant, knots, capacity):
    if variant in COUPLED_VARIANTS:
        from .parallel_depth_selection import features as coupled_features
        return coupled_features(row,variant,knots,capacity)
    if variant not in VARIANTS:raise ValueError('unknown split stage variant')
    if type(capacity) not in (int,float) or not math.isfinite(capacity) or capacity<=0:
        raise ValueError('positive finite capacity required')
    c=geometry(row);original=additive.fused_features(c,capacity)
    depth=c.k/(32*c.split_k_slices)
    if 'depth' in variant:
        if (not 2<=len(knots)<=64 or knots!=sorted(set(knots)) or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in knots)):
            raise ValueError('bounded sorted positive depth knots required')
        if not knots[0]<=depth<=knots[-1]:raise ValueError('unobserved partition depth; no flat extrapolation')
        # Fixed term absorbs the floor at the minimum observed depth. Same
        # slopes score q1/q2/q4..., rather than a separate table per candidate.
        values=(original[0],*original[2:],*(max(0,min(depth-lo,hi-lo))/128 for lo,hi in zip(knots,knots[1:])))
        names=(additive.FUSED_NAMES[0],*additive.FUSED_NAMES[2:],
               *(f'depth_{lo:g}_{hi:g}' for lo,hi in zip(knots,knots[1:])))
    else:
        values,names=original,additive.FUSED_NAMES
    if 'activation' in variant:
        values=(*values,float(c.split_k_slices>1));names=(*names,'split_serial_path_enabled')
    return names,values


def stage_scores(state, rows, original_scores):
    """Return per-candidate log microseconds, leaving every host bit unchanged."""
    if state['schema']!=SCHEMA or len(rows)!=len(original_scores):raise ValueError('stage schema mismatch')
    scores=list(original_scores)
    for i,r in enumerate(rows):
        if r['is_host']:continue
        if r['execution_context']!=state['execution_context']:raise ValueError('stage execution context mismatch')
        names,x=features(r,state['variant'],state['knots'],state['capacity'])
        fit=state['fit']
        if list(names)!=fit['names'] or len(x)!=len(fit['coefficients']):raise ValueError('stage feature schema mismatch')
        additive.LatencyFit(tuple(fit['names']),tuple(fit['coefficients']),
            tuple(fit['scales']),fit['kkt_max_violation']).validate(tuple(names))
        value=1000*math.fsum(c*v for c,v in zip(fit['coefficients'],x))
        if state.get('curvature_transfer') is not None:
            from .curvature_transfer_selection import transferred_value
            value=transferred_value(state,r)
        if not math.isfinite(value) or value<=0:raise ValueError('nonpositive stage cost')
        scores[i]=math.log(value)
    return scores


def fit(rows, prior, variant='depth', *, objective='latency'):
    # The base owns precision, correctness, whole-shape and prior-fold checks.
    base=curves.fit(rows,prior,mode='host_tiles_fused_residual')
    surfaces=curves.baseline.old.valid_rows(rows,'train');fused=[];weights=[]
    for surface in surfaces:
        rr=[r for r in surface if not r['is_host']]
        fused.extend(rr);weights.extend([1/math.sqrt(len(rr))]*len(rr))
    knots=sorted({r['shape'][2]/(32*geometry(r).split_k_slices) for r in fused})
    names=features(fused[0],variant,knots,prior['capacity'])[0]
    raw=[features(r,variant,knots,prior['capacity'])[1] for r in fused]
    if objective=='latency':
        solved=additive._solve(raw,[r['latency_us']/1000 for r in fused],weights,names,prior['ridge'])
    else:
        from .parallel_depth_selection import solve_relative
        solved=solve_relative(fused,raw,weights,names,prior['ridge'],objective)
    state={'schema':SCHEMA,'variant':variant,'knots':knots,'capacity':prior['capacity'],
           'ridge':prior['ridge'],'fit':asdict(solved),'execution_context':base['execution_context'],
           **({'training_objective':objective} if objective!='latency' else {}),
           'training_shapes':base['training_shapes'],'training_sources':base['training_sources'],
           'coefficient_meaning':'empirical full-program costs, not measured semaphore or communication time'}
    # Feature-name comparison must agree before and after JSON roundtrip.
    state['fit']['names']=list(state['fit']['names'])
    base['stage_prior']=state
    new_scores=stage_scores(state,fused,[0.]*len(fused))
    for r,score in zip(fused,new_scores):
        knot=next(v for v in base['curves'][curves.curve_key(r)] if v['k']==r['shape'][2])
        knot['residual_us']=r['latency_us']-math.exp(score)
    base['optimizer']='NNLS; fixed training-fold ridge and depth knots; no optimizer search'
    return base


predict=curves.predict
evaluate=curves.evaluate
