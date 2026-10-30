"""Shared low-parallelism/depth interactions; retrospective P4/FP16 only.

The rational concurrency coordinate is an empirical basis, not measured
occupancy or a queueing identity. No shape-specific gates or candidate labels.
"""
from __future__ import annotations
import math
from . import split_stage_selection as stage

VARIANTS=('depth',*stage.COUPLED_VARIANTS)


def features(row, variant, knots, capacity):
    if variant not in stage.COUPLED_VARIANTS:raise ValueError('unknown parallel depth variant')
    names,x=stage.features(row,'depth',knots,capacity)
    names,x=list(names),list(x)
    c=stage.geometry(row)
    ramps=[(n,v) for n,v in zip(names,x) if n.startswith('depth_')]
    if 'width' in variant:
        # g=1/g=2 instantiate different CTA widths/warp layouts. Independent
        # nonnegative curves remove the assumption that the wide kernel's
        # depth slopes can only exceed the narrow kernel's slopes.
        if c.n_tiles_per_cta not in (1,2):raise ValueError('unseen CTA width')
        kept=[(n,v) for n,v in zip(names,x) if not n.startswith('depth_') and n!='group_partition_K']
        names=[n for n,v in kept];x=[v for n,v in kept]
        g=c.n_tiles_per_cta
        weights=[('shared_',1.)]
        if 'chain' in variant:
            chains=c.estimated_ctas_per_op/c.split_k_slices
            u=capacity/(capacity+chains);weights=[('low_',u),('filled_',1-u)]
        for label,w in (('narrow_',2-g),('wide_',g-1)):
            for parallel,pw in weights:
                for n,v in ramps:
                    names.append(label+parallel+n);x.append(w*pw*v)
    elif 'blend' in variant or 'chain' in variant:
        # A low-concurrency curve and a filled-concurrency curve are mixed
        # with convex weights. Unlike an additive nonnegative u*r correction,
        # this does not force per-depth service to decrease with concurrency.
        concurrency=(c.estimated_ctas_per_op/c.split_k_slices if 'chain' in variant else c.estimated_ctas_per_op)
        u=capacity/(capacity+concurrency)
        kept=[(n,v) for n,v in zip(names,x) if not n.startswith('depth_')]
        names=[n for n,v in kept];x=[v for n,v in kept]
        for prefix,weight in (('low_',u),('filled_',1-u)):
            for n,v in ramps:
                names.append(prefix+n);x.append(weight*v)
    if variant.endswith('_group'):
        # Wider tile mainloops need not have a globally linear depth surcharge.
        # Keep group_fixed for the minimum-depth floor and all other controls.
        i=names.index('group_partition_K');names.pop(i);x.pop(i)
        for n,v in ramps:
            names.append('group_'+n);x.append((c.n_tiles_per_cta-1)*v)
    if 'parallel' in variant:
        # C is the unchanged fitted pressure scale, NOT the number of SMs.
        # This varies below C, unlike the old max(B/C-1,0) pressure feature.
        u=capacity/(capacity+c.estimated_ctas_per_op)
        names.append('low_parallelism');x.append(u)
        if variant!='depth_parallel_offset':
            for n,v in ramps:
                names.append('low_parallelism_'+n);x.append(u*v)
    if 'handoff' in variant:
        # Empirical exposure coordinate for the existing q-1 ordered edges.
        # Two prologue stages come from the pinned SM80 template. This is not
        # a measured wait time or a claim that all edges are on the critical path.
        d=c.k/(32*c.split_k_slices)
        u=capacity/(capacity+c.estimated_ctas_per_op)
        names.append('short_serial_exposure')
        x.append((c.split_k_slices-1)*u*2/(d+2))
    return tuple(names),tuple(x)


def solve_relative(rows,raw,weights,names,ridge,objective):
    """Convex controls: relative latency, optionally matched candidate gaps.

    All pairs are measured fused training candidates. Fixed gap floor uses the
    existing evaluation temperature; no near-tie exclusion or winner bonus.
    """
    if objective not in ('relative','relative_pair'):raise ValueError('unknown training objective')
    import numpy as np
    from scipy.optimize import nnls
    x=np.asarray(raw,dtype=float);y=np.asarray([r['latency_us']/1000 for r in rows])
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-9)
    w=np.asarray(weights)/y
    aa=[x/scale*w[:,None]];bb=[y*w]
    if objective=='relative_pair':
        groups={}
        for i,r in enumerate(rows):groups.setdefault(tuple(r['shape']),[]).append(i)
        ax=[];by=[]
        for ids in groups.values():
            pairs=[(i,j) for ii,i in enumerate(ids) for j in ids[ii+1:]]
            floor=stage.curves.baseline.SETTINGS['temperature']*min(y[i] for i in ids)
            for i,j in pairs:
                den=max(abs(y[i]-y[j]),floor)*math.sqrt(len(pairs))
                ax.append((x[i]-x[j])/scale/den);by.append((y[i]-y[j])/den)
        if ax:aa.append(np.asarray(ax));bb.append(np.asarray(by))
    penalty=np.eye(len(names))*math.sqrt(ridge);penalty[0,0]=0.
    a=np.vstack((*aa,penalty));b=np.r_[*bb,np.zeros(len(names))]
    coef=nnls(a,b,maxiter=1000)[0];gradient=a.T@(a@coef-b)
    violation=max(float(np.maximum(-gradient,0).max()),
        float(np.abs(gradient[coef>1e-10]).max()) if any(coef>1e-10) else 0.)
    result=stage.additive.LatencyFit(tuple(names),tuple(float(v) for v in coef/scale),
        tuple(float(v) for v in scale),violation)
    result.validate(tuple(names));return result


def fit(rows, prior, variant, *, objective='latency'):
    # No implicit default: none of these revisions passed misselection admission.
    if variant not in VARIANTS:raise ValueError('unknown parallel depth variant')
    return stage.fit(rows,prior,variant,objective=objective)


predict=stage.predict
evaluate=stage.evaluate
