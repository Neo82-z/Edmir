"""Shared low-parallelism/depth interactions; retrospective P4/FP16 only.

The rational concurrency coordinate is an empirical basis, not measured
occupancy or a queueing identity. No shape-specific gates or candidate labels.
"""
from __future__ import annotations
from . import split_stage_selection as stage

VARIANTS=('depth',*stage.COUPLED_VARIANTS)


def features(row, variant, knots, capacity):
    if variant not in stage.COUPLED_VARIANTS:raise ValueError('unknown parallel depth variant')
    names,x=stage.features(row,'depth',knots,capacity)
    names,x=list(names),list(x)
    c=stage.geometry(row)
    ramps=[(n,v) for n,v in zip(names,x) if n.startswith('depth_')]
    if 'blend' in variant or 'chain' in variant:
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


def fit(rows, prior, variant='depth_blend_group'):
    if variant not in VARIANTS:raise ValueError('unknown parallel depth variant')
    return stage.fit(rows,prior,variant)


predict=stage.predict
evaluate=stage.evaluate
