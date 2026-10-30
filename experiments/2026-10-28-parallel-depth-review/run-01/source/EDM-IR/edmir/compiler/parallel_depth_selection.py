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
    return tuple(names),tuple(x)


def fit(rows, prior, variant='depth_parallel_group'):
    if variant not in VARIANTS:raise ValueError('unknown parallel depth variant')
    return stage.fit(rows,prior,variant)


predict=stage.predict
evaluate=stage.evaluate
