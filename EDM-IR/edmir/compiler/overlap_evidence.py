"""Decision-margin and overlap evidence; never infer speedup from concurrency.

Public traces can validate interval accounting. They cannot calibrate a different
GPU, kernel, precision, or communication path. No fitted coefficients live here.
"""
from __future__ import annotations

from collections import defaultdict
import math


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive latency required')
    return float(value)


def decision_margin(host_us, fused_us, predicted_host_us, predicted_fused_us):
    """Positive log gap means fused is faster. Observed labels are diagnostic only."""
    h, f, ph, pf = map(_positive, (host_us, fused_us, predicted_host_us, predicted_fused_us))
    truth, predicted = math.log(h)-math.log(f), math.log(ph)-math.log(pf)
    eh, ef = math.log(ph)-math.log(h), math.log(pf)-math.log(f)
    return {'measured_log_host_over_fused': truth, 'predicted_log_host_over_fused': predicted,
        'host_log_error': eh, 'fused_log_error': ef, 'differential_log_error': eh-ef,
        'common_log_error': (eh+ef)/2, 'direction_wrong': truth*predicted < 0,
        'predicted_tie': predicted == 0, 'measured_tie': truth == 0,
        'measured_host_minus_fused_us': h-f, 'predicted_host_minus_fused_us': ph-pf,
        'differential_correction_to_measured_us': (h-f)-(ph-pf),
        'mechanism_cause': None, 'diagnostic_not_a_training_label': True}


def union_intervals(intervals):
    if len(intervals) > 200_000:
        raise ValueError('interval budget exceeded')
    checked = []
    for a, b in intervals:
        if not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) or b < a:
            raise ValueError('finite ordered interval required')
        if b > a:
            checked.append((a, b))
    result = []
    for a, b in sorted(checked):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def kernel_overlap(compute_intervals, communication_intervals):
    """Union/intersection of GPU kernel wall intervals, not SM utilization."""
    c, r = union_intervals(compute_intervals), union_intervals(communication_intervals)
    i = j = 0
    overlap = 0
    while i < len(c) and j < len(r):
        overlap += max(0, min(c[i][1], r[j][1])-max(c[i][0], r[j][0]))
        if c[i][1] < r[j][1]:
            i += 1
        else:
            j += 1
    tc = sum(b-a for a, b in c)
    tr = sum(b-a for a, b in r)
    return {'compute_kernel_union_us': tc, 'communication_kernel_union_us': tr,
        'kernel_intersection_us': overlap, 'kernel_union_us': tc+tr-overlap,
        'communication_kernel_concurrency_fraction': overlap/tr if tr else None,
        'nonoverlapping_communication_kernel_us': tr-overlap,
        'hidden_network_time_us': None, 'overlap_speedup': None, 'sm_occupancy': None}


def analyze_deepseek_trace(trace):
    """Explicit classification for the pinned V3/R1 trace, not a generic parser.

    The trace's first-to-last kernel window includes collection boundaries and
    barriers. Report interval accounting only; no per-request latency or causal
    saved time. Device/context clocks are never merged across GPUs.
    """
    events = trace['traceEvents']
    if len(events) > 200_000:
        raise ValueError('trace budget exceeded')
    groups = defaultdict(lambda: {'compute': [], 'communication': [], 'excluded_nccl': [], 'names': defaultdict(int)})
    for e in events:
        if e.get('cat') != 'kernel' or e.get('ph') != 'X':
            continue
        args = e['args']
        key = (args['device'], args.get('context'))
        name = e['name']
        if 'dpsk::ep::internode::' in name:
            role = 'communication'
        elif 'nccl' in name.lower():
            role = 'excluded_nccl'
        else:
            role = 'compute'
        groups[key][role].append((e['ts'], e['ts']+e['dur']))
        groups[key]['names'][(role, name)] += 1
    output = []
    for (device, context), g in sorted(groups.items(), key=lambda item: repr(item[0])):
        output.append({'device': device, 'context': context, **kernel_overlap(g['compute'], g['communication']),
            'event_counts': {k: len(g[k]) for k in ('compute', 'communication', 'excluded_nccl')},
            'classification': [{'role': role, 'kernel': name, 'count': count} for (role, name), count in sorted(g['names'].items())],
            'scope': 'all non-NCCL non-EP-internode kernels count as compute/auxiliary',
            'not_observed': ['RDMA progress after issuing kernel returns', 'isolated service times',
                             'resource contention slowdown', 'critical path exposure', 'SM occupancy']})
    return {'trace_event_count': len(events), 'devices': trace.get('deviceProperties'), 'distributed': trace.get('distributedInfo'),
            'groups': output, 'benchmark_identity': 'DeepSeek profile-data V3/R1 4496024',
            'transferable_a100_timing_labels': False}


CONTEXT_FIELDS = ('hardware', 'topology', 'world_size', 'kernel_revision', 'precision',
                  'communication_path', 'resource_partition', 'shape', 'timing_scope')


def paired_overlap_cost(record):
    """Measured accounting for the same two independently runnable stages.

    Require matching work/implementations/context in isolated and concurrent
    runs. This is inapplicable to opaque fused kernels without stage evidence.
    The stage intervals must cover completion of their work, including any
    asynchronous transfers. A GPU issuing-kernel interval alone is insufficient.
    Overlap benefit = interval intersection - service inflation - extra gaps.
    This identity is not an extrapolation model or a fitted overlap coefficient.
    """
    for mode in ('isolated', 'concurrent'):
        if set(CONTEXT_FIELDS)-set(record[mode]['context']):
            raise ValueError('incomplete execution context')
        if any(record[mode]['context'][k] is None for k in CONTEXT_FIELDS):
            raise ValueError('unknown context cannot calibrate a service')
        if not record[mode].get('source_id'):
            raise ValueError('measurement provenance required')
    if record['isolated']['context'] != record['concurrent']['context']:
        raise ValueError('isolated/concurrent execution context mismatch')
    if record.get('equivalent_work_verified') is not True:
        raise ValueError('equivalent work must be verified')
    if record.get('stage_completion_verified') is not True:
        raise ValueError('stage intervals must cover asynchronous work completion')
    solo = record['isolated']
    joint = record['concurrent']
    c0, r0, c1, r1 = map(_positive, (solo['compute_us'], solo['communication_us'], joint['compute_us'], joint['communication_us']))
    overlap = joint['stage_intersection_us']
    gap0, gap1 = solo['other_us'], joint['other_us']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (overlap, gap0, gap1)):
        raise ValueError('finite nonnegative intervals required')
    if overlap > min(c1, r1):
        raise ValueError('intersection exceeds a stage duration')
    t0, t1 = c0+r0+gap0, c1+r1-overlap+gap1
    inflation = (c1-c0)+(r1-r0)
    return {'sequential_us': t0, 'concurrent_us': t1, 'saved_us': t0-t1,
        'stage_intersection_us': overlap, 'service_inflation_us': inflation,
        'extra_other_us': gap1-gap0, 'overlap_is_beneficial': t1 < t0,
        'source_ids': [solo['source_id'], joint['source_id']],
        'accounting_identity_only': True, 'compiler_admission': False}
