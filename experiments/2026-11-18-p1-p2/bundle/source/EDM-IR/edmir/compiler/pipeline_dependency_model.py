"""Source-derived SM80 pipeline control with explicitly conditional services.

The fixed kernel has three stages and two warp MMA k-groups per K32 tile.
This is an aggregated CTA dependency graph, NOT a cycle-accurate GPU model:
per-warp arrival, instruction scheduling and shared-resource service are unknown.
No timing labels, fitting, candidate ranking or compiler admission are provided.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .dependency_wave_model import KernelCase, positive_int

MAX_DEPTH = 4096
MAX_NODES = 60_000


def profile(case: KernelCase):
    case.validate()
    positive_int('bounded pipeline depth', case.depth, MAX_DEPTH)
    warps = 4 * case.group_n
    threads = 32 * warps
    a, b = 2 * 64 * 32, 2 * 32 * 128 * case.group_n
    # Fixed full-tile, 16-byte access, row/row WarpRakedThreadMap specialization.
    ia, ib = a // (threads * 16), b // (threads * 16)
    d = case.depth
    return {
        'scope': 'source template counts per CTA; not compiled instruction counts',
        'depth': d, 'stages': 3, 'warp_k_groups_per_K32': 2,
        'warps': warps, 'threads': threads, 'cp_async_cache_policy': 'Global (.cg)',
        'A_copy_loop_iterations_per_thread_per_stage': ia,
        'B_copy_loop_iterations_per_thread_per_stage': ib,
        'full_tile_A_bytes_per_stage': a, 'full_tile_B_bytes_per_stage': b,
        'prologue_groups': 2, 'valid_K_groups': d, 'masked_K_groups': 2,
        'mainloop_iterations': d, 'warp_mma_calls': 2 * d,
        'mma_instructions_per_warp_per_K32': 32,
        'source_cp_async_fence_calls': d + 3,
        'PTX_wait_all_implicit_empty_commit': 1,
        'source_wait_group_1_calls': d + 1, 'source_final_drain_calls': 1,
        'source_barrier_calls_in_mma': d + 2,
        'full_tile_valid_copy_instruction_sites_per_thread': d * (ia + ib),
        'masked_copy_instruction_sites_per_thread': 2 * (ia + ib),
        'gemm_masked_group_sites': 2 * case.ctas,
        'gemm_mainloop_barrier_sites': (d + 2) * case.ctas,
        'tail_predication_also_required': case.m % 64 != 0 or case.n % (128 * case.group_n) != 0,
        'actual_registers': None, 'actual_occupancy': None,
        'physical_copy_bytes': None, 'prediction_us': None,
    }


@dataclass(frozen=True)
class PipelineService:
    # Aggregated CTA issue/compute durations are scenario inputs. A/B latency
    # begins after that operand's issue node; it is not derived from old labels.
    issue_A: float
    issue_B: float
    latency_A: float
    latency_B: float
    mma_half: float
    barrier: float
    masked_issue_fraction: float
    unit: str
    assumption: str

    def validate(self):
        for value in (self.issue_A, self.issue_B, self.latency_A, self.latency_B,
                      self.mma_half, self.barrier):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e9:
                raise ValueError('bounded finite nonnegative conditional service required')
        if (type(self.masked_issue_fraction) not in (int, float)
                or not math.isfinite(self.masked_issue_fraction)
                or not 0 <= self.masked_issue_fraction <= 1):
            raise ValueError('explicit masked issue fraction in [0,1] required')
        if not isinstance(self.unit, str) or not self.unit.strip() or not isinstance(self.assumption, str) or not self.assumption.strip():
            raise ValueError('explicit unit and service assumption required')


class DependencyGraph:
    """Append-only DAG. O(V+E) evaluation; every dependency names a prior node."""
    def __init__(self):
        self.nodes = []

    def add(self, name, kind, duration, parents=()):
        if len(self.nodes) >= MAX_NODES:
            raise ValueError('pipeline graph node budget exceeded')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid node duration')
        if any(type(p) is not int or not 0 <= p < len(self.nodes) for p in parents):
            raise ValueError('dependency must already exist')
        start = max((self.nodes[p]['end'] for p in parents), default=0.)
        self.nodes.append({'name': name, 'kind': kind, 'duration': duration,
                           'parents': tuple(parents), 'start': start, 'end': start + duration})
        return len(self.nodes) - 1


def simulate_cta(case: KernelCase, service: PipelineService, *, trace=False, max_seconds=2.):
    """Three-stage source dependency, with aggregate per-stage service inputs.

    Initial wait_group<1> waits for group 0. Each loop commits i+2 then
    waits for every group through i+1, before its second warp MMA k-group.
    A prefix-ready chain keeps all older-group constraints without O(D²)
    edges. Masked copies add issue/control sites, never valid global bytes.
    """
    p = profile(case); service.validate()
    if type(trace) is not bool or type(max_seconds) not in (int, float) or not 0 < max_seconds <= 10:
        raise ValueError('bounded trace and CPU deadline required')
    started = time.monotonic(); graph = DependencyGraph(); prefixes = []
    cursor = graph.add('entry', 'entry', 0.)
    initial_wait = loop_wait = 0.

    def issue(index, *, prologue=False):
        nonlocal cursor
        valid = index < case.depth
        factor = 1 if valid else service.masked_issue_fraction
        ia = p['A_copy_loop_iterations_per_thread_per_stage']
        ib = p['B_copy_loop_iterations_per_thread_per_stage']
        first_a, first_b = (ia + 1) // 2, (ib + 1) // 2
        # Prologue copies A then B; mainloop calls copy_tiles_and_advance
        # twice, preserving A/B subgroup order rather than merging all A first.
        chunks = [(ia, ib)] if prologue else [(first_a, first_b), (ia - first_a, ib - first_b)]
        ready = []
        for chunk, (na, nb) in enumerate(chunks):
            for operand, count, denominator, duration, latency in (
                    ('A', na, ia, service.issue_A, service.latency_A),
                    ('B', nb, ib, service.issue_B, service.latency_B)):
                if not count:
                    continue
                cursor = graph.add(f'{operand}_issue_{index}_{chunk}', 'issue', duration * count / denominator * factor, (cursor,))
                ready.append(graph.add(f'{operand}_ready_{index}_{chunk}', 'transfer', latency if valid else 0., (cursor,)))
        # Commit itself is a zero-duration ordering edge; its cost must be
        # included in the explicit issue scenario, not silently calibrated.
        cursor = graph.add(f'commit_{index}', 'commit', 0., (cursor,))
        prefix = graph.add(f'prefix_ready_{index}', 'ready', 0.,
                           tuple(ready) + (cursor,) + ((prefixes[-1],) if prefixes else ()))
        prefixes.append(prefix)

    def wait(index, label):
        nonlocal cursor
        before = graph.nodes[cursor]['end']
        cursor = graph.add(label, 'barrier', service.barrier, (cursor, prefixes[index]))
        return graph.nodes[cursor]['start'] - before

    issue(0, prologue=True); issue(1, prologue=True)
    initial_wait = wait(0, 'initial_wait_group_1')
    for i in range(case.depth):
        cursor = graph.add(f'mma_first_{i}', 'compute', service.mma_half, (cursor,))
        issue(i + 2)
        loop_wait += wait(i + 1, f'wait_group_1_{i}')
        cursor = graph.add(f'mma_second_{i}', 'compute', service.mma_half, (cursor,))
        if i % 128 == 0 and time.monotonic() - started > max_seconds:
            raise TimeoutError('pipeline graph CPU deadline')
    cursor = graph.add('final_empty_commit', 'commit', 0., (cursor,))
    cursor = graph.add('wait_all_implicit_empty_commit', 'implicit_commit', 0., (cursor,))
    drain_wait = wait(len(prefixes) - 1, 'final_wait_all')
    total = graph.nodes[cursor]['end']
    sequential = sum(n['duration'] for n in graph.nodes if n['kind'] in ('issue', 'barrier', 'compute'))
    if not math.isclose(total, sequential + initial_wait + loop_wait + drain_wait, rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError('exposed wait decomposition failed')
    if sum(n['kind'] == 'commit' for n in graph.nodes) != case.depth + 3:
        raise AssertionError('commit-count identity failed')
    result = {'profile': p, 'unit': service.unit, 'scenario_cta_time': total,
              'sequential_issue_compute_barrier_time': sequential,
              'initial_exposed_wait': initial_wait, 'loop_exposed_wait': loop_wait,
              'final_exposed_wait': drain_wait, 'node_count': len(graph.nodes),
              'edge_count': sum(len(n['parents']) for n in graph.nodes),
              'cpu_seconds': time.monotonic() - started, 'service': service.__dict__,
              'prediction_us': None, 'training_eligible': False, 'compiler_admission': False,
              'hardware_scheduling_verified': False,
              'unmodeled': ['individual warp issue and arrival', 'per-SM instruction throughput sharing',
                            'actual cp.async transaction completion', 'CTA admission and cache state',
                            'peer/source contention', 'whole-program CPU submission and synchronization']}
    if trace:
        result['nodes'] = graph.nodes
    return result
