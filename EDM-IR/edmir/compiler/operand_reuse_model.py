"""Conditional tensor-panel LRU replay, never a calibrated A100 simulation.

One destination executes P per-source GEMMs on its compute stream and reuses
one weight buffer. Panels represent valid tensor elements, not cache lines or
physical address mappings. CTA-complete reference orders are scenarios only.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time

from .dependency_wave_model import KernelCase, positive_int

MAX_REFERENCES=1_000_000
MAX_RESIDENT_PANELS=100_000


@dataclass(frozen=True)
class ReuseScenario:
    capacity_bytes: int
    tile_order: str
    remote_A_cached: bool
    share_weights: bool
    iterations: int

    def validate(self):
        if type(self.capacity_bytes) is not int or not 0<=self.capacity_bytes<=1<<30:
            raise ValueError('bounded cache capacity required')
        if self.tile_order not in ('m_fast','n_fast'):
            raise ValueError('explicit logical tile visit order required')
        if type(self.remote_A_cached) is not bool or type(self.share_weights) is not bool:
            raise ValueError('explicit cache placement and buffer sharing required')
        positive_int('iterations',self.iterations,4)


def empty_stats():
    return {name:{k:0 for k in ('references','read_bytes','write_bytes','read_hit_bytes',
            'read_miss_bytes','write_hit_bytes','allocated_bytes','dirty_eviction_bytes','bypass_write_bytes')}
            for name in ('A_local','A_remote','B','C')}


class PanelCache:
    """Byte-capacity LRU with write allocation; no read on full-panel store miss."""
    def __init__(self,capacity):
        if type(capacity) is not int or not 0<=capacity<=1<<30:
            raise ValueError('bounded integer capacity required')
        self.capacity=capacity;self.used=0;self.peak_used=0;self.peak_panels=0
        self.items=OrderedDict();self.stats=empty_stats()

    def access(self,key,size,kind,*,write=False,cacheable=True):
        positive_int('panel size',size,1<<30)
        if kind not in self.stats or type(write) is not bool or type(cacheable) is not bool:
            raise ValueError('invalid panel access')
        s=self.stats[kind];s['references']+=1;s['write_bytes' if write else 'read_bytes']+=size
        found=self.items.get(key)
        if found is not None and (found[0]!=size or found[1]!=kind):
            raise ValueError('same panel key changed extent or ownership')
        if found is not None and not cacheable:
            raise ValueError('cache placement changed within replay')
        if found is not None:
            self.items.move_to_end(key)
            s['write_hit_bytes' if write else 'read_hit_bytes']+=size
            if write:self.items[key]=(size,kind,True)
            return
        if not write:s['read_miss_bytes']+=size
        if not cacheable or size>self.capacity:
            if write:s['bypass_write_bytes']+=size
            return
        while self.used+size>self.capacity:
            _,(n,owner,dirty)=self.items.popitem(last=False);self.used-=n
            if dirty:self.stats[owner]['dirty_eviction_bytes']+=n
        if len(self.items)>=MAX_RESIDENT_PANELS:
            raise ValueError('bounded resident panel limit exceeded')
        self.items[key]=(size,kind,write);self.used+=size;s['allocated_bytes']+=size
        self.peak_used=max(self.peak_used,self.used);self.peak_panels=max(self.peak_panels,len(self.items))


def reference_count(case,iterations):
    case.validate();positive_int('iterations',iterations,4)
    # Every aligned K32 panel is visited once for each tile and source.
    return iterations*case.world_size*(2*case.chains*(case.k//32)+case.chains*(2*case.q-1))


def program_footprint(case,*,remote_A_cached,share_weights):
    case.validate()
    if type(remote_A_cached) is not bool or type(share_weights) is not bool:
        raise ValueError('explicit placement and sharing required')
    a=2*case.m*case.k;b=2*case.k*case.n*(1 if share_weights else case.world_size);c=2*case.m*case.n
    return {'addressed_input_bytes':case.world_size*a,'unique_weight_bytes':b,
            'output_bytes':case.world_size*c,
            'conditional_cacheable_bytes':(case.world_size if remote_A_cached else 1)*a+b+case.world_size*c,
            'unconditional_physical_cache_footprint_bytes':None}


def replay(case: KernelCase,scenario: ReuseScenario,*,max_seconds=2.):
    case.validate();scenario.validate()
    if type(max_seconds) not in (int,float) or not 0<max_seconds<=10:
        raise ValueError('bounded CPU timeout required')
    expected=reference_count(case,scenario.iterations)
    if expected>MAX_REFERENCES:raise ValueError('reference budget exceeded before traversal')
    start=time.monotonic();cache=PanelCache(scenario.capacity_bytes)
    mt=(case.m+63)//64;nt=(case.n+128*case.group_n-1)//(128*case.group_n)
    tiles=([(m,n) for n in range(nt) for m in range(mt)] if scenario.tile_order=='m_fast'
           else [(m,n) for m in range(mt) for n in range(nt)])
    passes=[];references=0
    for iteration in range(scenario.iterations):
        cache.stats=empty_stats()
        for source in range(case.world_size):
            for part in range(case.q):
                for m,n in tiles:
                    valid_m=min(64,case.m-m*64);valid_n=min(128*case.group_n,case.n-n*128*case.group_n)
                    for ki in range(part*case.depth,(part+1)*case.depth):
                        local=source==0
                        cache.access(('A',source,m,ki),2*valid_m*32,'A_local' if local else 'A_remote',
                                     cacheable=local or scenario.remote_A_cached)
                        cache.access(('B',0 if scenario.share_weights else source,ki,n),2*32*valid_n,'B')
                        references+=2
                        if references%1024==0 and time.monotonic()-start>max_seconds:
                            raise TimeoutError('operand replay CPU deadline')
                    output=('C',source,m,n);size=2*valid_m*valid_n
                    if part:
                        cache.access(output,size,'C');references+=1
                    cache.access(output,size,'C',write=True);references+=1
        totals=cache.stats
        # Work identity is independent of cache capacity, order and residency.
        counts=case.counts()
        expected_reads={'A_local':counts['logical_A_load_bytes_before_cache_reuse'],
                        'A_remote':(case.world_size-1)*counts['logical_A_load_bytes_before_cache_reuse'],
                        'B':case.world_size*counts['logical_B_load_bytes_before_cache_reuse'],
                        'C':case.world_size*counts['logical_extra_output_read_bytes']}
        for kind,amount in expected_reads.items():
            s=totals[kind]
            if s['read_bytes']!=amount or s['read_bytes']!=s['read_hit_bytes']+s['read_miss_bytes']:
                raise AssertionError('read-work conservation failed')
        if totals['C']['write_bytes']!=case.world_size*counts['logical_output_write_bytes']:
            raise AssertionError('output-work conservation failed')
        if cache.used>scenario.capacity_bytes:raise AssertionError('cache capacity exceeded')
        passes.append({'iteration':iteration,'initial_state':'cold' if iteration==0 else 'previous_iteration_residency',
                       'operands':totals,'resident_bytes_after':cache.used,
                       'dirty_bytes_after':sum(n for n,owner,dirty in cache.items.values() if dirty)})
    if references!=expected:raise AssertionError('reference-count identity failed')
    return {'scope':'one destination reference stream; no execution timing or all-GPU contention',
            'shape_P_M_K_N':[case.world_size,case.m,case.k,case.n],'q':case.q,'group_n':case.group_n,
            'scenario':scenario.__dict__,'footprint':program_footprint(case,remote_A_cached=scenario.remote_A_cached,share_weights=scenario.share_weights),
            'passes':passes,'references':references,'peak_resident_bytes':cache.peak_used,
            'peak_resident_panels':cache.peak_panels,'cpu_seconds':time.monotonic()-start,
            'physical_traffic_verified':False,'prediction_us':None,'chosen':None,
            'training_eligible':False,'compiler_admission':False,
            'unmodeled':['sector/set mapping and associativity','actual CTA admission/interleaving',
                        'cp.async pipeline timing','GPU cache placement for peer reads','data compression',
                        'source-GPU contention','semaphore polling','host costs','prefetch and persistence settings']}
