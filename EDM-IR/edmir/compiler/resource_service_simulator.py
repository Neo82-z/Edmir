"""Bounded conditional fluid simulation of ONE per-source split-K GEMM.

Compute deadlines and memory/link service overlap within a mainloop. Active
transfers share each resource equally; epilogues contend with mainloop memory
traffic. Waiting CTAs retain their slot. These are explicit scenario assumptions,
not CUDA scheduling, cache behavior, or calibrated A100 latency predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .dependency_wave_model import KernelCase, ADMISSION_ORDERS, positive_int

MAX_CTAS = 512


@dataclass(frozen=True)
class ResourceScenario:
    compute_fixed: float
    compute_per_iteration: float
    first_epilogue: float
    later_epilogue: float
    memory_bytes_per_unit: float
    link_bytes_per_unit: float
    a_delivery_fraction: float
    b_delivery_fraction: float
    unit: str
    assumption: str

    def validate(self):
        for x in (self.compute_fixed, self.compute_per_iteration,
                  self.first_epilogue, self.later_epilogue):
            if type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1e12:
                raise ValueError('finite nonnegative synthetic service required')
        for x in (self.memory_bytes_per_unit,self.link_bytes_per_unit):
            if type(x) not in (int,float) or not math.isfinite(x) or not 0 < x <= 1e18:
                raise ValueError('positive bounded bandwidth required')
        for x in (self.a_delivery_fraction,self.b_delivery_fraction):
            if type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError('delivery fraction must be explicit and within [0,1]')
        if not self.unit or not isinstance(self.unit,str) or not self.assumption or not isinstance(self.assumption,str):
            raise ValueError('explicit units and scenario assumptions required')


def simulate(case: KernelCase, scenario: ResourceScenario, *, slots, remote_source, order='partition_major', max_seconds=2.):
    case.validate(); scenario.validate(); positive_int('slots',slots,32768)
    if case.ctas > MAX_CTAS or type(remote_source) is not bool or order not in ADMISSION_ORDERS:
        raise ValueError('unsupported or unbounded simulation')
    if type(max_seconds) not in (int,float) or not 0 < max_seconds <= 10:
        raise ValueError('explicit bounded CPU deadline required')
    wall_start=time.monotonic()
    tasks = ([(t,s) for s in range(case.q) for t in range(case.chains)] if order=='partition_major'
             else [(t,s) for t in range(case.chains) for s in range(case.q)])
    nt=(case.n+128*case.group_n-1)//(128*case.group_n)
    pending=0;active={};finished={};now=0.;events=0;transitions=0;wait_integral=0.
    totals=[0.,0.];served=[0.,0.]
    bandwidth=(scenario.memory_bytes_per_unit,scenario.link_bytes_per_unit)

    def demands(tile,partition,stage):
        tm,tn=divmod(tile,nt)
        m=min(64,case.m-64*tm);n=min(128*case.group_n,case.n-128*case.group_n*tn)
        if stage=='main':
            a=2*m*(case.k//case.q)*scenario.a_delivery_fraction
            b=2*n*(case.k//case.q)*scenario.b_delivery_fraction
            return [b+(0 if remote_source else a), a if remote_source else 0.]
        return [2*m*n*(1 if partition==0 else 2),0.]

    def stage_ready(job):
        return job['deadline']<=now+1e-12*max(1.,now) and not any(job['remaining'])

    while len(finished)<len(tasks):
        if time.monotonic()-wall_start>max_seconds:
            raise TimeoutError('conditional simulator CPU deadline')
        changed=True
        while changed:
            changed=False
            for key,job in list(active.items()):
                tile,partition=key
                if job['stage']=='main' and stage_ready(job):
                    job['stage']='wait';transitions+=1;changed=True
                if job['stage']=='wait' and (partition==0 or (tile,partition-1) in finished):
                    job['stage']='epi';job['remaining']=demands(tile,partition,'epi')
                    job['deadline']=now+(scenario.first_epilogue if partition==0 else scenario.later_epilogue)
                    totals[0]+=job['remaining'][0];transitions+=1;changed=True
                if job['stage']=='epi' and stage_ready(job):
                    finished[key]=now;del active[key];transitions+=1;changed=True
            while pending<len(tasks) and len(active)<slots:
                tile,partition=tasks[pending];pending+=1
                work=demands(tile,partition,'main')
                active[tile,partition]={'stage':'main','remaining':work,
                    'deadline':now+scenario.compute_fixed+scenario.compute_per_iteration*case.depth}
                for i in range(2):totals[i]+=work[i]
                transitions+=1;changed=True
            if transitions>4*len(tasks):raise AssertionError('stage transition bound')
        if not active:
            if len(finished)==len(tasks):break
            raise RuntimeError('stalled dependency graph')
        users=[[j for j in active.values() if j['stage']!='wait' and j['remaining'][i]>0] for i in range(2)]
        rates=[bandwidth[i]/len(u) if u else 0. for i,u in enumerate(users)]
        possible=[j['deadline']-now for j in active.values() if j['stage']!='wait' and j['deadline']>now]
        possible += [j['remaining'][i]/rates[i] for i,u in enumerate(users) for j in u]
        if not possible:raise RuntimeError('no feasible next event')
        dt=min(possible)
        if dt<=0 or not math.isfinite(dt):raise ArithmeticError('invalid simulation event')
        wait_integral+=dt*sum(j['stage']=='wait' for j in active.values())
        for i,u in enumerate(users):
            for job in u:
                amount=min(job['remaining'][i],rates[i]*dt)
                old=job['remaining'][i];job['remaining'][i]=max(0.,old-amount)
                if job['remaining'][i] <= 1e-12*max(1.,old):
                    amount+=job['remaining'][i];job['remaining'][i]=0.
                served[i]+=amount
        now+=dt;events+=1
        if events>6*len(tasks):raise AssertionError('event bound')
    for expected,actual,bw in zip(totals,served,bandwidth):
        if not math.isclose(expected,actual,rel_tol=1e-10,abs_tol=1e-7):
            raise AssertionError('resource work not conserved')
        if actual>now*bw+1e-7*max(1.,actual):raise AssertionError('resource capacity exceeded')
    return {'scope':'one per-source GEMM, not whole program',
            'shape_P_M_K_N':[case.world_size,case.m,case.k,case.n],'q':case.q,'group_n':case.group_n,
            'unit':scenario.unit,'scenario_makespan':now,'remote_source':remote_source,'order':order,
            'memory_service_bytes':totals[0],'link_service_bytes':totals[1],
            'service_bytes_are_measured_physical_traffic':False,
            'resource_work_lower_bound':max(t/b for t,b in zip(totals,bandwidth)),
            'resident_wait_slot_time':wait_integral,'completed_ctas':len(finished),
            'events':events,'stage_transitions':transitions,'cpu_seconds':time.monotonic()-wall_start,
            'prediction_us':None,'training_eligible':False,'compiler_admission':False,
            'assumption':scenario.assumption,
            'unmodeled':['GPU placement','per-SM issue competition','cache-line coalescing and reuse',
                         'source HBM contention for remote A','host submission','other source GEMMs and collectives']}
