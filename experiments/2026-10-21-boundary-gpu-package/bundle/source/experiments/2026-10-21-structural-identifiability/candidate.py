"""Two frozen, experimental resource-coordinate fitting policies.

No shape-specific corrections, no deployment admission, no GPU operations.
The explicit empirical bases are not physical bandwidth measurements.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'EDM-IR/tools')]
from edmir.compiler.traffic_latency_selection import (
    CAPACITIES, RIDGES, INPUT_THRESHOLDS, DEPTH_THRESHOLDS, HOST_NAMES,
    host_features, _solve, training_rows_weights, _domain,
)
from edmir.compiler.resource_geometry import ResourceGeometry
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.factorized_selection import GroupSupport
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from edmir.compiler.measurement_dataset import P4_ENV

FUSED_NAMES = ('fixed', 'partition_K', 'serial_stages', 'extra_ctas',
               'partition_K_pressure', 'serial_pressure',
               'A_working_set', 'B_working_set', 'A_size_squared', 'B_size_squared',
               'group_fixed', 'group_partition_K', 'group_serial_stages',
               'group_K', 'group_ctas', 'output_tiles')


def fused_features(row, capacity):
    geo = ResourceGeometry.from_candidate(row)
    if capacity not in CAPACITIES:
        raise ValueError('unknown pressure scale')
    k, q, g = row.k/4096, row.split_k_slices, row.n_tiles_per_cta
    grid = geo.output_tiles
    pressure = max(grid*q/capacity - 1, 0.)
    a, b = geo.a_working_set_bytes/(8*1024**2), geo.b_working_set_bytes/(32*1024**2)
    return (1., k/q, q-1., grid*(q-1)/108, k/q*pressure,
            (q-1)*pressure/16, a, b, a*a, b*b,
            *(float(g-1)*v for v in (1., k/q, q-1., k, grid*q/108)), grid/108)


def fit_path(surfaces, host, parameters, ridge):
    rows, weights = training_rows_weights(surfaces, host)
    raw = [host_features(r, *parameters) if host else fused_features(
        ScheduleCandidate.from_measurement(r, P4_ENV), parameters) for r in rows]
    return _solve(raw, [r.measured_ms for r in rows], weights,
                  HOST_NAMES if host else FUSED_NAMES, ridge)


@dataclass
class Candidate:
    host_fits: tuple
    fused_fits: tuple
    training_shapes: tuple
    grouped_support: tuple
    host_cv: tuple
    fused_cv: tuple
    average: bool

    @classmethod
    def fit(cls, data, *, average=False, timeout_seconds=15):
        deadline = time.monotonic() + timeout_seconds
        if type(average) is not bool or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('invalid bounded fitting configuration')
        if (data.environment_id != P4_ENV or data.baseline != PER_SOURCE_CUTLASS
                or len(data.rows) > 4096 or any(not _domain(r) for r in data.rows)):
            raise ValueError('archive contract required')
        ss = [{n:s[n] for n in sorted(s)} for _,s in sorted(data.validate().items())]
        groups = sorted({r.group_id for r in data.rows})
        if not 3 <= len(groups) <= 32 or not 3 <= len(ss) <= 256:
            raise ValueError('bounded, grouped training data required')
        for s in ss:
            if len({r.group_id for r in s.values()}) != 1:
                raise ValueError('inconsistent surface grouping')
            for n,r in s.items():
                if r.n_tiles_per_cta == 2 and (r.rows_per_peer != 64 or n.removesuffix('_groupedn2') not in s):
                    raise ValueError('paired grouped observations required at M64')
        if len({(r.rows_per_peer,r.n,r.group_id) for r in data.rows}) != len({(r.rows_per_peer,r.n) for r in data.rows}):
            raise ValueError('one M/N sweep cannot cross inner folds')
        def tune(host):
            params = [(t,b) for t in INPUT_THRESHOLDS for b in DEPTH_THRESHOLDS] if host else CAPACITIES
            scores = []
            for param in params:
                for ridge in RIDGES:
                    errors = []
                    for held in groups:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('bounded nested fit deadline exceeded')
                        train = [s for s in ss if s['host'].group_id != held]
                        test = [s for s in ss if s['host'].group_id == held]
                        f = fit_path(train, host, param, ridge)
                        for s in test:
                            rows = [s['host']] if host else [r for r in s.values() if not r.is_host]
                            e = []
                            for r in rows:
                                x = host_features(r,*param) if host else fused_features(ScheduleCandidate.from_measurement(r,P4_ENV),param)
                                e.append(((f.predict(x)-r.measured_ms)/s['host'].measured_ms)**2)
                            errors.append(math.fsum(e)/len(e))
                    scores.append((param,ridge,math.fsum(errors)/len(errors)))
            selected = scores if average else [min(scores,key=lambda s:(s[2],s[0],s[1]))]
            fits=[]
            for p,r,_ in selected:
                if time.monotonic() >= deadline:
                    raise TimeoutError('bounded final fit deadline exceeded')
                fits.append((p,r,fit_path(ss,host,p,r)))
            return tuple(fits),tuple(scores)
        h,hc = tune(True)
        f,fc = tune(False)
        grouped = [r for r in data.rows if r.n_tiles_per_cta == 2]
        support=[]
        for m,q in sorted({(r.rows_per_peer,r.split_k_slices) for r in grouped}):
            rows=[r for r in grouped if (r.rows_per_peer,r.split_k_slices)==(m,q)]
            support.append(GroupSupport(m,q,min(r.k for r in rows),max(r.k for r in rows),min(r.n for r in rows),max(r.n for r in rows),len(rows)))
        return cls(h,f,tuple(sorted({r.shape for r in data.rows})),tuple(support),hc,fc,average)

    def select(self, candidates, *, baseline, allow_extrapolation=False):
        baseline.validate()
        if baseline != PER_SOURCE_CUTLASS or type(allow_extrapolation) is not bool:
            raise ValueError('wrong host identity or extrapolation flag')
        if (not candidates or len(candidates)>11 or 'host' not in candidates
                or any(not isinstance(r,ScheduleCandidate) for r in candidates.values())):
            raise TypeError('one bounded static surface with host required')
        if (len({r.shape for r in candidates.values()})!=1 or any(n!=r.family for n,r in candidates.items())):
            raise ValueError('one consistently named workload required')
        predictions={}
        for name,r in sorted(candidates.items()):
            r.validate()
            if r.environment_id != P4_ENV or not _domain(r):
                raise ValueError('outside archived scope')
            if not allow_extrapolation and any(not min(s[i] for s in self.training_shapes)<=r.shape[i]<=max(s[i] for s in self.training_shapes) for i in (1,2,3)):
                raise ValueError('training-range extrapolation requires diagnostic opt-in')
            if r.n_tiles_per_cta==2 and not any(s.covers(r) for s in self.grouped_support):
                continue
            fits=self.host_fits if r.is_host else self.fused_fits
            predictions[name]=math.fsum(f.predict(host_features(r,*p) if r.is_host else fused_features(r,p)) for p,_,f in fits)/len(fits)
        choice=min(predictions,key=lambda n:(predictions[n],n!='host',n))
        return {'choice':choice,'predicted_ms':predictions,'scores':{n:math.log(v/predictions['host']) for n,v in predictions.items()},
                'eligible_for_compiler_admission':False}

    def audit(self):
        return {'average':self.average,'host_fits':[(p,r,asdict(f)) for p,r,f in self.host_fits],
                'fused_fits':[(p,r,asdict(f)) for p,r,f in self.fused_fits],
                'host_cv':self.host_cv,'fused_cv':self.fused_cv,'training_shapes':self.training_shapes}
