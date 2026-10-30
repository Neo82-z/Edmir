"""Joint selection of existing cost fits using training-only decisions.

This changes parameter selection, not the arithmetic cost identity or runtime.
The archive has already been inspected; nested replay is retrospective.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'EDM-IR'), str(ROOT/'EDM-IR/tools')]
from edmir.compiler.traffic_latency_selection import (
    CAPACITIES, RIDGES, INPUT_THRESHOLDS, DEPTH_THRESHOLDS, HOST_NAMES,
    FUSED_NAMES as TRAFFIC_NAMES, host_features, fused_features as traffic_features,
    _solve, _domain, training_rows_weights,
)
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.factorized_selection import GroupSupport
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS
from edmir.compiler.measurement_dataset import P4_ENV

_spec = importlib.util.spec_from_file_location('reviewed_resource_basis',
    ROOT/'experiments/2026-10-21-structural-identifiability/candidate.py')
_resource = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _resource
_spec.loader.exec_module(_resource)

VARIANTS = {'traffic_joint_mean': ('traffic', 'mean_regret'),
            'resource_joint_mean': ('resource', 'mean_regret'),
            'resource_joint_worst': ('resource', 'worst_group_mean_regret')}
HOST_GRID = tuple((t,b,r) for t in INPUT_THRESHOLDS for b in DEPTH_THRESHOLDS for r in RIDGES)
FUSED_GRID = tuple((c,r) for c in CAPACITIES for r in RIDGES)


def fused_features(row, capacity, basis):
    if basis == 'traffic':
        return traffic_features(row, capacity)
    if basis != 'resource':
        raise ValueError('unknown basis')
    if not isinstance(row, ScheduleCandidate):
        row = ScheduleCandidate.from_measurement(row, P4_ENV)
    return _resource.fused_features(row, capacity)


def grouped_support(surfaces):
    rows = [r for s in surfaces for r in s.values() if r.n_tiles_per_cta == 2]
    result = []
    for m,q in sorted({(r.rows_per_peer,r.split_k_slices) for r in rows}):
        selected = [r for r in rows if (r.rows_per_peer,r.split_k_slices) == (m,q)]
        result.append(GroupSupport(m,q,min(r.k for r in selected),max(r.k for r in selected),
                                   min(r.n for r in selected),max(r.n for r in selected),len(selected)))
    return tuple(result)


def eligible(row, support):
    return row.n_tiles_per_cta == 1 or any(cell.covers(row) for cell in support)


def component_system(surfaces, host, parameters, basis):
    rows, weights = training_rows_weights(surfaces, host)
    names = HOST_NAMES if host else (TRAFFIC_NAMES if basis == 'traffic' else _resource.FUSED_NAMES)
    raw = [host_features(r,*parameters[:2]) if host else fused_features(r,parameters[0],basis) for r in rows]
    return raw, [r.measured_ms for r in rows], weights, names, parameters[-1]


def fit_component(surfaces, host, parameters, basis):
    return _solve(*component_system(surfaces,host,parameters,basis))


def validate_data(data):
    if len(data.rows) > 4096 or data.environment_id != P4_ENV or data.baseline != PER_SOURCE_CUTLASS:
        raise ValueError('bounded archived A100/per-source CUTLASS data required')
    surfaces = [{n:s[n] for n in sorted(s)} for _,s in sorted(data.validate().items())]
    groups = sorted({r.group_id for r in data.rows})
    if not 3 <= len(groups) <= 32 or not 3 <= len(surfaces) <= 256 or any(not _domain(r) for r in data.rows):
        raise ValueError('outside bounded archived scope')
    if len({(r.rows_per_peer,r.n,r.group_id) for r in data.rows}) != len({(r.rows_per_peer,r.n) for r in data.rows}):
        raise ValueError('one M/N sweep cannot cross inner folds')
    for s in surfaces:
        if len({r.group_id for r in s.values()}) != 1:
            raise ValueError('one shape cannot cross inner folds')
        for n,r in s.items():
            if r.n_tiles_per_cta == 2 and (r.rows_per_peer != 64 or n.removesuffix('_groupedn2') not in s):
                raise ValueError('paired grouped observations required at M64')
    return surfaces, groups


def joint_scores(surfaces, groups, basis, deadline):
    import numpy as np
    regrets, pair_errors, group_means, folds = [], [], [], []
    for held in groups:
        train = [s for s in surfaces if s['host'].group_id != held]
        test = [s for s in surfaces if s['host'].group_id == held]
        support = grouped_support(train)
        fits = []
        for host,grid in ((True,HOST_GRID),(False,FUSED_GRID)):
            selected = []
            for parameters in grid:
                if time.monotonic() >= deadline:
                    raise TimeoutError('joint selection fit deadline exceeded')
                selected.append(fit_component(train,host,parameters,basis))
            fits.append(selected)
        fold_regrets, entries = [], []
        for s in test:
            names = ['host'] + [n for n in sorted(s) if n != 'host' and eligible(s[n],support)]
            h = np.array([f.predict(host_features(s['host'],*p[:2])) for f,p in zip(fits[0],HOST_GRID)])
            f = np.array([[fit.predict(fused_features(s[n],p[0],basis)) for n in names[1:]]
                          for fit,p in zip(fits[1],FUSED_GRID)])
            predictions = np.concatenate((np.broadcast_to(h[:,None,None],(48,12,1)),
                                          np.broadcast_to(f[None,:,:],(48,12,len(names)-1))),axis=2)
            chosen = predictions.argmin(axis=2)
            observed = np.array([s[n].measured_ms for n in names])
            regret = observed[chosen]/min(r.measured_ms for r in s.values()) - 1
            difference = f[None,:,:] - h[:,None,None]
            pair_error = np.mean(((difference-(observed[1:]-observed[0]))/observed[0])**2,axis=2)
            regrets.append(regret); fold_regrets.append(regret); pair_errors.append(pair_error)
            entries.append({'shape':s['host'].shape,'eligible':names})
        group_means.append(np.mean(fold_regrets,axis=0))
        folds.append({'heldout':held,'training_shapes':sorted(s['host'].shape for s in train),
                      'test':entries,'component_fits':60})
    mean, maximum = np.mean(regrets,axis=0), np.max(regrets,axis=0)
    worst, paired = np.max(group_means,axis=0), np.mean(pair_errors,axis=0)
    return tuple({'host_parameters':hp,'fused_parameters':fp,'mean_regret':float(mean[i,j]),
                  'max_regret':float(maximum[i,j]),'worst_group_mean_regret':float(worst[i,j]),
                  'paired_difference_mse':float(paired[i,j])}
                 for i,hp in enumerate(HOST_GRID) for j,fp in enumerate(FUSED_GRID)), tuple(folds)


@lru_cache(maxsize=128)
def _bank(data,basis):
    # The key includes every immutable training row and its measured latency.
    # No outer test rows, persistent files, or previous-model predictions enter.
    surfaces,groups = validate_data(data)
    return joint_scores(surfaces,groups,basis,time.monotonic()+30.)


def parameter_key(record, objective):
    values = (record['mean_regret'],record['max_regret'],record['paired_difference_mse'])
    if objective == 'worst_group_mean_regret':
        values = (record[objective],) + values
    elif objective != 'mean_regret':
        raise ValueError('unknown objective')
    return (*values,record['host_parameters'],record['fused_parameters'])


@dataclass(frozen=True)
class Candidate:
    method: str
    basis: str
    objective: str
    host: object
    fused: object
    host_parameters: tuple
    fused_parameters: tuple
    training_shapes: tuple
    grouped_support: tuple
    cv_scores: tuple
    inner_folds: tuple

    @classmethod
    def fit(cls,data,*,method='resource_joint_mean',timeout_seconds=30.,use_cache=True):
        started=time.monotonic()
        if method not in VARIANTS or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:
            raise ValueError('unknown or unbounded fit configuration')
        basis,objective=VARIANTS[method]
        surfaces,groups=validate_data(data)
        if time.monotonic()-started>=timeout_seconds:
            raise TimeoutError('joint fit deadline exceeded')
        # Canonical order avoids cache/order dependence; cache cannot alter the objective.
        from dataclasses import replace
        canonical=replace(data,rows=tuple(sorted(data.rows,key=lambda r:(r.shape,r.family,r.group_id))))
        scores,folds=(_bank(canonical,basis) if use_cache and timeout_seconds>=30.
                      else joint_scores(surfaces,groups,basis,started+timeout_seconds))
        best=min(scores,key=lambda r:parameter_key(r,objective))
        hp,fp=best['host_parameters'],best['fused_parameters']
        h=fit_component(surfaces,True,hp,basis)
        f=fit_component(surfaces,False,fp,basis)
        if time.monotonic()-started>=timeout_seconds:
            raise TimeoutError('joint fit deadline exceeded')
        return cls(method,basis,objective,h,f,hp,fp,tuple(sorted(s['host'].shape for s in surfaces)),
                   grouped_support(surfaces),scores,folds)

    def select(self,candidates,*,baseline,allow_extrapolation=False):
        baseline.validate()
        if baseline!=PER_SOURCE_CUTLASS or type(allow_extrapolation) is not bool:
            raise ValueError('wrong baseline or extrapolation flag')
        if not candidates or len(candidates)>11 or 'host' not in candidates or any(not isinstance(r,ScheduleCandidate) for r in candidates.values()):
            raise TypeError('static candidate surface with host required')
        if len({r.shape for r in candidates.values()})!=1 or any(n!=r.family for n,r in candidates.items()):
            raise ValueError('one consistently named shape required')
        predictions={}
        for name,row in sorted(candidates.items()):
            row.validate()
            if row.environment_id!=P4_ENV or not _domain(row):
                raise ValueError('outside archive domain')
            if not allow_extrapolation and any(not min(s[i] for s in self.training_shapes)<=row.shape[i]<=max(s[i] for s in self.training_shapes) for i in (1,2,3)):
                raise ValueError('training-range extrapolation requires diagnostic opt-in')
            if not eligible(row,self.grouped_support):
                continue
            predictions[name]=(self.host.predict(host_features(row,*self.host_parameters[:2])) if row.is_host
                               else self.fused.predict(fused_features(row,self.fused_parameters[0],self.basis)))
        choice=min(predictions,key=lambda n:(predictions[n],n!='host',n))
        return {'choice':choice,'predicted_ms':predictions,
                'scores':{n:math.log(t/predictions['host']) for n,t in predictions.items()},
                'eligible_for_compiler_admission':False}

    def to_dict(self):
        return {**asdict(self),'cv_candidates':len(self.cv_scores),'compiler_admission':False}
