"""Experimental shared residual scores trained on measured candidate costs.

The existing additive model is a fold-local empirical prior, not a physical
timing identity. A shared numerical network corrects its log latency. The
softmin weights are a training policy, NOT calibrated probabilities of being
fastest. Only observed candidates enter the objective. No compiler admission.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import time

from .additive_latency_selection import AdditiveLatencySelector
from .execution_cost_selection import ceildiv
from .factorized_selection import IdentifiedMeasurements
from .host_baseline import PER_SOURCE_CUTLASS
from .measurement_dataset import P4_ENV
from .relative_selection import ScheduleCandidate
from .traffic_latency_selection import _domain

SCHEMA = "edmir.shared-candidate-selector.v1"
VARIANTS = {
    "shared_linear_latency": (0, 0.0),
    "shared_linear_cost": (0, 1.0),
    "shared_mlp_cost": (8, 1.0),
}
FEATURE_NAMES = (
    "host", "log_M", "log_K", "log_N", "log_q", "log_group_N",
    "partition_K", "serial_boundaries", "output_tiles_per_SM",
    "ctas_per_SM", "integer_waves", "valid_tile_fraction",
    "log1p_A_MiB", "log1p_B_MiB", "log1p_extra_output_MiB",
    "log1p_remote_input_MiB", "remote_A_replay", "log_prior_ms",
)
TEMPERATURE, RIDGE = 0.05, 0.001
MAX_ITER, MAX_EVAL, MAX_SECONDS = 400, 900, 30.0
SEEDS = (0, 1)


def static_features(row: ScheduleCandidate, prior_ms: float):
    """Counts from the SM80 contract; no measured bandwidth/occupancy claims.

    Host uses the registered 64x128x32 per-source kernel. Its communication
    schedule differs, encoded by `host` and replay; no fake host split stages.
    SM count is a fixed environment property. Cross-hardware transfer is not
    learned from this single-environment archive and is explicitly rejected.
    """
    if not isinstance(row, ScheduleCandidate):
        raise TypeError("static candidates only; latency labels are forbidden")
    row.validate()
    if row.environment_id != P4_ENV or not _domain(row):
        raise ValueError("outside the archived A100 P4 domain")
    if not math.isfinite(prior_ms) or prior_ms <= 0:
        raise ValueError("positive finite prior required")
    m, k, n, q, g = row.rows_per_peer, row.k, row.n, row.split_k_slices, row.n_tiles_per_cta
    tiles = ceildiv(m, 64)*ceildiv(n, 128*g)
    mib = 2**20
    return (float(row.is_host), math.log2(m/64), math.log2(k/4096), math.log2(n/512),
            math.log2(q), math.log2(g), k/(4096*q), (q-1)/16,
            tiles/108, tiles*q/108, float(ceildiv(tiles*q, 108)),
            m*n/(tiles*64*128*g), math.log1p(2*m*k/mib), math.log1p(2*k*n/mib),
            math.log1p(4*m*n*(q-1)/mib), math.log1p(row.remote_input_bytes/mib),
            row.remote_a_replay/16, math.log(prior_ms))


def canonical_choice(predictions):
    if not predictions or any(not math.isfinite(v) for v in predictions.values()):
        raise ValueError("finite nonempty scores required")
    # Exact ties only. There is no host margin or input-order preference.
    return min(predictions, key=lambda n: (predictions[n], n != "host", n))


def parameter_count(dim, width):
    return dim + 1 + (dim*width + 2*width if width else 0)


def forward(theta, x, width):
    import numpy as np
    d = x.shape[-1]
    out = x @ theta[:d] + theta[d]
    if width:
        w = theta[d+1:d+1+d*width].reshape(d, width)
        b = theta[d+1+d*width:d+1+d*width+width]
        h = np.tanh(x @ w + b)
        out = out + h @ theta[-width:]
    return out


def objective(theta, x, base, targets, mask, row_weights, width, cost_weight,
              temperature=TEMPERATURE, ridge=RIDGE):
    """Exact full-information expected regret, plus log-time anchoring.

    x: [shapes, padded candidates, features]. Every shape has weight one.
    Log-time anchoring balances host and all fused candidates (1/2 each).
    Missing candidates have zero mask and never become invented observations.
    Analytic derivatives are checked against central finite differences.
    """
    import numpy as np
    logt = base + forward(theta, x, width)
    error = np.where(mask, logt-targets, 0.0)
    count = x.shape[0]
    mse = float(np.sum(row_weights*error**2)/count)
    grad_logt = 2*row_weights*error/count
    logits = np.where(mask, -logt/temperature, -np.inf)
    logits = logits-np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    best = np.min(np.where(mask, targets, np.inf), axis=1, keepdims=True)
    regret = np.expm1(np.where(mask, targets-best, 0.0))
    expected = np.sum(p*regret, axis=1, keepdims=True)
    cost = float(expected.mean())
    grad_logt -= cost_weight*p*(regret-expected)/(temperature*count)
    grad_logt *= mask
    d = x.shape[-1]
    gradient = np.zeros_like(theta)
    gradient[:d] = np.einsum("sjd,sj->d", x, grad_logt)
    gradient[d] = grad_logt.sum()
    if width:
        offset = d+1
        w = theta[offset:offset+d*width].reshape(d, width)
        b = theta[offset+d*width:offset+d*width+width]
        h = np.tanh(x@w+b)
        dh = grad_logt[..., None]*theta[-width:]*(1-h*h)
        gradient[offset:offset+d*width] = np.einsum("sjd,sjh->dh", x, dh).ravel()
        gradient[offset+d*width:offset+d*width+width] = dh.sum(axis=(0, 1))
        gradient[-width:] = np.einsum("sjh,sj->h", h, grad_logt)
    value = mse + cost_weight*cost + ridge*float(theta@theta)
    gradient += 2*ridge*theta
    if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("nonfinite shared objective")
    return value, gradient


def training_tensors(surfaces, prior):
    import numpy as np
    raw, bases, targets, names = [], [], [], []
    for surface in surfaces:
        candidates = {n: ScheduleCandidate.from_measurement(r, P4_ENV) for n, r in surface.items()}
        predicted = prior.select(candidates, baseline=PER_SOURCE_CUTLASS)["predicted_ms"]
        ns = sorted(predicted)
        # Prior eligibility may remove unsupported candidates, never add labels.
        names.append(tuple(ns))
        raw.append([static_features(candidates[n], predicted[n]) for n in ns])
        bases.append([math.log(predicted[n]) for n in ns])
        targets.append([math.log(surface[n].measured_ms) for n in ns])
    s, j, d = len(raw), max(map(len, raw)), len(FEATURE_NAMES)
    x, base, target = np.zeros((s,j,d)), np.zeros((s,j)), np.zeros((s,j))
    mask, weights = np.zeros((s,j), bool), np.zeros((s,j))
    for i, (xx, bb, yy, ns) in enumerate(zip(raw, bases, targets, names)):
        size = len(xx)
        if size < 2 or "host" not in ns:
            raise ValueError("a measured host and fused candidate are required")
        x[i,:size], base[i,:size], target[i,:size], mask[i,:size] = xx, bb, yy, True
        weights[i,:size] = [.5 if n == "host" else .5/(size-1) for n in ns]
    # Scaler is fit only on this training fold, with one unit of mass per shape.
    means = np.sum(x*weights[...,None], axis=(0,1))/s
    variance = np.sum((x-means)**2*weights[...,None], axis=(0,1))/s
    scales = np.maximum(np.sqrt(variance), .1)
    x = np.where(mask[...,None], (x-means)/scales, 0.0)
    return x, base, target, mask, weights, means, scales, tuple(names)


def _fit(dataset, method, timeout_seconds):
    import numpy as np
    from scipy.optimize import minimize
    deadline = time.monotonic()+timeout_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("shared candidate fit deadline exceeded")
    original = dataset.validate()
    surfaces = [{n:s[n] for n in sorted(s)} for _,s in sorted(original.items())]
    prior = AdditiveLatencySelector.fit(dataset, timeout_seconds=max(1e-9, deadline-time.monotonic()))
    check()
    x, base, target, mask, weights, means, scales, names = training_tensors(surfaces, prior)
    check()
    d = len(FEATURE_NAMES)
    width, cost_weight = VARIANTS[method]
    # Exact ridge initialization of the shared affine residual, using training data only.
    a = np.concatenate((x, np.ones((*x.shape[:2],1))), axis=-1).reshape(-1,d+1)
    w = weights.ravel()/len(surfaces)
    affine = np.linalg.solve(a.T@(a*w[:,None])+RIDGE*np.eye(d+1), a.T@(w*(target-base).ravel()))
    theta0 = np.zeros(parameter_count(d,width)); theta0[:d+1] = affine
    starts, parameters = [], []
    for seed in (SEEDS if width else (0,)):
        check()
        theta = theta0.copy()
        if width:
            theta[d+1:d+1+d*width] = np.random.default_rng(seed).normal(0, .15/math.sqrt(d), d*width)
        def loss(t):
            check()
            return objective(t,x,base,target,mask,weights,width,cost_weight)
        initial = loss(theta)[0]
        result = minimize(loss,theta,method="L-BFGS-B",jac=True,
                          options={"maxiter":MAX_ITER,"maxfun":MAX_EVAL,"maxls":25,"gtol":1e-7,"ftol":1e-12})
        check()
        value, gradient = loss(result.x)
        if value > initial+1e-10 or not np.all(np.isfinite(result.x)):
            raise ArithmeticError("optimizer worsened its initial objective")
        starts.append({"seed":seed,"initial_loss":initial,"final_loss":value,
                       "iterations":int(result.nit),"evaluations":int(result.nfev),
                       "success":bool(result.success),"status":int(result.status),"message":str(result.message),
                       "gradient_max_abs":float(np.max(np.abs(gradient)))})
        parameters.append(tuple(map(float,result.x)))
    best = min(range(len(starts)),key=lambda i:(starts[i]["final_loss"],starts[i]["seed"]))
    return SharedCandidateSelector(method, prior, tuple(map(float,means)), tuple(map(float,scales)), parameters[best],
                                   {"starts":starts,"selected_start":best,"training_candidates":names,
                                    "parameter_count":len(parameters[best]),"width":width,
                                    "cost_weight":cost_weight,"temperature":TEMPERATURE,"ridge":RIDGE,
                                    "maxiter":MAX_ITER,"maxfun":MAX_EVAL})


@lru_cache(maxsize=128)
def _fit_cached(dataset, method):
    return _fit(dataset, method, MAX_SECONDS)


@dataclass(frozen=True)
class SharedCandidateSelector:
    method: str
    prior: AdditiveLatencySelector
    means: tuple[float, ...]
    scales: tuple[float, ...]
    parameters: tuple[float, ...]
    training_audit: dict

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, method="shared_mlp_cost",
            timeout_seconds=MAX_SECONDS, use_cache=True):
        if method not in VARIANTS or type(use_cache) is not bool:
            raise ValueError("unknown shared selector method or cache flag")
        if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:
            raise ValueError("finite positive timeout required")
        if not isinstance(dataset,IdentifiedMeasurements) or len(dataset.rows)>4096:
            raise ValueError("bounded identified training dataset required")
        if dataset.environment_id != P4_ENV or dataset.baseline != PER_SOURCE_CUTLASS:
            raise ValueError("archived A100 P4 and per-source CUTLASS required")
        canonical = replace(dataset,rows=tuple(sorted(dataset.rows,key=lambda r:(r.shape,r.family,r.group_id))))
        result = (_fit_cached(canonical,method) if use_cache and timeout_seconds>=MAX_SECONDS
                  else _fit(canonical,method,min(timeout_seconds,MAX_SECONDS)))
        result.validate()
        # Cached state is never exposed as a mutable object to callers.
        import copy
        return replace(result,training_audit=copy.deepcopy(result.training_audit))

    def validate(self):
        if self.method not in VARIANTS:
            raise ValueError("unknown model method")
        self.prior.validate()
        if len(self.means)!=len(FEATURE_NAMES) or len(self.scales)!=len(FEATURE_NAMES):
            raise ValueError("wrong feature dimensions")
        if len(self.parameters)!=parameter_count(len(FEATURE_NAMES),VARIANTS[self.method][0]):
            raise ValueError("wrong parameter dimensions")
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (*self.means,*self.scales,*self.parameters)):
            raise ValueError("nonfinite or nonnumeric state")
        if any(v<=0 for v in self.scales):
            raise ValueError("positive feature scales required")
        width,cost_weight=VARIANTS[self.method]
        expected={"width":width,"cost_weight":cost_weight,"temperature":TEMPERATURE,
                  "ridge":RIDGE,"parameter_count":len(self.parameters),"maxiter":MAX_ITER,"maxfun":MAX_EVAL}
        if any(self.training_audit.get(k)!=v for k,v in expected.items()):
            raise ValueError("inconsistent training definition")

    def select(self,candidates,*,baseline,allow_extrapolation=False):
        import numpy as np
        # The prior enforces static-only inputs, names, hardware, host identity,
        # aligned partitions, grouped-N training support and diagnostic extrapolation.
        previous = self.prior.select(candidates,baseline=baseline,allow_extrapolation=allow_extrapolation)
        names = sorted(previous["predicted_ms"])
        raw = np.array([static_features(candidates[n],previous["predicted_ms"][n]) for n in names])
        x = (raw-self.means)/self.scales
        # Independent row scoring: candidate order/set size is not a feature.
        residual = forward(np.asarray(self.parameters),x,VARIANTS[self.method][0])
        logs = np.log([previous["predicted_ms"][n] for n in names])+residual
        if not np.all(np.isfinite(logs)) or np.max(np.abs(logs))>80:
            raise FloatingPointError("invalid predicted log latency")
        predictions = dict(zip(names,map(float,np.exp(logs))))
        choice = canonical_choice(predictions)
        logits = -logs/TEMPERATURE; weights=np.exp(logits-np.max(logits)); weights/=weights.sum()
        return {"choice":choice,"predicted_ms":predictions,
                "scores":{n:math.log(t/predictions['host']) for n,t in predictions.items()},
                "policy_weights":dict(zip(names,map(float,weights))),
                "weight_semantics":"softmin selection weights, not calibrated optimality probabilities",
                "eligible_for_compiler_admission":False}

    def to_dict(self):
        import copy
        return {"schema":SCHEMA,"method":self.method,"feature_names":list(FEATURE_NAMES),
                "prior":self.prior.to_dict(),"means":list(self.means),"scales":list(self.scales),
                "parameters":list(self.parameters),"training_audit":copy.deepcopy(self.training_audit),
                "training_shapes":list(self.prior.training_shapes),"compiler_admission":False}

    @classmethod
    def from_dict(cls,obj):
        import copy
        keys={'schema','method','feature_names','prior','means','scales','parameters',
              'training_audit','training_shapes','compiler_admission'}
        if (not isinstance(obj,dict) or set(obj)!=keys or obj.get('schema')!=SCHEMA
                or obj.get('feature_names')!=list(FEATURE_NAMES) or obj.get('compiler_admission') is not False):
            raise ValueError("unknown shared model schema")
        result=cls(obj['method'],AdditiveLatencySelector.from_dict(obj['prior']),tuple(obj['means']),
                   tuple(obj['scales']),tuple(obj['parameters']),copy.deepcopy(obj['training_audit']))
        result.validate()
        if [list(s) for s in result.prior.training_shapes] != [list(s) for s in obj['training_shapes']]:
            raise ValueError("inconsistent training provenance")
        return result
