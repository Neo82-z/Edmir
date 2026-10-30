"""Traffic-coupled host/fused latency model for the archived A100 P4 data.

Input M*K and depth/N responses have separate empirical bases. Fused base
output tiles and extra partition work are separated. Relative-error training
weights are derived only from training labels; inference never sees labels.
These fitted bases do not identify physical bandwidth or cache thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Mapping

from .execution_cost_selection import ExecutionGeometry, ceildiv
from .factorized_selection import GroupSupport, IdentifiedMeasurements
from .host_baseline import HostBaseline, PER_SOURCE_CUTLASS
from .measurement_dataset import P4_ENV
from .relative_selection import ScheduleCandidate
from .two_stage_selection import _validate_geometry

SCHEMA = "edmir.traffic-latency-selector.v1"
CAPACITIES = (16, 32, 64, 108)
RIDGES = (.001, .01, .1)
INPUT_THRESHOLDS = (1, 2, 4, 8)
DEPTH_THRESHOLDS = (.25, .5, 1., 2.)
HOST_NAMES = ("fixed", "K", "K_squared", "depth_tail_log_N", "input_size_log_tail", "input_elements")
FUSED_NAMES = ("fixed", "partition_K", "serial_stages", "extra_ctas", "partition_K_pressure",
               "serial_pressure", "depth_tail_output_tiles", "group_fixed", "group_partition_K",
               "group_serial_stages", "group_K", "group_ctas", "output_tiles")
MAX_ROWS, MAX_SHAPES, MAX_GROUPS = 4096, 256, 32


def host_features(row, input_threshold, depth_threshold):
    _validate_geometry(row)
    if input_threshold not in INPUT_THRESHOLDS or depth_threshold not in DEPTH_THRESHOLDS:
        raise ValueError("unknown host cost basis")
    k, n = row.k/4096, math.log2(row.n/128)
    a = row.rows_per_peer*row.k/(64*4096)
    # A coupled input-size basis, not an identified communication service time.
    return (1., k, k*k, max(k-depth_threshold, 0.)*n,
            max(math.log2(a/input_threshold), 0.), a/16)


def fused_features(row, capacity):
    _validate_geometry(row)
    if row.is_host or capacity not in CAPACITIES:
        raise ValueError("fused features require a fused candidate and known pressure threshold")
    k, q, g = row.k/4096, row.split_k_slices, row.n_tiles_per_cta
    grid = ceildiv(row.rows_per_peer, 64)*ceildiv(row.n, 128*g)
    pressure = max(grid*q/capacity-1, 0.)
    return (1., k/q, q-1., grid*(q-1)/108, k/q*pressure,
            (q-1)*pressure/16, max(k-1., 0.)*grid/108,
            *(float(g-1)*v for v in (1., k/q, q-1., k, grid*q/108)), grid/108)


def _domain(row):
    # The historical release contains only the M sweep at N=512 and N sweep
    # at M=64. Do not imply evidence for arbitrary combinations of both axes.
    return (row.world_size == 4 and 32 <= row.rows_per_peer <= 512
            and 512 <= row.k <= 8192 and 128 <= row.n <= 2048
            and (row.rows_per_peer == 64 or row.n == 512)
            and row.k % (32*row.split_k_slices) == 0)


def _finite_nonnegative(values):
    return all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)


@dataclass(frozen=True, slots=True)
class LatencyFit:
    names: tuple[str, ...]
    coefficients: tuple[float, ...]
    scales: tuple[float, ...]
    kkt_max_violation: float

    def validate(self, names):
        if self.names != names or len(self.coefficients) != len(names) or len(self.scales) != len(names):
            raise ValueError("invalid latency feature schema")
        if not _finite_nonnegative((*self.coefficients, *self.scales, self.kkt_max_violation)):
            raise ValueError("invalid nonnegative fitted state")
        if any(s <= 0 for s in self.scales) or self.coefficients[0] <= 0 or self.kkt_max_violation > 1e-7:
            raise ValueError("invalid fit or failed numerical certificate")

    def predict(self, x):
        value = math.fsum(c*v for c, v in zip(self.coefficients, x))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid predicted latency")
        return value


def _solve(raw, targets, weights, names, ridge):
    import numpy as np
    from scipy.optimize import nnls
    x, y = np.asarray(raw, dtype=float), np.asarray(targets, dtype=float)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-9)
    w = np.asarray(weights, dtype=float)
    x, y = x/scale*w[:, None], y*w
    penalty = np.eye(len(names))*math.sqrt(ridge)
    penalty[0, 0] = 0.
    a, b = np.vstack((x, penalty)), np.r_[y, np.zeros(len(names))]
    coef = nnls(a, b, maxiter=1000)[0]
    gradient = a.T@(a@coef-b)
    # KKT conditions certify the convex nonnegative least-squares solution.
    violation = max(float(np.max(np.maximum(-gradient, 0))),
                    float(np.max(np.abs(gradient[coef > 1e-10]))) if any(coef > 1e-10) else 0.)
    result = LatencyFit(names, tuple(float(v) for v in coef/scale), tuple(float(v) for v in scale), violation)
    result.validate(names)
    return result


def training_rows_weights(surfaces, host):
    rows, weights = [], []
    for surface in surfaces:
        selected = [surface["host"]] if host else [r for r in surface.values() if not r.is_host]
        rows.extend(selected)
        weights.extend([1/math.sqrt(len(selected))]*len(selected))
    if not host:
        mean = math.fsum(r.measured_ms for r in rows)/len(rows)
        relative = [mean/r.measured_ms for r in rows]
        norm = math.sqrt(math.fsum(v*v for v in relative)/len(relative))
        weights = [w*v/norm for w, v in zip(weights, relative)]
    return rows, weights


def _fit_path(surfaces, host, parameters, ridge):
    rows, weights = training_rows_weights(surfaces, host)
    raw = [host_features(r, *parameters) if host else fused_features(r, parameters) for r in rows]
    return _solve(raw, [r.measured_ms for r in rows], weights, HOST_NAMES if host else FUSED_NAMES, ridge)


@dataclass(frozen=True, slots=True)
class TrafficLatencySelector:
    host: LatencyFit
    fused: LatencyFit
    capacity: int
    ridge: float
    input_threshold: int
    depth_threshold: float
    host_ridge: float
    host_cv_scores: tuple[tuple[int, float, float, float], ...]
    environment_id: str
    baseline_id: str
    training_shapes: tuple[tuple[int, int, int, int], ...]
    trained_q: tuple[int, ...]
    grouped_support: tuple[GroupSupport, ...]
    cv_scores: tuple[tuple[int, float, float], ...]
    training_bounds: tuple[tuple[int, int], ...]

    def validate(self):
        self.host.validate(HOST_NAMES)
        self.fused.validate(FUSED_NAMES)
        if type(self.capacity) is not int or self.capacity not in CAPACITIES or self.ridge not in RIDGES:
            raise ValueError("unknown hyperparameters")
        if (type(self.input_threshold) is not int or self.input_threshold not in INPUT_THRESHOLDS
                or type(self.depth_threshold) not in (int, float)
                or self.depth_threshold not in DEPTH_THRESHOLDS or self.host_ridge not in RIDGES):
            raise ValueError("unknown host hyperparameters")
        required = {(t, b, r) for t in INPUT_THRESHOLDS for b in DEPTH_THRESHOLDS for r in RIDGES}
        if (len(self.host_cv_scores) != len(required)
                or {(t, b, r) for t, b, r, _ in self.host_cv_scores} != required
                or not _finite_nonnegative(v for _, _, _, v in self.host_cv_scores)):
            raise ValueError("incomplete or invalid host inner-selection audit")
        host_best = min(self.host_cv_scores, key=lambda x: (x[3], x[0], x[1], x[2]))
        if host_best[:3] != (self.input_threshold, self.depth_threshold, self.host_ridge):
            raise ValueError("host parameters differ from training-only selection")
        if self.environment_id != P4_ENV or self.baseline_id != PER_SOURCE_CUTLASS.baseline_id:
            raise ValueError("only the archived A100 P4/CUTLASS environment is supported")
        if not 3 <= len(self.training_shapes) <= MAX_SHAPES or len(set(self.training_shapes)) != len(self.training_shapes):
            raise ValueError("invalid training provenance")
        for s in self.training_shapes:
            if (len(s) != 4 or any(type(v) is not int or v <= 0 for v in s) or s[0] != 4
                    or not (32 <= s[1] <= 512 and 512 <= s[2] <= 8192 and 128 <= s[3] <= 2048)
                    or (s[1] != 64 and s[3] != 512)):
                raise ValueError("invalid training shape")
        bounds = tuple((min(s[i] for s in self.training_shapes), max(s[i] for s in self.training_shapes)) for i in (1, 2, 3))
        if self.training_bounds != bounds:
            raise ValueError("invalid training bounds")
        if (not self.trained_q or 1 not in self.trained_q or len(set(self.trained_q)) != len(self.trained_q)
                or any(type(q) is not int or q not in (1, 2, 4, 8, 16) for q in self.trained_q)):
            raise ValueError("invalid split support")
        if len(self.cv_scores) != len(CAPACITIES)*len(RIDGES):
            raise ValueError("incomplete nested selection audit")
        if {(c, r) for c, r, _ in self.cv_scores} != {(c, r) for c in CAPACITIES for r in RIDGES}:
            raise ValueError("invalid nested search coverage")
        if not _finite_nonnegative(v for _, _, v in self.cv_scores):
            raise ValueError("nonfinite validation loss")
        best = min(self.cv_scores, key=lambda t: (t[2], t[0], t[1]))
        if (self.capacity, self.ridge) != best[:2]:
            raise ValueError("parameters do not match training-only selection")
        if len({(s.m, s.q) for s in self.grouped_support}) != len(self.grouped_support):
            raise ValueError("duplicate grouped support")
        for s in self.grouped_support:
            if (any(type(v) is not int or v <= 0 for v in asdict(s).values()) or s.m != 64
                    or s.q not in self.trained_q or s.k_min > s.k_max or s.n_min > s.n_max
                    or not (bounds[1][0] <= s.k_min <= s.k_max <= bounds[1][1])
                    or not (bounds[2][0] <= s.n_min <= s.n_max <= bounds[2][1])):
                raise ValueError("invalid grouped support")

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, timeout_seconds=30.):
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("finite positive fitting deadline required")
        deadline = time.monotonic()+timeout_seconds
        if len(dataset.rows) > MAX_ROWS:
            raise ValueError("training data exceeds bounded local fitter")
        original = dataset.validate()
        if dataset.environment_id != P4_ENV or any(not _domain(r) for r in dataset.rows):
            raise ValueError("training data outside audited A100 P4 geometry")
        # Stable iteration order, including when the archive row order changes.
        surfaces = [{n: s[n] for n in sorted(s)} for _, s in sorted(original.items())]
        groups = sorted({r.group_id for r in dataset.rows})
        if not 3 <= len(surfaces) <= MAX_SHAPES or not 3 <= len(groups) <= MAX_GROUPS:
            raise ValueError("at least three bounded training groups required")
        if any(len({r.group_id for r in s.values()}) != 1 for s in surfaces):
            raise ValueError("one workload cannot have conflicting fold identities")
        by_shape = {}
        for s in surfaces:
            r = next(iter(s.values()))
            key = (r.rows_per_peer, r.n)
            if key in by_shape and by_shape[key] != r.group_id:
                raise ValueError("one M/N sweep cannot be split across inner folds")
            by_shape[key] = r.group_id
            for n, r in s.items():
                if r.n_tiles_per_cta == 2 and (r.rows_per_peer != 64 or n.removesuffix("_groupedn2") not in s):
                    raise ValueError("grouped support requires M64 and paired ungrouped data")
        fold_data = [(tuple(s for s in surfaces if next(iter(s.values())).group_id != g),
                      tuple(s for s in surfaces if next(iter(s.values())).group_id == g)) for g in groups]
        def tune(host):
            parameters = tuple((t, b) for t in INPUT_THRESHOLDS for b in DEPTH_THRESHOLDS) if host else CAPACITIES
            scores = []
            for param in parameters:
                for ridge in RIDGES:
                    errors = []
                    for train, test in fold_data:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("bounded nested fitting deadline exceeded")
                        fit = _fit_path(train, host, param, ridge)
                        for surface in test:
                            rows = [surface["host"]] if host else [r for r in surface.values() if not r.is_host]
                            norm = surface["host"].measured_ms
                            error = math.fsum(((fit.predict(host_features(r, *param) if host else fused_features(r, param))
                                               -r.measured_ms)/norm)**2 for r in rows)/len(rows)
                            errors.append(error)
                    scores.append((param, ridge, math.fsum(errors)/len(errors)))
            best = min(scores, key=lambda x: (x[2], x[0], x[1]))
            return best, scores
        (host_parameters, host_ridge, _), host_scores = tune(True)
        (c, ridge, _), scores = tune(False)
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded nested fitting deadline exceeded")
        host = _fit_path(surfaces, True, host_parameters, host_ridge)
        fused = _fit_path(surfaces, False, c, ridge)
        grouped = [r for r in dataset.rows if r.n_tiles_per_cta == 2]
        support = []
        for m, q in sorted({(r.rows_per_peer, r.split_k_slices) for r in grouped}):
            rows = [r for r in grouped if (r.rows_per_peer, r.split_k_slices) == (m, q)]
            support.append(GroupSupport(m, q, min(r.k for r in rows), max(r.k for r in rows),
                                        min(r.n for r in rows), max(r.n for r in rows), len(rows)))
        shapes = tuple(sorted({r.shape for r in dataset.rows}))
        bounds = tuple((min(s[i] for s in shapes), max(s[i] for s in shapes)) for i in (1, 2, 3))
        result = cls(host, fused, c, ridge, *host_parameters, host_ridge,
                     tuple((*p, r, loss) for p, r, loss in host_scores),
                     dataset.environment_id, dataset.baseline.baseline_id, shapes,
                     tuple(sorted({r.split_k_slices for r in dataset.rows if not r.is_host})), tuple(support), tuple(scores), bounds)
        result.validate()
        return result

    def _explain(self, row, allow_extrapolation):
        if not isinstance(row, ScheduleCandidate):
            raise TypeError("inference requires static candidates without timing labels")
        row.validate()
        if row.environment_id != self.environment_id or not _domain(row):
            raise ValueError("candidate outside archived environment/geometry contract")
        outside = any(not lo <= v <= hi for v, (lo, hi) in zip(row.shape[1:], self.training_bounds))
        if outside and not allow_extrapolation:
            raise ValueError("training-range extrapolation requires explicit diagnostic opt-in")
        if row.split_k_slices not in self.trained_q:
            return {"eligible": False, "reason": "split factor absent from training"}
        if row.n_tiles_per_cta == 2 and not any(s.covers(row) for s in self.grouped_support):
            return {"eligible": False, "reason": "no paired group support at exact M/q within observed K/N bounds"}
        fit, x = (self.host, host_features(row, self.input_threshold, self.depth_threshold)) if row.is_host else (self.fused, fused_features(row, self.capacity))
        return {"eligible": True, "predicted_ms": fit.predict(x), "training_range_extrapolation": outside,
                "terms_ms": dict(zip(fit.names, (c*v for c, v in zip(fit.coefficients, x)))),
                "execution_counts": None if row.is_host else asdict(ExecutionGeometry.from_candidate(row))}

    def select(self, candidates: Mapping[str, ScheduleCandidate], *, baseline: HostBaseline,
               allow_extrapolation=False):
        self.validate()
        baseline.validate()
        if baseline.baseline_id != self.baseline_id:
            raise ValueError("host denominator differs from fitted baseline")
        if type(allow_extrapolation) is not bool:
            raise ValueError("extrapolation flag must be boolean")
        if not candidates or len(candidates) > 11 or any(not isinstance(r, ScheduleCandidate) for r in candidates.values()):
            raise TypeError("one bounded surface of static candidates required")
        if ("host" not in candidates or any(n != r.family for n, r in candidates.items())
                or len({(r.environment_id, r.shape) for r in candidates.values()}) != 1):
            raise ValueError("one workload, consistent names, and explicit host required")
        predictions = {n: self._explain(r, allow_extrapolation) for n, r in sorted(candidates.items())}
        latencies = {n: d["predicted_ms"] for n, d in predictions.items() if d["eligible"]}
        if "host" not in latencies:
            raise ValueError("host must remain eligible")
        choice = min(latencies, key=lambda n: (latencies[n], n != "host", n))
        scores = {n: math.log(t/latencies["host"]) for n, t in latencies.items()}
        return {"choice": choice, "predicted_ms": latencies, "scores": scores, "predictions": predictions,
                "reason": "minimum predicted absolute latency with symmetric host/fused comparison",
                "eligible_for_compiler_admission": False}

    def to_dict(self):
        self.validate()
        return {"schema": SCHEMA, **asdict(self), "eligible_for_compiler_admission": False}

    @classmethod
    def from_dict(cls, obj):
        required = {f.name for f in fields(cls)} | {"schema", "eligible_for_compiler_admission"}
        if not isinstance(obj, dict) or set(obj) != required or obj["schema"] != SCHEMA or obj["eligible_for_compiler_admission"] is not False:
            raise ValueError("invalid serialized selector schema")
        try:
            args = {k: v for k, v in obj.items() if k in {f.name for f in fields(cls)}}
            for key in ("host", "fused"):
                item = args[key]
                if not isinstance(item, dict) or set(item) != {f.name for f in fields(LatencyFit)}:
                    raise ValueError("invalid serialized latency fit schema")
                args[key] = LatencyFit(tuple(item["names"]), tuple(item["coefficients"]), tuple(item["scales"]), item["kkt_max_violation"])
            for key in ("training_shapes", "cv_scores", "host_cv_scores", "training_bounds"):
                args[key] = tuple(tuple(v) for v in args[key])
            args["trained_q"] = tuple(args["trained_q"])
            args["grouped_support"] = tuple(GroupSupport(**v) for v in args["grouped_support"])
            result = cls(**args)
            result.validate()
        except (TypeError, KeyError, IndexError, AttributeError) as error:
            raise ValueError("malformed serialized selector") from error
        return result
