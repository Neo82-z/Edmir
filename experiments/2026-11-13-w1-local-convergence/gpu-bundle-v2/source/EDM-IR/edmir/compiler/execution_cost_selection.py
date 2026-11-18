"""Experimental SM80 execution-geometry selector; not compiler admission.

Kernel-derived counts and fitted time ratios are deliberately separate. This
module never treats logical bytes as measured HBM traffic or CTA rounds as
measured occupancy. NumPy is needed only for fitting this optional model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Mapping

from .factorized_selection import GroupSupport, IdentifiedMeasurements, features
from .host_baseline import HostBaseline, PER_SOURCE_CUTLASS
from .relative_selection import ScheduleCandidate
from .splitk_revision import split_features
from .two_stage_selection import _validate_geometry

GROUP_NAMES = ("grouped_indicator", "grouped_log_K_over_4096", "grouped_log_q",
               "grouped_log_plain_grid_over_64", "grouped_log_q_x_log_plain_grid_over_64")
EXECUTION_NAMES = (
    "ideal_round_work_log_ratio", "log_q", "log_q_x_log_K_over_4096",
    "log_q_x_log_actual_grid_over_64", "log_q_x_log_valid_output_tile_fraction",
    "extra_ctas_over_108", "extra_serial_tile_elements_over_131072",
    "log1p_extra_output_bytes_over_tensor_footprint", "inverse_K_iteration_depth_delta",
)
LEGACY_NAMES = ("log_wave_work_ratio", "log_q", "extra_partitions_over_16",
                "inverse_k_tile_depth_delta", "log_q_x_log_K_over_4096",
                "log_q_x_log_plain_grid_over_64")


def ceildiv(a, b):
    return (a+b-1)//b


@dataclass(frozen=True, slots=True)
class ExecutionGeometry:
    """Per source-destination GEMM; excludes unknown service times and caching."""
    tile_m: int
    tile_n: int
    threads_per_cta: int
    output_tiles: int
    total_ctas: int
    extra_ctas: int
    total_k_iterations: int
    partition_k_iterations: int
    partition_k_span: int
    ordered_epilogue_stages: int
    extra_serial_tile_elements: int
    logical_output_write_bytes: int
    logical_output_read_bytes: int
    extra_output_bytes: int
    tensor_footprint_bytes: int
    valid_output_tile_fraction: float

    @classmethod
    def from_candidate(cls, row):
        _validate_geometry(row)
        if row.is_host:
            raise ValueError("fused geometry cannot describe a host implementation")
        m, k, n, q, g = row.rows_per_peer, row.k, row.n, row.split_k_slices, row.n_tiles_per_cta
        tile_m, tile_n = 64, 128*g
        tiles = ceildiv(m, tile_m)*ceildiv(n, tile_n)
        iterations = ceildiv(k, 32)
        partition = ceildiv(iterations, q)
        # beta=0 on partition 0; later partitions read and accumulate D.
        writes, reads = 2*m*n*q, 2*m*n*(q-1)
        return cls(tile_m, tile_n, 128*g, tiles, tiles*q, tiles*(q-1),
                   iterations, partition, partition*32, q,
                   (q-1)*tile_m*tile_n, writes, reads, writes+reads-2*m*n,
                   2*(m*k+k*n+m*n), m*n/(tiles*tile_m*tile_n))


def group_vector(row):
    """Keep K and q independent; their ratio alone aliases observed regimes."""
    if row.n_tiles_per_cta != 2:
        return (0.,)*len(GROUP_NAMES)
    log_q = math.log2(row.split_k_slices)
    plain_grid = ceildiv(row.rows_per_peer, 64)*ceildiv(row.n, 128)
    log_grid = math.log2(plain_grid/64)
    return (1., math.log2(row.k/4096), log_q, log_grid, log_q*log_grid)


def execution_vector(row):
    geo = ExecutionGeometry.from_candidate(row)
    log_q = math.log2(row.split_k_slices)
    base_rounds, split_rounds = ceildiv(geo.output_tiles, 108), ceildiv(geo.total_ctas, 108)
    return (
        math.log2(split_rounds*geo.partition_k_iterations/(base_rounds*geo.total_k_iterations)),
        log_q, log_q*math.log2(row.k/4096), log_q*math.log2(geo.output_tiles/64),
        log_q*math.log2(geo.valid_output_tile_fraction), geo.extra_ctas/108,
        geo.extra_serial_tile_elements/131072,
        math.log2(1+geo.extra_output_bytes/geo.tensor_footprint_bytes),
        1/geo.partition_k_iterations-1/geo.total_k_iterations,
    )


def feature_names(execution_terms):
    return ("intercept", "log_M", "log_K", "log_N",
            *(EXECUTION_NAMES if execution_terms else LEGACY_NAMES), *GROUP_NAMES)


def feature_vector(row, *, execution_terms):
    _validate_geometry(row)
    if row.is_host:
        return (0.,)*len(feature_names(execution_terms))
    if execution_terms:
        split = execution_vector(row)
    else:
        log_q = math.log2(row.split_k_slices)
        grid = ceildiv(row.rows_per_peer, 64)*ceildiv(row.n, 128)
        split = (*split_features(row, "waves"), log_q*math.log2(row.k/4096),
                 log_q*math.log2(grid/64))
    result = (1., *features(row, "gate"), *split, *group_vector(row))
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite geometry feature")
    return result


def training_pairs(surface, balanced_host):
    """Total mass one per workload; balanced mode fixes host mass at one half."""
    rows = sorted(surface.values(), key=lambda r: r.family)
    fused = [r for r in rows if not r.is_host]
    if not balanced_host:
        weight = 1/math.comb(len(rows), 2)
        return [(a, b, weight) for a, b in combinations(rows, 2)]
    if len(fused) < 2:
        return [(r, surface["host"], 1/len(fused)) for r in fused]
    return ([(r, surface["host"], .5/len(fused)) for r in fused]
            + [(a, b, .5/math.comb(len(fused), 2)) for a, b in combinations(fused, 2)])


@dataclass(frozen=True, slots=True)
class ExecutionCostSelector:
    execution_terms: bool
    balanced_host: bool
    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    environment_id: str
    baseline_id: str
    training_shapes: tuple[tuple[int, int, int, int], ...]
    trained_q: tuple[int, ...]
    grouped_support: tuple[GroupSupport, ...]
    ridge: float
    design_rank: int
    gradient_max_abs: float

    def validate(self):
        if type(self.execution_terms) is not bool or type(self.balanced_host) is not bool:
            raise ValueError("invalid feature/loss mode")
        if self.names != feature_names(self.execution_terms):
            raise ValueError("feature schema mismatch")
        if any(len(v) != len(self.names) for v in (self.means, self.scales, self.coefficients)):
            raise ValueError("inconsistent fitted vector dimensions")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (*self.means, *self.scales, *self.coefficients, self.ridge, self.gradient_max_abs)):
            raise ValueError("nonfinite or invalid fitted state")
        if self.ridge != 1. or any(v <= 0 for v in self.scales):
            raise ValueError("fixed positive ridge/scales required")
        if self.means[0] != 0. or any(self.means[4:]) or self.scales[0] != 1.:
            raise ValueError("only gate coordinates may be centered")
        if self.baseline_id != PER_SOURCE_CUTLASS.baseline_id or not self.environment_id.strip():
            raise ValueError("unknown training identity")
        if len(set(self.training_shapes)) != len(self.training_shapes) or len(self.training_shapes) < 2:
            raise ValueError("invalid training provenance")
        if any(len(s) != 4 or s[0] != 4 or any(type(v) is not int or v < 1 for v in s)
               for s in self.training_shapes):
            raise ValueError("this revision is restricted to P4")
        if (not self.trained_q or 1 not in self.trained_q or len(set(self.trained_q)) != len(self.trained_q)
                or any(type(q) is not int or q not in (1, 2, 4, 8, 16) for q in self.trained_q)):
            raise ValueError("invalid split support")
        if type(self.design_rank) is not int or not 0 < self.design_rank <= len(self.names):
            raise ValueError("invalid design rank")
        for support in self.grouped_support:
            if (support.m not in {s[1] for s in self.training_shapes} or support.q not in self.trained_q
                    or support.k_min > support.k_max or support.n_min > support.n_max
                    or any(type(v) is not int or v < 1 for v in asdict(support).values())):
                raise ValueError("invalid grouped support")

    @classmethod
    def fit(cls, dataset: IdentifiedMeasurements, *, execution_terms=True, balanced_host=True):
        import numpy as np
        if type(execution_terms) is not bool or type(balanced_host) is not bool:
            raise ValueError("feature/loss flags must be booleans")
        surfaces = dataset.validate()
        if len(surfaces) < 2 or any(r.world_size != 4 for r in dataset.rows):
            raise ValueError("at least two P4 training workloads required")
        for surface in surfaces.values():
            for name, row in surface.items():
                if row.n_tiles_per_cta == 2 and name.removesuffix("_groupedn2") not in surface:
                    raise ValueError("grouped measurements require a same-q ungrouped pair")
        raw = [[feature_vector(r, execution_terms=execution_terms) for r in s.values() if not r.is_host]
               for s in surfaces.values()]
        width = len(raw[0][0])
        means = np.zeros(width)
        means[1:4] = np.mean([np.mean(g, axis=0) for g in raw], axis=0)[1:4]
        scales = np.maximum(np.sqrt(np.mean([
            np.mean((np.asarray(g)-means)**2, axis=0) for g in raw], axis=0)), 1e-9)
        scales[0] = 1.

        def z(row):
            if row.is_host:
                return np.zeros(width)
            return (np.asarray(feature_vector(row, execution_terms=execution_terms))-means)/scales

        x, y = [], []
        for surface in surfaces.values():
            for a, b, weight in training_pairs(surface, balanced_host):
                x.append((z(a)-z(b))*math.sqrt(weight))
                y.append((math.log(a.measured_ms)-math.log(b.measured_ms))*math.sqrt(weight))
        x, y = np.asarray(x), np.asarray(y)
        penalty = np.ones(width)
        penalty[0] = 0.
        coefficients = np.linalg.lstsq(np.vstack((x, np.diag(np.sqrt(penalty)))),
                                        np.r_[y, np.zeros(width)], rcond=None)[0]
        grouped = [r for r in dataset.rows if r.n_tiles_per_cta == 2]
        support = []
        for m, q in sorted({(r.rows_per_peer, r.split_k_slices) for r in grouped}):
            subset = [r for r in grouped if (r.rows_per_peer, r.split_k_slices) == (m, q)]
            support.append(GroupSupport(m, q, min(r.k for r in subset), max(r.k for r in subset),
                                        min(r.n for r in subset), max(r.n for r in subset), len(subset)))
        result = cls(execution_terms, balanced_host, feature_names(execution_terms),
                     tuple(means.tolist()), tuple(scales.tolist()), tuple(coefficients.tolist()),
                     dataset.environment_id, dataset.baseline.baseline_id,
                     tuple(sorted({r.shape for r in dataset.rows})),
                     tuple(sorted({r.split_k_slices for r in dataset.rows if r.n_tiles_per_cta == 1})),
                     tuple(support), 1., int(np.linalg.matrix_rank(x)),
                     float(np.max(np.abs(x.T@(x@coefficients-y)+penalty*coefficients))))
        result.validate()
        return result

    def transformed(self, row):
        if row.is_host:
            return (0.,)*len(self.names)
        return tuple((v-m)/s for v, m, s in zip(
            feature_vector(row, execution_terms=self.execution_terms), self.means, self.scales))

    def explain(self, candidate, *, baseline: HostBaseline):
        self.validate()
        baseline.validate()
        if baseline.baseline_id != self.baseline_id:
            raise ValueError("host denominator differs from fitted baseline")
        if not isinstance(candidate, ScheduleCandidate):
            raise TypeError("inference requires static candidates without timing labels")
        candidate.validate()
        if candidate.environment_id != self.environment_id or candidate.world_size != 4:
            raise ValueError("environment/world-size transfer is not supported")
        reason = None
        if candidate.split_k_slices not in self.trained_q:
            reason = "split factor absent from training"
        elif candidate.n_tiles_per_cta == 2 and not any(s.covers(candidate) for s in self.grouped_support):
            reason = "no paired grouped support at exact M/q within observed K/N bounds"
        if reason:
            return {"eligible": False, "reason": reason, "log_ratio": None}
        terms = {n: c*x for n, c, x in zip(self.names, self.coefficients, self.transformed(candidate))}
        score = sum(terms.values())
        if not math.isfinite(score):
            raise ValueError("nonfinite score")
        return {"eligible": True, "log_ratio": score, "contributions": terms,
                "geometry": None if candidate.is_host else asdict(ExecutionGeometry.from_candidate(candidate))}

    def select(self, candidates: Mapping[str, ScheduleCandidate], *, baseline: HostBaseline):
        if not candidates or any(not isinstance(r, ScheduleCandidate) for r in candidates.values()):
            raise TypeError("only static candidates are accepted")
        if any(name != row.family for name, row in candidates.items()):
            raise ValueError("candidate identity mismatch")
        if "host" not in candidates or len({(r.environment_id, r.shape) for r in candidates.values()}) != 1:
            raise ValueError("one workload and explicit host are required")
        detail = {n: self.explain(r, baseline=baseline) for n, r in candidates.items()}
        scores = {n: d["log_ratio"] for n, d in detail.items() if d["eligible"]}
        best = min((n for n in scores if n != "host"), key=lambda n: (scores[n], n), default=None)
        chosen = best if best is not None and scores[best] < 0. else "host"
        return {"choice": chosen, "scores": scores, "predictions": detail,
                "reason": "minimum fitted relative cost; no accuracy guarantee",
                "eligible_for_compiler_admission": False}

    def to_dict(self):
        self.validate()
        return {"schema": "edmir.execution-cost-selector.v1", **asdict(self),
                "eligible_for_compiler_admission": False}


def direction_diagnostics(times, choice, eligible, *, ratio=1.05):
    """Scoring-only: symmetric host/fused errors, never an inference input."""
    if not math.isfinite(ratio) or ratio <= 1 or not times or "host" not in times:
        raise ValueError("positive comparison margin and host measurements required")
    if any(not math.isfinite(t) or t <= 0 for t in times.values()):
        raise ValueError("all scoring times must be finite and positive")
    if choice not in times or choice not in eligible or any(n not in times for n in eligible):
        raise ValueError("unknown chosen/eligible candidate")
    fused = [times[n] for n in eligible if n != "host"]
    best_fused = min(fused, default=math.inf)
    return {
        "false_host": choice == "host" and times["host"] > ratio*best_fused,
        "false_fused": choice != "host" and times[choice] > ratio*times["host"],
        "regret_all_observed": times[choice]/min(times.values())-1,
        "regret_eligible": times[choice]/min(times[n] for n in eligible)-1,
        "relative_to_host": times[choice]/times["host"]-1,
    }
