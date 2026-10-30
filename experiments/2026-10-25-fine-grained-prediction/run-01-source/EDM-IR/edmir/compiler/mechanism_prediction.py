"""Small, inspectable latency calibration for one frozen execution context.

Counts are kernel-derived. Coefficients are end-to-end statistical associations,
not identified bandwidths, CPU times, occupancy, or semaphore stall times.
This module has no compiler-admission path and never reads a held-out label.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
import time

SCHEMA = "edmir.mechanism-prediction.v1"
POLICY = "resource_queues_v1"
RIDGE = 1e-6  # Fixed numerical regularization, not a searched parameter.
MAX_ROWS, MAX_SHAPES, MAX_FEATURES = 512, 64, 12
HOSTS = ("host", "host_compiler_cublas", "host_merged_cutlass", "host_merged_cublaslt")
FUSED_NAMES = ("fixed", "partition_depth_64", "grouped_partition_depth_64",
               "ctas_over_108", "unique_A_MiB", "logical_A_replay_MiB",
               "extra_serial_output_MiB")
HOST_NAMES = ("fixed", "remote_A_MiB", "logical_work_Gflop")
DETAIL_VARIANTS = ("launch_detail", "partition_detail", "service_detail")
VARIANTS = ("reduced_counts", "coupled_counts", "cutlass_depth", *DETAIL_VARIANTS)
DETAIL_NAMES = ("dynamic_smem_optins_over_16", "partition_A_MiB", "logical_B_replay_MiB",
                "SM_spread_deficit", "ordered_epilogue_boundaries")


def geometry(shape, family):
    if (len(shape) != 4 or any(type(v) is not int or not 0 < v <= 2**20 for v in shape)
            or shape[0] != 4):
        raise ValueError("only bounded P4 geometry is described")
    p, m, k, n = shape
    if n % 8 or k % 32:
        raise ValueError("FP16 aligned N and K required by this audited domain")
    if family in HOSTS:
        return {"kind": family, "actual_gemm_M": p*m if "merged" in family else m,
                "gemm_count": p if "merged" in family else p*p,
                "remote_A_bytes": 2*p*(p-1)*m*k,
                "logical_flops": 2*p*p*m*n*k, "library_internal_geometry": None}
    match = re.fullmatch(r"splitk(2|4|8)(_groupedn2)?", family)
    if not match:
        raise ValueError("unsupported measured family")
    q, g = int(match[1]), 2 if match[2] else 1
    if k % (32*q):
        raise ValueError("this calibration requires equal, aligned split-K partitions")
    mt, nt = (m+63)//64, (n+128*g-1)//(128*g)
    return {"kind": "fused", "q": q, "g": g, "tile": [64, 128*g, 32],
            "partition_iterations": k//(32*q), "ctas": mt*nt*q,
            "unique_A_bytes": 2*m*k, "logical_A_replay_bytes": 2*m*k*nt,
            "extra_serial_output_bytes": 4*m*n*(q-1),
            "threads_per_cta": 128*g, "shared_storage_bytes": 36864 if g == 1 else 61440,
            "gemm_count": p*p, "queue_edges": 31, "unit_duration_chain": 11,
            "counter_scope": "Per GEMM, except graph counts. Logical bytes, not measured traffic."}


def basis(shape, family, variant="coupled_counts"):
    if variant not in VARIANTS:
        raise ValueError("unknown fixed basis")
    x = geometry(shape, family)
    if x["kind"] != "fused":
        names, values = HOST_NAMES, (1., x["remote_A_bytes"]/2**20, x["logical_flops"]/1e9)
        # CUTLASS has a known K=32 mainloop. Equal MK and MKN do not imply
        # equal depth or equal parallelism. cuBLAS's internal tile is unknown.
        if variant not in ("reduced_counts", "coupled_counts") and family in ("host", "host_merged_cutlass"):
            names += ("cutlass_K_iterations_over_128",)
            values += (((shape[2]+31)//32)/128,)
        return names, values
    values = (1., x["partition_iterations"]/64,
              (x["g"]-1)*x["partition_iterations"]/64,
              x["ctas"]/108, x["unique_A_bytes"]/2**20,
              x["logical_A_replay_bytes"]/2**20, x["extra_serial_output_bytes"]/2**20)
    indices = (0, 1, 3) if variant == "reduced_counts" else range(len(values))
    names, values = tuple(FUSED_NAMES[i] for i in indices), tuple(values[i] for i in indices)
    if variant in DETAIL_VARIANTS:
        p, m, k, n = shape
        # The frozen launcher performs the dynamic-smem opt-in on every GEMM.
        # Opt-in and g2 resource changes cannot be causally separated here.
        extra = (p*p*int(x["shared_storage_bytes"] >= 48*1024)/16,
                 x["unique_A_bytes"]/x["q"]/2**20,
                 2*k*n*((m+63)//64)/2**20,
                 max(1-x["ctas"]/108, 0.), x["q"]-1.)
        size = {"launch_detail": 1, "partition_detail": 3, "service_detail": 5}[variant]
        names, values = names+DETAIL_NAMES[:size], values+extra[:size]
    return names, values


def detailed_geometry(shape, family):
    """Static bounds for the archived SM80 binary, never achieved occupancy.

    REG counts are from the frozen cuobjdump, shared sizes from type introspection.
    Largest SMEM carveout and raw register counts give an optimistic upper bound;
    allocation granularity, actual carveout, other kernels and scheduling can
    reduce it. All these g-only quantities are confounded in this dataset.
    """
    x = geometry(shape, family)
    if x["kind"] != "fused":
        return {"host_identity": family, "fused_resource_bounds": None}
    p, m, k, n = shape
    threads, shared = x["threads_per_cta"], x["shared_storage_bytes"]
    registers = 140 if x["g"] == 1 else 134
    warps = threads//32
    register_bound = 65536//(registers*threads)
    smem_bound = (164*1024)//(shared+1024)
    resident_bound = min(register_bound, smem_bound, 64//warps, 32)
    mt, nt = (m+63)//64, (n+128*x["g"]-1)//(128*x["g"])
    return {"m_tiles": mt, "n_tiles": nt, "output_tiles": mt*nt,
        "partition_k_elements": k//x["q"],
        "partition_A_bytes": 2*m*k//x["q"], "partition_B_bytes": 2*k*n//x["q"],
        "max_valid_A_bytes_per_CTA": 2*min(m,64)*k//x["q"],
        "max_valid_B_bytes_per_CTA": 2*min(n,128*x["g"])*k//x["q"],
        "logical_B_replay_bytes": 2*k*n*mt,
        "semaphore_slots": mt*nt, "ordered_boundaries_per_output_tile": x["q"]-1,
        "registers_per_thread": registers, "warps_per_CTA": warps,
        "raw_registers_per_CTA": registers*threads,
        "register_only_resident_block_upper_bound": register_bound,
        "maximum_smem_resident_block_upper_bound": smem_bound,
        "combined_resident_block_upper_bound": resident_bound,
        "possible_active_SM_upper_bound": min(x["ctas"],108),
        "SM_spread_deficit_proxy": max(1-x["ctas"]/108,0.),
        "grid_warps": x["ctas"]*warps,
        "resource_capacity_round_lower_bound": math.ceil(x["ctas"]/(108*resident_bound)),
        "dynamic_smem_optin_calls_per_execution": p*p*int(shared >= 48*1024),
        "cudaGetLastError_calls_per_execution": p*p,
        "achieved_occupancy": None, "actual_smem_carveout": None,
        "scope": "Per GEMM unless named per_execution; optimistic resource bounds only."}


def linear_solve(a, b):
    """Partial-pivot elimination; dimension is at most twelve."""
    n = len(b)
    a = [list(row)+[v] for row, v in zip(a, b)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        if abs(a[pivot][j]) < 1e-12:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v/scale for v in a[j]]
        for i in range(n):
            if i != j:
                c = a[i][j]
                a[i] = [u-c*v for u, v in zip(a[i], a[j])]
    return [row[-1] for row in a]


def matrix_rank(rows, tolerance=1e-9):
    a = [list(row) for row in rows]
    rank = 0
    for j in range(len(a[0])):
        if rank == len(a):
            break
        pivot = max(range(rank, len(a)), key=lambda i: abs(a[i][j]))
        if abs(a[pivot][j]) <= tolerance:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][j]
        a[rank] = [v/scale for v in a[rank]]
        for i in range(rank+1, len(a)):
            c = a[i][j]
            a[i] = [u-c*v for u, v in zip(a[i], a[rank])]
        rank += 1
    return rank


def nonnegative_fit(raw, targets, weights, *, timeout_seconds=10.):
    """Enumerate <=4096 faces of a strictly convex nonnegative quadratic.

    Unit-RMS scaling and weights use training rows only. A KKT certificate is
    checked against the full objective, including fixed ridge. No optimizer
    convergence loop, random seed, or hyperparameter search is present.
    """
    if (not raw or len(raw) > MAX_ROWS or not 1 <= len(raw[0]) <= MAX_FEATURES
            or len(raw) != len(targets) or len(raw) != len(weights)):
        raise ValueError("invalid/bounded fit dimensions")
    d = len(raw[0])
    if any(len(row) != d for row in raw):
        raise ValueError("ragged feature matrix")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in [v for row in raw for v in row]+list(targets)+list(weights)):
        raise ValueError("finite nonnegative fit values required")
    if any(v <= 0 for v in (*targets, *weights)):
        raise ValueError("positive targets and weights required")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("finite positive timeout required")
    deadline = time.monotonic()+timeout_seconds
    scales = [max(math.sqrt(math.fsum(row[j]**2 for row in raw)/len(raw)), 1e-12) for j in range(d)]
    a = [[v/scales[j]*math.sqrt(w) for j, v in enumerate(row)] for row, w in zip(raw, weights)]
    y = [v*math.sqrt(w) for v, w in zip(targets, weights)]
    gram = [[math.fsum(row[i]*row[j] for row in a)+(RIDGE if i == j else 0.)
             for j in range(d)] for i in range(d)]
    rhs = [math.fsum(row[j]*v for row, v in zip(a, y)) for j in range(d)]
    best = None
    for mask in range(1 << d):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded calibration deadline exceeded")
        active = [j for j in range(d) if mask & (1 << j)]
        solution = linear_solve([[gram[i][j] for j in active] for i in active], [rhs[i] for i in active])
        if solution is None or any(v < -1e-8 for v in solution):
            continue
        z = [0.]*d
        for j, v in zip(active, solution):
            z[j] = max(0., v)
        residual = [math.fsum(v*c for v, c in zip(row, z))-t for row, t in zip(a, y)]
        objective = math.fsum(v*v for v in residual)+RIDGE*math.fsum(v*v for v in z)
        if best is None or objective < best[0]:
            best = (objective, z)
    if best is None:
        raise ValueError("no feasible fit")
    objective, z = best
    grad = [math.fsum(gram[i][j]*z[j] for j in range(d))-rhs[i] for i in range(d)]
    violation = max(abs(v) if c > 1e-8 else max(-v, 0.) for v, c in zip(grad, z))
    relative = violation/max(1., max(abs(v) for v in rhs))
    if relative > 1e-8:
        raise ValueError("nonnegative solve failed KKT check")
    return {"coefficients_us": [c/s for c, s in zip(z, scales)], "training_scales": scales,
            "objective_us2": objective, "kkt_relative_violation": relative,
            "unregularized_rank": matrix_rank([[v/s for v, s in zip(row, scales)] for row in raw]),
            "feature_count": d, "enumerated_faces": 1 << d, "ridge": RIDGE}


@dataclass(frozen=True)
class MechanismModel:
    variant: str
    context_id: str
    models: dict
    training_config_ids: tuple[str, ...]
    training_shapes: tuple[tuple[int, ...], ...]
    quality_mode: str = "stable_only"

    @classmethod
    def fit(cls, configs, *, variant="coupled_counts", include_flagged=False):
        if not configs or len(configs) > MAX_ROWS or variant not in VARIANTS:
            raise ValueError("invalid bounded calibration data")
        configs = sorted(configs, key=lambda r: r["config_id"])
        ids = [r["config_id"] for r in configs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate calibration configuration")
        contexts = {r["context_id"] for r in configs}
        if len(contexts) != 1 or not next(iter(contexts)):
            raise ValueError("execution contexts cannot be pooled")
        if type(include_flagged) is not bool:
            raise ValueError("explicit sensitivity flag required")
        if any((not r["eligible_for_calibration"] and not include_flagged)
               or (r.get("quality_flags") and not include_flagged)
               or not r["primary_scope"] or r["warmup"] != 20
               or r["dependency_policy"] != (None if r["family"] in HOSTS else POLICY)
               or r["metadata_mode"] != ("merged" if r["family"].startswith("host_merged_") else "prepared")
               for r in configs):
            raise ValueError("quality, policy, or protocol mismatch")
        shapes = tuple(sorted({tuple(r["shape"]) for r in configs}))
        if not 2 <= len(shapes) <= MAX_SHAPES:
            raise ValueError("calibration needs 2..64 complete shape groups")
        by_kind = {}
        for r in configs:
            kind = geometry(r["shape"], r["family"])["kind"]
            by_kind.setdefault(kind, []).append(r)
        models = {}
        for kind, rows in sorted(by_kind.items()):
            count = Counter(tuple(r["shape"]) for r in rows)
            if len(count) < 2:
                continue
            vectors = [basis(r["shape"], r["family"], variant)[1] for r in rows]
            fit = nonnegative_fit(vectors, [r["observed_us"] for r in rows],
                                  [1/count[tuple(r["shape"])] for r in rows])
            models[kind] = {**fit, "names": basis(rows[0]["shape"], rows[0]["family"], variant)[0],
                "training_shapes": sorted(count), "config_ids": [r["config_id"] for r in rows],
                "families": sorted({r["family"] for r in rows}),
                "family_shapes": {f: sorted({tuple(r["shape"]) for r in rows if r["family"] == f})
                                  for f in sorted({r["family"] for r in rows})},
                "feature_bounds": [[min(v[j] for v in vectors), max(v[j] for v in vectors)]
                                   for j in range(len(vectors[0]))]}
        return cls(variant, next(iter(contexts)), models, tuple(ids), shapes,
                   "all_primary_sensitivity" if include_flagged else "stable_only")

    def predict(self, shape, family, *, context_id, dependency_policy=None):
        expected_policy = None if family in HOSTS else POLICY
        policy = expected_policy if dependency_policy is None else dependency_policy
        if context_id != self.context_id or policy != expected_policy:
            raise ValueError("new source/environment/policy requires separately qualified evidence")
        geo = geometry(shape, family)
        if geo["kind"] not in self.models:
            raise ValueError("no calibrated implementation family")
        fit = self.models[geo["kind"]]
        names, values = basis(shape, family, self.variant)
        terms = {name: value*c for name, value, c in zip(names, values, fit["coefficients_us"])}
        estimate = math.fsum(terms.values())
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError("invalid latency prediction")
        extrapolation = [name for name, value, (lo, hi) in zip(names, values, fit["feature_bounds"])
                         if value < lo-1e-12 or value > hi+1e-12]
        return {"predicted_us": estimate, "contributions_us": terms, "geometry": geo,
                "execution_detail": detailed_geometry(shape, family),
                "feature_extrapolation": extrapolation,
                "family_seen": family in fit["families"],
                "shape_seen": list(shape) in [list(s) for s in fit["training_shapes"]],
                "coefficient_identification": "rank_deficient" if fit["unregularized_rank"] < fit["feature_count"] else "full_column_rank_not_causal_identification",
                "training_shape_count_for_implementation": len(fit["training_shapes"]),
                "training_shapes_for_exact_candidate_family": fit["family_shapes"].get(family, []),
                "library_algorithm_prediction": "not modeled separately" if family in ("host_compiler_cublas", "host_merged_cublaslt") else "fixed audited kernel family",
                "compiler_admission": False,
                "physical_bandwidth_or_stall_attribution": None}

    def to_dict(self):
        return {"schema": SCHEMA, **asdict(self), "compiler_admission": False}


def select_prediction(predictions):
    """Deterministic tie rule; no dependence on traversal order."""
    if not predictions or len({r["candidate_id"] for r in predictions}) != len(predictions):
        raise ValueError("nonempty, unique candidates required")
    if any(not math.isfinite(r["predicted_us"]) or r["predicted_us"] <= 0 for r in predictions):
        raise ValueError("invalid candidate latency")
    return min(predictions, key=lambda r: (r["predicted_us"], r["candidate_id"]))["candidate_id"]


def conditional_policy_latency(old_us, nonscaling_fraction, service_multiplier):
    """A declared scenario, NOT a calibrated forecast or speedup guarantee.

    T' = T * [a + (1-a) * (4/11) * b]. a and b are unmeasured.
    Uniform kernel durations, negligible new gaps, and the same fixed cost
    are additional assumptions. b may exceed 1 under added concurrency.
    """
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in
                (old_us, nonscaling_fraction, service_multiplier)) or old_us <= 0
            or not 0 <= nonscaling_fraction <= 1 or service_multiplier <= 0):
        raise ValueError("invalid explicit scenario")
    return old_us*(nonscaling_fraction+(1-nonscaling_fraction)*4/11*service_multiplier)


def critical_path_us(ancestors, durations):
    """Recompute a DAG longest path, including ties; no fixed-path subtraction."""
    if set(ancestors) != set(durations) or len(durations) > 4096:
        raise ValueError("incomplete/bounded DAG required")
    if any(not math.isfinite(v) or v < 0 for v in durations.values()):
        raise ValueError("nonnegative finite durations required")
    remaining, finish = set(durations), {}
    while remaining:
        ready = sorted(i for i in remaining if set(ancestors[i]) <= finish.keys())
        if not ready:
            raise ValueError("cyclic or missing dependency")
        for i in ready:
            finish[i] = max((finish[j] for j in ancestors[i]), default=0.)+durations[i]
            remaining.remove(i)
    return max(finish.values(), default=0.)
