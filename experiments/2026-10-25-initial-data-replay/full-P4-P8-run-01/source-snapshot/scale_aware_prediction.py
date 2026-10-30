"""Bounded P4/P8 cost hypothesis for the archived FP16 direct-peer program.

This is a retrospective research model, not a compiler admission rule. Counts
describe the archived cyclic resource-queue schedule. Coefficients fitted to
whole-program wall times are associations, not measured hardware service rates.
The newer uniform-peer dependency policy is deliberately a different context.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re

from .host_baseline import PER_SOURCE_CUTLASS
from .measurement_dataset import P4_ENV, P8_ENV
from .mechanism_prediction import nonnegative_fit
from .prediction_support import linear_target_support
from .regime_prediction import max_branch_fit

SCHEMA = "edmir.scale-aware-legacy-prediction.v1"
ENVIRONMENTS = {P4_ENV: 4, P8_ENV: 8}
FAMILIES = ("host", "direct", "splitk2", "splitk4", "splitk8", "splitk16",
            "direct_groupedn2", "splitk2_groupedn2", "splitk4_groupedn2",
            "splitk8_groupedn2", "splitk16_groupedn2")
HOST_NAMES = ("program_GEMMs_over_16", "remote_A_per_rank_MiB",
              "work_per_rank_Gflop", "sequential_K_depth_over_512")
FUSED_NAMES = ("program_GEMMs_over_16", "grouped_chain_depth_over_704",
               "grouped_launches_over_16", "chain_SM_spread_deficit_over_11",
               "chain_epilogue_boundaries_over_11", "chain_extra_output_MiB_over_11",
               "chain_K_depth_over_704", "remote_A_replay_per_rank_MiB_over_3")


@lru_cache(maxsize=2)
def legacy_graph(p):
    """Reproduce queue dependencies of the archived one-chunk fused program.

    Independent verification executes model/schedule/dependence.py from the
    August source archive. The unit-duration chain is a structural proxy only.
    """
    if type(p) is not int or p not in (4, 8):
        raise ValueError("audited graph requires P4 or P8")
    last, depth, edges = {}, {}, set()
    for step in range(p):
        for dst in range(p):
            src = (dst + step) % p
            op = (step, dst)
            resources = {(src, "comm"), (dst, "comm"), (dst, "compute")}
            predecessors = {last[r] for r in resources if r in last}
            edges.update((a, op) for a in predecessors)
            depth[op] = 1 + max((depth[a] for a in predecessors), default=0)
            for resource in resources:
                last[resource] = op
    return len(depth), len(edges), max(depth.values())


def geometry(shape, family):
    if (len(shape) != 4 or any(type(v) is not int or not 0 < v <= 2**20 for v in shape)
            or shape[0] not in (4, 8) or family not in FAMILIES):
        raise ValueError("bounded archived P4/P8 family required")
    p, m, k, n = shape
    if k % 32 or n % 8:
        raise ValueError("aligned FP16 shape required")
    g = 2 if family.endswith("_groupedn2") else 1
    match = re.fullmatch(r"splitk(2|4|8|16)(?:_groupedn2)?", family)
    q = int(match[1]) if match else 1
    if k % (32*q):
        raise ValueError("equal aligned K partitions required")
    mt, nt = (m+63)//64, (n+128*g-1)//(128*g)
    ops, edges, chain = legacy_graph(p)
    return {
        "shape": tuple(shape), "family": family, "q": q, "g": g,
        "rows_per_destination": p*m, "gemms_per_destination": p,
        "program_gemms": p*p, "remote_peers_per_destination": p-1,
        "unique_remote_A_per_rank_bytes": 2*(p-1)*m*k,
        "unique_remote_A_program_bytes": 2*p*(p-1)*m*k,
        "unique_local_A_per_rank_bytes": 2*m*k,
        "logical_flops_per_rank": 2*p*m*n*k,
        "m_tiles": mt, "n_tiles": nt, "ctas_per_gemm": mt*nt*q,
        "K_iterations_per_partition": k//(32*q),
        "logical_A_per_gemm_bytes": 2*m*k*nt,
        "logical_remote_A_per_rank_bytes": 2*(p-1)*m*k*nt,
        "logical_local_A_per_rank_bytes": 2*m*k*nt,
        "logical_B_per_gemm_bytes": 2*k*n*mt,
        "extra_output_per_gemm_bytes": 4*m*n*(q-1),
        "serial_epilogue_boundaries_per_output_tile": q-1,
        "serial_split_K_enabled": q > 1,
        "fused_graph_ops": ops, "fused_queue_edges": edges,
        "fused_unit_duration_chain": chain,
        "HBM_bytes": None, "NVLink_bytes": None, "registers_per_thread": None,
        "achieved_occupancy": None,
        "note": "Logical useful-byte counts, excluding padding/transactions/cache effects. Fused graph fields do not describe host dependencies. No archived binary resource constants are assumed.",
    }


def basis(shape, family):
    x = geometry(shape, family)
    p, _, k, _ = shape
    if family == "host":
        return HOST_NAMES, (p*p/16, x["unique_remote_A_per_rank_bytes"]/2**20,
                            x["logical_flops_per_rank"]/1e9, p*k/(32*512))
    chain = x["fused_unit_duration_chain"]/11
    depth = x["K_iterations_per_partition"]/64
    return FUSED_NAMES, (
        p*p/16, chain*(x["g"]-1)*depth, p*p*(x["g"]-1)/16,
        chain*max(1-x["ctas_per_gemm"]/108, 0.), chain*(x["q"]-1),
        chain*x["extra_output_per_gemm_bytes"]/2**20,
        chain*depth, x["logical_remote_A_per_rank_bytes"]/(3*2**20))


@dataclass(frozen=True)
class ScaleAwareModel:
    environment_id: str
    retained_fraction: float
    models: dict
    training_shapes: tuple
    training_vectors: dict

    @classmethod
    def fit(cls, rows, *, environment_id, retained_fraction=0.):
        if (environment_id not in ENVIRONMENTS or not 1 <= len(rows) <= 512
                or retained_fraction not in (0., .25)):
            raise ValueError("bounded archive context and predeclared rho required")
        rows = sorted(rows, key=lambda r: (r.shape, r.family))
        if len({(r.shape, r.family) for r in rows}) != len(rows):
            raise ValueError("duplicate calibration candidate")
        surfaces = {}
        for row in rows:
            row.validate()
            if row.world_size != ENVIRONMENTS[environment_id]:
                raise ValueError("world size contradicts calibration environment")
            geometry(row.shape, row.family)
            surfaces.setdefault(row.shape, set()).add(row.family)
        if not 2 <= len(surfaces) <= 64 or any("host" not in s or len(s) < 2 for s in surfaces.values()):
            raise ValueError("need 2..64 host/fused calibration surfaces")
        fitted, vectors = {}, {}
        for kind in ("host", "fused"):
            selected = [r for r in rows if (r.family == "host") == (kind == "host")]
            counts = Counter(r.shape for r in selected)
            values = [basis(r.shape, r.family)[1] for r in selected]
            targets = [1000*r.measured_ms for r in selected]
            weights = [1/counts[r.shape] for r in selected]
            result = (nonnegative_fit(values, targets, weights) if kind == "host" else
                      max_branch_fit(values, targets, weights, retained_fraction=retained_fraction))
            fitted[kind] = {**result, "names": HOST_NAMES if kind == "host" else FUSED_NAMES}
            vectors[kind] = tuple(values)
        return cls(environment_id, retained_fraction, fitted, tuple(sorted(surfaces)), vectors)

    def predict(self, shape, family, *, environment_id, allow_transfer=False):
        if (environment_id not in ENVIRONMENTS or ENVIRONMENTS[environment_id] != shape[0]
                or type(allow_transfer) is not bool
                or (environment_id != self.environment_id and not allow_transfer)):
            raise ValueError("explicit matching environment or transfer request required")
        kind = "host" if family == "host" else "fused"
        names, values = basis(shape, family)
        terms = [v*c for v, c in zip(values, self.models[kind]["coefficients_us"])]
        predicted = (math.fsum(terms) if kind == "host" else
                     math.fsum(terms[:-2])+max(terms[-2:])+self.retained_fraction*min(terms[-2:]))
        return {"predicted_us": predicted, "feature_values": dict(zip(names, values)),
                "potential_terms_us": dict(zip(names, terms)),
                "linear_support": linear_target_support(self.training_vectors[kind], values),
                "environment_transfer": environment_id != self.environment_id,
                "compiler_admission": False}

    def rank(self, shape, families, *, environment_id, allow_transfer=False):
        families = tuple(families)
        if not families or len(families) > 11 or len(set(families)) != len(families):
            raise ValueError("unique bounded candidate names required")
        forecasts = {f: self.predict(shape, f, environment_id=environment_id,
                                    allow_transfer=allow_transfer) for f in families}
        return sorted(families, key=lambda f: (forecasts[f]["predicted_us"], f)), forecasts

    def to_dict(self):
        return {"schema": SCHEMA, "environment_id": self.environment_id,
                "host_identity": PER_SOURCE_CUTLASS.to_dict(),
                "dependency_policy": "resource_queues_v1",
                "retained_fraction": self.retained_fraction, "models": self.models,
                "training_shapes": self.training_shapes,
                "training_vectors": self.training_vectors, "compiler_admission": False,
                "limitations": ["One-P calibration cannot identify cross-P scaling exponents.",
                    "Unit-duration graph depth is not a timed critical path.",
                    "Grouped launch count also confounds grouped resource differences.",
                    "Remote useful-byte branch omits separately fitted local/B/transaction effects.",
                    "Split-K serial on/off may change code generation beyond these counts.",
                    "Historical development only; no independent validation or measured physical rates."]}
