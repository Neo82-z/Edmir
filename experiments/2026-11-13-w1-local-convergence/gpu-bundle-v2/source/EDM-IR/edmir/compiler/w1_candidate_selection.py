"""W1 development selector: evidence admission precedes shared scoring.

This module never opens files or calls a device. Importers must verify raw
receipts before constructing evidence. Digests bind records, not prove them.
The bounded ridge comparison is a diagnostic, not an admitted compiler model.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import time

from . import shape_decision_model as base
from .two_stage_selection import parse_schedule_family

SCHEMA = "edmir.w1-shared-score.v1"
ERROR_CONTRACT = "max_abs_over_max_1_abs_ref_le_0.02_v1"
HOSTS = ("host", "host_merged_cutlass", "host_merged_cublaslt")
EXTRA = ("log_warps_per_cta", "log_grid_warps_per_sm", "log_staging_lower_bound_bytes",
         "log_serial_feedback_bytes", "log_k_depth_times_reference_rounds",
         "log_k_depth_over_reference_fill", "log_q_times_warps_per_cta",
         "log_logical_A_replay_bytes", "log_logical_B_replay_bytes",
         "log_host_nccl_groups", "log_program_gemm_launches",
         "log_fp32_partial_buffer_bytes", "log_final_conversion_launches")
CONTRACT_FIELDS = ("shape", "candidate_id", "execution_context", "implementation_id",
                   "precision", "error_contract", "phase", "source_id")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def axes(family):
    original = family.removesuffix("_fp32reduce")
    if original in HOSTS:
        if original != family:
            raise ValueError("host cannot borrow fused reduction identity")
        return 1, 1
    q, g = parse_schedule_family(original)
    if original != family and q == 1:
        raise ValueError("FP32 reduction needs q > 1")
    return q, g


def precision(family):
    q, _ = axes(family)
    opaque = family == "host_merged_cublaslt"
    return dict(input="fp16", weight="fp16", output="fp16", accumulator="fp32",
                accumulator_evidence="api_requested" if opaque else "source",
                reduction=("library_unknown" if opaque else "single_output_cast" if q == 1
                           else "serial_fp32_then_fp16" if family.endswith("_fp32reduce")
                           else "serial_fp16_feedback"),
                partial=("unknown" if opaque else "none" if q == 1
                         else "fp32" if family.endswith("_fp32reduce") else "fp16"))


def contract(row):
    p, m, k, n = row["shape"]
    if any(type(v) is not int or v <= 0 for v in (p, m, k, n)) or p != 8:
        raise ValueError("W1 requires explicit positive P=8,M,K,N")
    if row["precision"] != precision(row["candidate_id"]):
        raise ValueError("unsupported dtype/reduction: no FP16 labels for BF16 or new storage")
    if row["phase"] != "measure" or row["error_contract"] != ERROR_CONTRACT:
        raise ValueError("paired-phase or different error contract is not W1 primary scoring")
    if any(not row[k] for k in ("implementation_id", "source_id", "execution_context")):
        raise ValueError("explicit source and execution identity required")
    return digest({key: row[key] for key in CONTRACT_FIELDS})


def admission(row, required_inputs):
    """All prescribed input/reference hashes and full-output checks are required.

    A failure dominates; absence is unknown, never a pass. No shape-wide q ban.
    Required inputs are supplied by the caller's protocol, not inferred from
    the candidate's checks, which could silently omit a failed seed.
    """
    binding = contract(row)
    if not required_inputs or len(required_inputs) != len(set(required_inputs)):
        raise ValueError("nonempty prescribed correctness inputs required")
    records = row.get("checks", [])
    seen, failed = set(), False
    for check in records:
        seed = str(check["seed"])
        if seed in seen or seed not in required_inputs:
            raise ValueError("duplicate or unprescribed correctness seed")
        seen.add(seed)
        if check["contract_sha256"] != binding or not check.get("record_sha256"):
            raise ValueError("correctness evidence bound to another implementation/condition")
        if check["input"] != required_inputs[seed]:
            raise ValueError("input/reference identity mismatch")
        p, m, _, n = row["shape"]
        if check["checked_elements"] != p*p*m*n:
            raise ValueError("sampled checks cannot admit the full output")
        if type(check["passed"]) is not bool:
            raise ValueError("explicit correctness verdict required")
        failed |= not check["passed"]
    if failed:
        return "failed"
    return "passed" if seen == set(required_inputs) else "unknown"


def coupling(row):
    """Source-derived counts only. Packing is not achieved GPU occupancy.

    The FP16 SM80 kernels have 64x(128g)x32 CTAs, 32x64 warps and
    three operand stages. cuBLASLt tiling stays unknown (no CUTLASS proxy).
    Traffic counts are logical requests, not physical HBM/NVLink bytes.
    """
    p, m, k, n = row["shape"]
    family = row["candidate_id"]
    q, g = axes(family)
    sm = row["sm_count"]
    if type(sm) is not int or not 1 <= sm <= 1024:
        raise ValueError("explicit SM count required")
    host = family in HOSTS
    merged = family.startswith("host_merged")
    launches = p if merged else p*p
    program = {"host_nccl_groups": p*(p-1) if host else 0,
               "program_gemm_launches": launches}
    if family == "host_merged_cublaslt":
        return dict(geometry=None, **program)
    gm = p*m if merged else m
    tm, tn, tk = 64, 128*g, 32
    mt, nt = math.ceil(gm/tm), math.ceil(n/tn)
    tiles, ctas = mt*nt, mt*nt*q
    rounds = math.ceil(ctas/sm)
    warps = (tm//32)*(tn//64)
    depth = math.ceil(k/(tk*q))
    # Legacy serial SplitK writes FP16 each partition. FP32 variant writes
    # FP32 per partition and performs a separate final conversion launch.
    partial_bytes = 4 if family.endswith("_fp32reduce") else 2
    geo = dict(q=q, g=g, gemm_m=gm, output_tiles=tiles, ctas=ctas, warps_per_cta=warps,
               grid_warps_per_sm=ctas*warps/sm, k_iterations_per_partition=depth,
               reference_rounds=rounds, reference_fill=ctas/(rounds*sm),
               operand_staging_lower_bound_bytes=3*tk*(tm+tn)*2,
               serial_feedback_bytes=2*(q-1)*gm*n*partial_bytes,
               logical_A_replay_bytes=mt*nt*tm*k*2,
               logical_B_replay_bytes=mt*nt*tn*k*2,
               observed_residency=None, measured_peer_bytes=None,
               partial_buffer_bytes=gm*n*4 if family.endswith("_fp32reduce") else 0,
               conversion_launches=launches if family.endswith("_fp32reduce") else 0)
    return dict(geometry=geo, **program)


def features(row, variant):
    contract(row)
    original = original_geometry_features(row)
    if "features" in row and row["features"] != original:
        raise ValueError("supplied geometry differs from source-derived shape and candidate")
    if variant not in ("original_geometry", "coupled_geometry"):
        raise ValueError("undeclared diagnostic variant")
    result = list(original)
    if variant == "coupled_geometry":
        c = coupling(row); g = c["geometry"]; log = math.log1p
        result += ([log(g["warps_per_cta"]), log(g["grid_warps_per_sm"]),
                    log(g["operand_staging_lower_bound_bytes"]), log(g["serial_feedback_bytes"]),
                    log(g["k_iterations_per_partition"]*g["reference_rounds"]),
                    log(g["k_iterations_per_partition"]/g["reference_fill"]),
                    log(g["q"]*g["warps_per_cta"]), log(g["logical_A_replay_bytes"]),
                    log(g["logical_B_replay_bytes"])] if g else [0.]*9)
        result += [log(c["host_nccl_groups"]), log(c["program_gemm_launches"])]
        result += [log(g["partial_buffer_bytes"]),log(g["conversion_launches"])] if g else [0.,0.]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite features")
    return result


def original_geometry_features(row):
    """The existing 31-dimensional basis under the audited W1 identity.

    Do not pass W1 through the pre-W1 ledger's pinned runtime hashes: W1 has
    shared NCCL groups and revised lifetimes. Geometry equations are retained;
    the importer separately verifies the W1 sealed source/build/raw receipts.
    """
    c=coupling(row); geo=c["geometry"]
    p,m,k,n=row["shape"]; family=row["candidate_id"]; host=family in HOSTS
    merged=family.startswith("host_merged");gm=p*m if merged else m; log=math.log1p
    q,g=axes(family)
    x=[float(host),float(merged),float(geo is None),float(not host),1.,
       log(p),log(m),log(k),log(n),log(row["sm_count"]),
       log(q) if geo else 0.,log(g) if geo else 0.,log(1 if merged else p),
       log(2*gm*k*n),log(2*gm*k),log(2*k*n),log(2*p*(p-1)*m*k)]
    if geo:
        mt,nt=math.ceil(gm/64),math.ceil(n/(128*g));tiles=mt*nt;ctas=tiles*q;sm=row["sm_count"]
        depth=geo["k_iterations_per_partition"]
        x += [log(mt),log(nt),log(tiles),log(ctas),log(depth),gm*n/(tiles*64*128*g),
              ((gm-1)%64+1)/64,((n-1)%(128*g)+1)/(128*g),float(math.ceil(ctas/sm)),
              ((ctas-1)%sm+1)/sm,log(tiles*(q-1)),log(2*tiles*64*128*g*k),
              math.log2(g)*log(depth),math.log2(q)*log(tiles)]
    else:x += [0.]*14
    if len(x)!=len(base.FEATURES):raise AssertionError("feature schema drift")
    return x


def support_key(row):
    return digest({k: row[k] for k in ("candidate_id", "implementation_id", "precision",
                                     "execution_context", "error_contract", "phase", "source_id")})


def fit(rows, required_inputs, *, variant="original_geometry"):
    """One fixed ridge solve, no hyperparameter/seed/epoch search on held-out data.

    Equal shape mass, then equal host/fused mass. Coefficients calibrate log
    elapsed time, not physical service times. Only whole-shape folds are legal.
    """
    import numpy as np
    begin = time.monotonic()
    if not 4 <= len(rows) <= 256:
        raise ValueError("bounded development dataset required")
    grouped = defaultdict(list)
    for row in rows:
        if row.get("split") != "train" or row.get("measurement_kind") != "observed":
            raise ValueError("explicit observed training partition required")
        if admission(row, required_inputs[row["shape_tag"]]) != "passed":
            raise ValueError("failed or unknown candidate has no training label")
        if not math.isfinite(row["latency_us"]) or row["latency_us"] <= 0:
            raise ValueError("positive measured latency required")
        grouped[tuple(row["shape"])].append(row)
    if len(grouped) < 2 or len({(r["execution_context"],r["source_id"]) for r in rows}) != 1:
        raise ValueError("multiple whole shapes in one execution/source identity required")
    ordered, weights = [], []
    for shape in sorted(grouped):
        surface = sorted(grouped[shape], key=lambda r:r["candidate_id"])
        hosts = sum(r["candidate_id"] in HOSTS for r in surface)
        if hosts in (0,len(surface)) or len({r["candidate_id"] for r in surface}) != len(surface):
            raise ValueError("distinct host and fused candidates required")
        ordered += surface
        weights += [.5/len(grouped)/(hosts if r["candidate_id"] in HOSTS else len(surface)-hosts) for r in surface]
    x = np.asarray([features(r,variant) for r in ordered],dtype=np.float64)
    y = np.log([r["latency_us"] for r in ordered]); w = np.asarray(weights)
    means = w@x; scales = np.maximum(np.sqrt(w@((x-means)**2)),.1)
    z = np.column_stack(((x-means)/scales,np.ones(len(x))))
    regularizer = .01*np.eye(z.shape[1]); regularizer[-1,-1] = 0.
    theta = np.linalg.solve(z.T@(w[:,None]*z)+regularizer,z.T@(w*y))
    if not np.isfinite(theta).all():
        raise FloatingPointError("nonfinite shared coefficients")
    return dict(schema=SCHEMA,variant=variant,features=list(base.FEATURES)+(list(EXTRA) if variant=="coupled_geometry" else []),
                means=means.tolist(),scales=scales.tolist(),theta=theta.tolist(),ridge=.01,
                execution_context=rows[0]["execution_context"],source_id=rows[0]["source_id"],
                support=sorted({support_key(r) for r in rows}),training_shapes=[list(s) for s in sorted(grouped)],
                shape_axis_ranges={axis:[min(s[i] for s in grouped),max(s[i] for s in grouped)]
                                   for i,axis in enumerate(("P","M","K","N"))},
                design_rank=int(np.linalg.matrix_rank(z)),coefficient_count=z.shape[1],
                constant_training_features=[name for name,spread in zip(
                    list(base.FEATURES)+(list(EXTRA) if variant=="coupled_geometry" else []),np.ptp(x,axis=0)) if spread==0],
                compiler_admission=False,elapsed_seconds=time.monotonic()-begin)


def predict(model, queries, required_inputs):
    """Labels are neither read nor accepted as a source for missing scores."""
    import numpy as np
    if model["schema"] != SCHEMA:
        raise ValueError("unsupported W1 model")
    if not 1 <= len(queries) <= 32 or len({r["candidate_id"] for r in queries}) != len(queries):
        raise ValueError("bounded, unique candidate surface required")
    if len({tuple(r["shape"]) for r in queries}) != 1:
        raise ValueError("one query shape required")
    scored, excluded = {}, {}
    for row in sorted(queries,key=lambda r:r["candidate_id"]):
        name=row["candidate_id"]
        status=admission(row,required_inputs)
        if status != "passed":
            excluded[name]=status;continue
        if support_key(row) not in model["support"]:
            excluded[name]="unsupported_execution_or_implementation";continue
        x=(np.asarray(features(row,model["variant"]))-model["means"])/model["scales"]
        log_us=float(x@np.asarray(model["theta"][:-1])+model["theta"][-1])
        if not math.isfinite(log_us) or not -20 < log_us < 30:
            raise FloatingPointError("prediction outside finite development range")
        scored[name]=math.exp(log_us)
    chosen=min(scored,key=lambda name:(scored[name],name)) if scored else None
    outside={axis:dict(value=queries[0]["shape"][i],training_range=model["shape_axis_ranges"][axis])
             for i,axis in enumerate(("P","M","K","N"))
             if not model["shape_axis_ranges"][axis][0] <= queries[0]["shape"][i] <= model["shape_axis_ranges"][axis][1]}
    return dict(scores_us=scored,excluded=excluded,chosen=chosen,
                extrapolated_shape_axes=outside,
                recommendation_status="diagnostic_extrapolation" if outside else "development_diagnostic",
                compiler_admission=False,complete_eligible_coverage=not any(
                    v=="unsupported_execution_or_implementation" for v in excluded.values()))
