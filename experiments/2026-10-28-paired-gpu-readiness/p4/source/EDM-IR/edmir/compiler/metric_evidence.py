"""Keep logical counts, resource bounds and unobserved hardware metrics separate.

Enumeration is independent of the closed-form geometry implementation. Valid
FP16 bytes are counts under the stated tiling, not load instruction/sector/HBM
measurements. Never promote static checks to causal performance attribution.
"""
from __future__ import annotations
import re

from .mechanism_prediction import HOSTS, geometry, detailed_geometry

MAX_ENUMERATED_CTAS = 100000


def enumerate_tiles(shape, family):
    if (len(shape) != 4 or any(type(x) is not int or not 0 < x <= 2**20 for x in shape)
            or shape[0] != 4):
        raise ValueError("bounded P4 shape required")
    _, m, k, n = shape
    match = re.fullmatch(r"splitk(2|4|8)(_groupedn2)?", family)
    if not match:
        raise ValueError("enumeration only supports audited fused tiles")
    q = int(match[1]); width = 256 if match[2] else 128
    if k % (32*q) or n % 8:
        raise ValueError("equal aligned partitions required")
    if ((m+63)//64)*((n+width-1)//width)*q > MAX_ENUMERATED_CTAS:
        raise ValueError("tile enumeration budget exceeded")
    ctas = a = b = output = boundaries = 0
    depths = set()
    for row in range(0, m, 64):
        for col in range(0, n, width):
            mr, nc = min(64, m-row), min(width, n-col)
            for part in range(q):
                first, end = part*k//q, (part+1)*k//q
                depths.add(len(range(first, end, 32)))
                ctas += 1; a += 2*mr*(end-first); b += 2*(end-first)*nc
                if part:
                    output += 4*mr*nc  # FP16 partial output read plus write.
                    boundaries += 1
    return {"ctas":ctas, "logical_A_replay_bytes":a, "logical_B_replay_bytes":b,
            "extra_serial_output_bytes":output, "partition_iterations":next(iter(depths)),
            "ordered_tile_boundaries":boundaries}


def metric_evidence(shape, family):
    geo = geometry(shape, family)
    if family in HOSTS:
        return {"candidate_family":family, "shape":list(shape),
                "execution_counts":{k:geo[k] for k in ("gemm_count", "actual_gemm_M", "remote_A_bytes", "logical_flops")},
                "count_scope":"gemm_count, remote_A_bytes and flops: whole P4 execution; actual_gemm_M: one GEMM",
                "library_internal_geometry":"not inferred for library kernels",
                "timing_identity":"host execute plus synchronize; CPU submission may overlap device work",
                "separately_measured_stage_costs":None}
    enum = enumerate_tiles(shape, family)
    detail = detailed_geometry(shape, family)
    expected = {k:geo[k] for k in ("ctas", "logical_A_replay_bytes", "extra_serial_output_bytes", "partition_iterations")}
    expected.update(logical_B_replay_bytes=detail["logical_B_replay_bytes"],
                    ordered_tile_boundaries=detail["output_tiles"]*(geo["q"]-1))
    if enum != expected:
        raise ValueError("independent enumeration disagrees with closed-form counts")
    return {"candidate_family":family, "shape":list(shape),
            "verified_logical_counts":enum, "enumeration_matches":True,
            "scope":"per GEMM; valid-element logical accesses, not memory transactions",
            "static_resource_bounds":{k:v for k,v in detail.items() if "bound" in k},
            "unidentified_hardware_metrics":{"HBM_bytes":None, "NVLink_bytes":None,
                "L2_sectors":None, "achieved_occupancy":None, "semaphore_stall_us":None,
                "CPU_submission_exposure_us":None},
            "validation_level":"counts only; hardware metrics and causal latency not validated"}
