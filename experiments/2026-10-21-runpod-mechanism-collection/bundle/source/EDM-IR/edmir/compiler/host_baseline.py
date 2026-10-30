"""Explicit host identities. Equal names do not imply equal implementations.

Library/build versions and hardware belong to the environment fingerprint;
these contracts identify the algorithm, partitioning, and timing boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

WALL_TIMER = "host steady_clock around backend.execute() plus backend.synchronize()"


@dataclass(frozen=True, slots=True)
class HostBaseline:
    baseline_id: str
    gemm_provider: str
    gemm_kernel: str
    partition: str
    communication: str = "host_nccl_send_recv_and_local_copy"
    schedule: str = "materialize_then_gemm"
    dtype: str = "float16"
    layout: str = "row_major"
    timing_scope: str = WALL_TIMER

    def validate(self):
        if HOST_BASELINES.get(self.baseline_id) != self:
            raise ValueError("unknown or inconsistent host baseline identity")

    def to_dict(self):
        self.validate()
        return asdict(self)


PER_SOURCE_CUTLASS = HostBaseline(
    "host_nccl_cutlass_sm80_per_source_v1", "cutlass", "sm80_64x128x32", "P_sequential_M_gemms")
PER_SOURCE_CUBLAS = HostBaseline(
    "host_nccl_cublas_per_source_v1", "cublas", "library_default", "P_sequential_M_gemms")
# Contracts for required new comparators, not claims that these were measured
# or that their collectors are already implemented.
MERGED_CUTLASS = HostBaseline(
    "host_nccl_cutlass_sm80_merged_v1", "cutlass", "sm80_64x128x32", "one_PM_gemm")
MERGED_CUBLASLT = HostBaseline(
    "host_nccl_cublaslt_merged_v1", "cublaslt", "recorded_algorithm", "one_PM_gemm")
HOST_BASELINES = {b.baseline_id: b for b in (
    PER_SOURCE_CUTLASS, PER_SOURCE_CUBLAS, MERGED_CUTLASS, MERGED_CUBLASLT)}


def host_baseline(baseline_id: str) -> HostBaseline:
    try:
        return HOST_BASELINES[baseline_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("host baseline must have a registered implementation identity") from exc
