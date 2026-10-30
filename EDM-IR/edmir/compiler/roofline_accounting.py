from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .two_stage_selection import ScheduleMeasurement, parse_schedule_family


SCHEMA_ROOFLINE_ACCOUNTING = "edmir.roofline-accounting.v1"
DEFAULT_ELEMENT_BYTES = 2
DEFAULT_THREADBLOCK_M = 64
DEFAULT_THREADBLOCK_N = 128


@dataclass(frozen=True, slots=True)
class HardwareCeilings:
    """Optimistic hardware ceilings used only for lower-bound accounting."""

    tensor_tflops: float = 312.0
    hbm_gbps: float = 2039.0
    lsa_gbps: float = 274.0
    alltoall_gbps: float = 269.82

    def validate(self) -> None:
        if min(
            self.tensor_tflops,
            self.hbm_gbps,
            self.lsa_gbps,
            self.alltoall_gbps,
        ) <= 0.0:
            raise ValueError("roofline ceilings must be positive")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    family: str
    world_size: int
    rows_per_peer: int
    k: int
    n: int
    element_bytes: int = DEFAULT_ELEMENT_BYTES
    measured_ms: float | None = None
    group_id: str = "projection"

    def validate(self) -> None:
        parse_schedule_family(self.family)
        if self.world_size < 2:
            raise ValueError("roofline world size must be at least two")
        if min(self.rows_per_peer, self.k, self.n, self.element_bytes) <= 0:
            raise ValueError("roofline dimensions and element size must be positive")
        if self.measured_ms is not None and self.measured_ms <= 0.0:
            raise ValueError("measured latency must be positive")

    @classmethod
    def from_measurement(cls, row: ScheduleMeasurement) -> "ScheduleSpec":
        return cls(
            family=row.family,
            world_size=row.world_size,
            rows_per_peer=row.rows_per_peer,
            k=row.k,
            n=row.n,
            measured_ms=row.measured_ms,
            group_id=row.group_id,
        )


@dataclass(frozen=True, slots=True)
class RooflineAccount:
    group_id: str
    family: str
    world_size: int
    rows_per_peer: int
    batch_rows_per_rank: int
    batch_rows_global: int
    k: int
    n: int
    element_bytes: int
    split_k_slices: int
    n_tiles_per_cta: int
    rho_n: int
    gamma: int
    flops_per_rank: int
    unique_remote_bytes_per_rank: int
    replayed_remote_bytes_per_rank: int
    local_activation_bytes_per_rank: int
    local_weight_bytes_per_rank: int
    local_output_bytes_per_rank: int
    local_staging_bytes_per_rank: int
    reduction_bytes_per_rank: int
    local_bytes_per_rank: int
    remote_intensity_flop_per_byte: float
    local_intensity_flop_per_byte: float
    compute_floor_ms: float
    remote_floor_ms: float
    hbm_floor_ms: float
    memory_floor_ms: float
    roofline_floor_ms: float
    critical_path_floor_ms: float
    lower_bound_bottleneck: str
    measured_ms: float | None
    achieved_tflops: float | None
    effective_remote_gbps: float | None
    floor_fraction: float | None
    unexplained_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _bytes_to_ms(byte_count: int, bandwidth_gbps: float) -> float:
    return byte_count / (bandwidth_gbps * 1e6)


def _flops_to_ms(flops: int, throughput_tflops: float) -> float:
    return flops / (throughput_tflops * 1e9)


def account_schedule(
    spec: ScheduleSpec,
    *,
    ceilings: HardwareCeilings = HardwareCeilings(),
) -> RooflineAccount:
    """Calculate the fixed compute/memory lower-bound ledger for one candidate.

    All byte counts are per destination rank because the measured latency is a
    synchronized per-iteration latency. ``batch_rows_per_rank=P*M`` is the
    post-All2All batch consumed by one destination; ``batch_rows_global`` is
    the sum across destinations and must not be confused with request batch.
    """

    spec.validate()
    ceilings.validate()
    q, g = parse_schedule_family(spec.family)
    p = spec.world_size
    m = spec.rows_per_peer
    k = spec.k
    n = spec.n
    s = spec.element_bytes
    host = spec.family == "host"

    n_tiles = math.ceil(n / DEFAULT_THREADBLOCK_N)
    rho_n = 1 if host else math.ceil(n_tiles / g)
    batch_rank = p * m
    batch_global = p * batch_rank
    # The retained host executes P sequential per-source GEMMs. Gamma refers
    # to one launched GEMM, while FLOPs/bytes below cover the whole destination.
    gamma = (
        math.ceil(m / DEFAULT_THREADBLOCK_M) * n_tiles
        if host
        else math.ceil(m / DEFAULT_THREADBLOCK_M) * rho_n * q
    )
    flops = 2 * batch_rank * k * n

    unique_remote = (p - 1) * m * k * s
    replayed_remote = unique_remote if host else unique_remote * rho_n

    # Compulsory local bytes for the lower bound. The fused path reads only
    # the destination-local A tile from HBM; other source A tiles are charged
    # to LSA. The host path additionally materializes remote A into staging.
    local_activation = (p * m * k * s) if host else (m * k * s)
    local_weight = k * n * s
    local_output = batch_rank * n * s
    local_staging = unique_remote if host else 0
    # CUTLASS serial split-K writes one extra FP16 output and reads it back for
    # every additional partition: four bytes per output element per boundary.
    reduction = 4 * batch_rank * n * (q - 1) if q > 1 else 0
    local_bytes = (
        local_activation
        + local_weight
        + local_output
        + local_staging
        + reduction
    )

    compute_ms = _flops_to_ms(flops, ceilings.tensor_tflops)
    remote_bandwidth = ceilings.alltoall_gbps if host else ceilings.lsa_gbps
    remote_ms = _bytes_to_ms(replayed_remote, remote_bandwidth)
    hbm_ms = _bytes_to_ms(local_bytes, ceilings.hbm_gbps)
    memory_ms = max(remote_ms, hbm_ms)
    roofline_ms = max(compute_ms, memory_ms)
    # Host staging and GEMM are separate phases in the measured baseline.
    # This is reported as a baseline critical-path floor; the paper's fused
    # headline remains max(T_compute, T_mem).
    critical_path_ms = remote_ms + max(compute_ms, hbm_ms) if host else roofline_ms
    bottleneck = max(
        ((compute_ms, "compute"), (remote_ms, "remote"), (hbm_ms, "hbm")),
        key=lambda item: item[0],
    )[1]

    measured = spec.measured_ms
    achieved_tflops = flops / (measured * 1e9) if measured is not None else None
    effective_remote = (
        replayed_remote / (measured * 1e6) if measured is not None else None
    )
    floor_fraction = critical_path_ms / measured if measured is not None else None
    unexplained = measured - critical_path_ms if measured is not None else None
    return RooflineAccount(
        group_id=spec.group_id,
        family=spec.family,
        world_size=p,
        rows_per_peer=m,
        batch_rows_per_rank=batch_rank,
        batch_rows_global=batch_global,
        k=k,
        n=n,
        element_bytes=s,
        split_k_slices=q,
        n_tiles_per_cta=g,
        rho_n=rho_n,
        gamma=gamma,
        flops_per_rank=flops,
        unique_remote_bytes_per_rank=unique_remote,
        replayed_remote_bytes_per_rank=replayed_remote,
        local_activation_bytes_per_rank=local_activation,
        local_weight_bytes_per_rank=local_weight,
        local_output_bytes_per_rank=local_output,
        local_staging_bytes_per_rank=local_staging,
        reduction_bytes_per_rank=reduction,
        local_bytes_per_rank=local_bytes,
        remote_intensity_flop_per_byte=flops / replayed_remote,
        local_intensity_flop_per_byte=flops / local_bytes,
        compute_floor_ms=compute_ms,
        remote_floor_ms=remote_ms,
        hbm_floor_ms=hbm_ms,
        memory_floor_ms=memory_ms,
        roofline_floor_ms=roofline_ms,
        critical_path_floor_ms=critical_path_ms,
        lower_bound_bottleneck=bottleneck,
        measured_ms=measured,
        achieved_tflops=achieved_tflops,
        effective_remote_gbps=effective_remote,
        floor_fraction=floor_fraction,
        unexplained_ms=unexplained,
    )


__all__ = [
    "DEFAULT_ELEMENT_BYTES",
    "HardwareCeilings",
    "RooflineAccount",
    "SCHEMA_ROOFLINE_ACCOUNTING",
    "ScheduleSpec",
    "account_schedule",
]
