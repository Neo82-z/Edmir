from __future__ import annotations

from dataclasses import dataclass, replace

from .model import Program
from .target import (
    GEMM_KERNEL_SPLIT_K,
    FusedScheduleContract,
    FusedTrafficEstimate,
    Target,
)


SCHEMA_FUSED_SCHEDULE_SPACE = "edmir.fused_schedule_space.v1"


@dataclass(frozen=True, slots=True)
class FusedSchedulePoint:
    """One statically analyzable point in the fused schedule space."""

    gemm_kernel: str
    n_tiles_per_cta: int
    legal: bool
    execution_ready: bool
    reason: str
    traffic: FusedTrafficEstimate | None = None

    @property
    def family(self) -> str:
        split = GEMM_KERNEL_SPLIT_K.get(self.gemm_kernel, 1)
        base = "direct" if split == 1 else f"splitk{split}"
        return base if self.n_tiles_per_cta == 1 else f"{base}_groupedn{self.n_tiles_per_cta}"

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "gemm_kernel": self.gemm_kernel,
            "n_tiles_per_cta": self.n_tiles_per_cta,
            "legal": self.legal,
            "execution_ready": self.execution_ready,
            "reason": self.reason,
            "traffic": None if self.traffic is None else self.traffic.to_dict(),
        }


def enumerate_fused_schedule_space(
    program: Program,
    base_target: Target,
    *,
    gemm_kernels: tuple[str, ...],
    n_tiles_per_cta: tuple[int, ...] = (1, 2, 4),
) -> tuple[FusedSchedulePoint, ...]:
    """Enumerate direct/split-K x grouped-N without making latency claims.

    Legality failures remain explicit points instead of disappearing from the
    search space.  ``execution_ready`` is independently false for grouped-N
    points until the runtime kernel exists.
    """

    if not gemm_kernels or len(set(gemm_kernels)) != len(gemm_kernels):
        raise ValueError("gemm_kernels must be non-empty and unique")
    if (
        not n_tiles_per_cta
        or any(value <= 0 for value in n_tiles_per_cta)
        or len(set(n_tiles_per_cta)) != len(n_tiles_per_cta)
    ):
        raise ValueError("n_tiles_per_cta must be non-empty, positive, and unique")

    points: list[FusedSchedulePoint] = []
    for kernel in gemm_kernels:
        for grouped_n in n_tiles_per_cta:
            target = replace(
                base_target,
                gemm_kernel=kernel,
                fused_schedule=FusedScheduleContract(grouped_n),
            )
            try:
                traffic = target.estimate_fused_traffic(program)
            except ValueError as error:
                points.append(
                    FusedSchedulePoint(
                        gemm_kernel=kernel,
                        n_tiles_per_cta=grouped_n,
                        legal=False,
                        execution_ready=False,
                        reason=str(error),
                    )
                )
                continue
            execution_ready = target.fused_schedule.runtime_implemented
            points.append(
                FusedSchedulePoint(
                    gemm_kernel=kernel,
                    n_tiles_per_cta=grouped_n,
                    legal=True,
                    execution_ready=execution_ready,
                    reason=(
                        "runtime kernel available"
                        if execution_ready
                        else "grouped-N runtime kernel not implemented"
                    ),
                    traffic=traffic,
                )
            )
    return tuple(points)


__all__ = [
    "FusedSchedulePoint",
    "SCHEMA_FUSED_SCHEDULE_SPACE",
    "enumerate_fused_schedule_space",
]
