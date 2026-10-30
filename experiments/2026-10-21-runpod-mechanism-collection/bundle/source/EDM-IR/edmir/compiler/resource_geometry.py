"""Auditable counts for one SM80 per-source GEMM, without timing claims.

Logical operand loads count valid tensor elements before inter-CTA cache
reuse. They are neither measured fabric/HBM traffic nor unconditional lower
bounds on physical traffic. Aligned K partitions cover disjoint K intervals.
"""
from dataclasses import asdict, dataclass

from .execution_cost_selection import ExecutionGeometry, ceildiv
from .relative_selection import ScheduleCandidate


@dataclass(frozen=True, slots=True)
class ResourceGeometry:
    m_tiles: int
    n_tiles: int
    output_tiles: int
    partition_count: int
    partition_k_iterations: int
    a_working_set_bytes: int
    b_working_set_bytes: int
    logical_a_load_bytes: int
    logical_b_load_bytes: int
    output_write_bytes: int
    extra_output_read_write_bytes: int
    semaphore_slots: int
    serial_epilogue_boundaries_per_tile: int

    @classmethod
    def from_candidate(cls, row: ScheduleCandidate):
        if not isinstance(row, ScheduleCandidate):
            raise TypeError("static candidate required; timing labels are not inputs")
        row.validate()
        if row.is_host:
            raise ValueError("host execution has a separate baseline contract")
        if row.k % (32 * row.split_k_slices):
            raise ValueError("this ledger requires aligned, nonempty K partitions")
        geo = ExecutionGeometry.from_candidate(row)
        m, k, n = row.rows_per_peer, row.k, row.n
        mt, nt = ceildiv(m, geo.tile_m), ceildiv(n, geo.tile_n)
        a, b = 2*m*k, 2*k*n
        return cls(mt, nt, mt*nt, row.split_k_slices,
                   geo.partition_k_iterations, a, b, a*nt, b*mt,
                   geo.logical_output_write_bytes, geo.extra_output_bytes,
                   mt*nt if row.split_k_slices > 1 else 0,
                   row.split_k_slices - 1)

    def to_dict(self):
        return asdict(self)


def axis_interaction(row: ScheduleCandidate) -> float:
    """Zero on the archived M=64 or N=512 axes; no fitted coefficient exists.

    This is a design-identifiability diagnostic, not a latency correction.
    Squaring permits a nonnegative smooth unobserved interaction witness.
    """
    if not isinstance(row, ScheduleCandidate):
        raise TypeError("static candidate required")
    row.validate()
    return ((row.rows_per_peer/64 - 1) * (row.n/512 - 1)) ** 2
