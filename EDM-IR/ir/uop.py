from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ops import OpKind, check_op, namespace


@dataclass(frozen=True, slots=True)
class TensorDesc:
    """Coarse tensor-like data carried by serving execution.

    EDM-IR does not need a full shape tracker in the first version.  This
    descriptor keeps enough information to reason about data movement and
    readiness without committing to compiler-style tensor semantics.
    """

    name: str | None = None
    shape: tuple[int | str | None, ...] | None = None
    dtype: str | None = None
    role: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class MovementDesc:
    """Communication or movement metadata attached to a UOp."""

    kind: str
    backend: str | None = None
    size_bytes: int | None = None
    group: str | None = None
    group_ranks: tuple[int, ...] = ()
    src_ranks: tuple[int, ...] = ()
    dst_ranks: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DataRef:
    """A coarse read/write reference used later by readiness tracking."""

    data_id: str
    access: str
    location: str | None = None


@dataclass(frozen=True, slots=True)
class UOp:
    """A serving-level execution event.

    Unlike tinygrad-style expression UOps, EDM UOps represent observed
    execution.  Identity is the event id, not structural equality.
    """

    id: int
    op: OpKind
    name: str
    start_us: float
    end_us: float

    phase: str | None = None
    request_id: str | None = None
    batch_id: str | None = None
    microbatch_id: str | None = None

    global_rank: int | None = None
    local_rank: int | None = None
    tp_rank: int | None = None
    dp_rank: int | None = None
    pp_rank: int | None = None
    ep_rank: int | None = None

    device: str | None = None
    stream: str | None = None

    tensor: TensorDesc | None = None
    movement: MovementDesc | None = None
    reads: tuple[DataRef, ...] = ()
    writes: tuple[DataRef, ...] = ()

    attrs: dict[str, Any] = field(default_factory=dict)
    raw_events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        check_op(self.op)
        if self.end_us < self.start_us:
            raise ValueError(f"negative duration for uop %{self.id}: {self.start_us}->{self.end_us}")

    @property
    def dur_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def op_class(self) -> str:
        return namespace(self.op)

    def short(self) -> str:
        stream = f" stream={self.stream}" if self.stream is not None else ""
        phase = f" phase={self.phase}" if self.phase is not None else ""
        return f"%{self.id} {self.op} {self.name} [{self.start_us:.3f},{self.end_us:.3f}]us{phase}{stream}"

