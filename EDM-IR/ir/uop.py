from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ops import OpKind, check_op, namespace

DataKind = str

DATA_BUFFER = "buffer"
DATA_PARAM = "param"
DATA_CONST = "const"
ALL_DATA_KINDS = frozenset({DATA_BUFFER, DATA_PARAM, DATA_CONST})

_MISSING = object()


def check_data_kind(kind: str) -> None:
    if kind not in ALL_DATA_KINDS:
        raise ValueError(f"unknown EDM data kind: {kind}")


@dataclass(frozen=True, slots=True)
class TensorDesc:
    """Coarse tensor-like data carried by serving execution.

    EDM-IR does not need a full shape tracker in the first version.  This
    descriptor keeps enough information to reason about data movement and
    readiness without committing to compiler-style tensor semantics.
    """

    name: str | None = None
    kind: DataKind | None = None
    shape: tuple[int | str | None, ...] | None = None
    dtype: str | None = None
    role: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.kind is not None:
            check_data_kind(self.kind)


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
    kind: DataKind = DATA_BUFFER
    location: str | None = None

    def __post_init__(self) -> None:
        check_data_kind(self.kind)


def buffer_ref(data_id: str, access: str = "read", location: str | None = None) -> DataRef:
    return DataRef(data_id=data_id, access=access, kind=DATA_BUFFER, location=location)


def param_ref(data_id: str, access: str = "read", location: str | None = None) -> DataRef:
    return DataRef(data_id=data_id, access=access, kind=DATA_PARAM, location=location)


def const_ref(data_id: str, access: str = "read", location: str | None = None) -> DataRef:
    return DataRef(data_id=data_id, access=access, kind=DATA_CONST, location=location)


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

    def attr(self, key: str, default: Any = None) -> Any:
        """Read an annotation without forcing every key into the UOp schema."""
        value = getattr(self, key, _MISSING)
        if value is not _MISSING and value is not None:
            return value
        return self.attrs.get(key, default)

    def short(self) -> str:
        stream = f" stream={self.stream}" if self.stream is not None else ""
        phase = f" phase={self.phase}" if self.phase is not None else ""
        return f"%{self.id} {self.op} {self.name} [{self.start_us:.3f},{self.end_us:.3f}]us{phase}{stream}"
