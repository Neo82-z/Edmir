from __future__ import annotations

from typing import Any

from .graph import EDMGraph
from .ops import (
    COMM_ALL_GATHER,
    COMPUTE_OTHER,
    DATA_DEPENDENCY,
    EdgeKind,
    EVENT_RECORD,
    EVENT_WAIT,
    OpKind,
    PHASE_BEGIN,
    PHASE_BOUNDARY,
    PHASE_END,
    STREAM_ORDER,
)
from .uop import DataRef, MovementDesc, TensorDesc, UOp


class EDMBuilder:
    """Small DSL for constructing trace-shaped EDM graphs."""

    def __init__(self, default_phase: str | None = None) -> None:
        self.graph = EDMGraph()
        self.default_phase = default_phase
        self._next_id = 0

    def uop(
        self,
        op: OpKind,
        name: str,
        start_us: float,
        end_us: float,
        **kwargs: Any,
    ) -> UOp:
        kwargs.setdefault("phase", self.default_phase)
        uop = UOp(
            id=self._next_id,
            op=op,
            name=name,
            start_us=start_us,
            end_us=end_us,
            **kwargs,
        )
        self._next_id += 1
        return self.graph.add_uop(uop)

    def edge(self, src: UOp, dst: UOp, kind: EdgeKind, **attrs: Any) -> None:
        self.graph.add_edge(src.id, dst.id, kind, **attrs)

    def sequence(self, *uops: UOp, kind: EdgeKind = STREAM_ORDER, **attrs: Any) -> None:
        for src, dst in zip(uops, uops[1:]):
            self.edge(src, dst, kind, **attrs)

    def phase_begin(self, phase: str, start_us: float = 0.0) -> UOp:
        return self.uop(PHASE_BEGIN, phase, start_us, start_us, phase=phase)

    def phase_end(self, phase: str, end_us: float) -> UOp:
        return self.uop(PHASE_END, phase, end_us, end_us, phase=phase)

    def compute(
        self,
        name: str,
        start_us: float,
        end_us: float,
        op: OpKind = COMPUTE_OTHER,
        **kwargs: Any,
    ) -> UOp:
        return self.uop(op, name, start_us, end_us, **kwargs)

    def movement(
        self,
        name: str,
        start_us: float,
        end_us: float,
        op: OpKind = COMM_ALL_GATHER,
        movement: MovementDesc | None = None,
        **kwargs: Any,
    ) -> UOp:
        return self.uop(op, name, start_us, end_us, movement=movement, **kwargs)

    def comm(
        self,
        name: str,
        start_us: float,
        end_us: float,
        op: OpKind = COMM_ALL_GATHER,
        movement: MovementDesc | None = None,
        **kwargs: Any,
    ) -> UOp:
        return self.movement(name, start_us, end_us, op=op, movement=movement, **kwargs)

    def record_event(self, name: str, ts_us: float, **kwargs: Any) -> UOp:
        return self.uop(EVENT_RECORD, name, ts_us, ts_us, **kwargs)

    def wait_event(self, name: str, start_us: float, end_us: float | None = None, **kwargs: Any) -> UOp:
        return self.uop(EVENT_WAIT, name, start_us, start_us if end_us is None else end_us, **kwargs)

    def bind_phase(self, begin: UOp, end: UOp, *body: UOp) -> None:
        for uop in body:
            self.edge(begin, uop, PHASE_BOUNDARY)
            self.edge(uop, end, PHASE_BOUNDARY)

    def data_edge(self, src: UOp, dst: UOp, data_id: str) -> None:
        self.edge(src, dst, DATA_DEPENDENCY, data_id=data_id)

    def build(self) -> EDMGraph:
        return self.graph


__all__ = [
    "DataRef",
    "EDMBuilder",
    "MovementDesc",
    "TensorDesc",
]
