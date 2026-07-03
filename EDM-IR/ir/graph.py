from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .ops import EdgeKind, check_edge_kind
from .uop import UOp


@dataclass(frozen=True, slots=True)
class Edge:
    src: int
    dst: int
    kind: EdgeKind
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_edge_kind(self.kind)


@dataclass
class EDMGraph:
    uops: dict[int, UOp] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_uop(self, uop: UOp) -> UOp:
        if uop.id in self.uops:
            raise ValueError(f"duplicate uop id: %{uop.id}")
        self.uops[uop.id] = uop
        return uop

    def add_edge(self, src: int, dst: int, kind: EdgeKind, **attrs: Any) -> Edge:
        if src not in self.uops:
            raise KeyError(f"missing source uop %{src}")
        if dst not in self.uops:
            raise KeyError(f"missing destination uop %{dst}")
        edge = Edge(src=src, dst=dst, kind=kind, attrs=dict(attrs))
        self.edges.append(edge)
        return edge

    def successors(self) -> dict[int, list[int]]:
        succ: dict[int, list[int]] = defaultdict(list)
        for edge in self.edges:
            succ[edge.src].append(edge.dst)
        return succ

    def predecessors(self) -> dict[int, list[int]]:
        pred: dict[int, list[int]] = defaultdict(list)
        for edge in self.edges:
            pred[edge.dst].append(edge.src)
        return pred

    def topological_order(self) -> list[int]:
        indegree = {uid: 0 for uid in self.uops}
        succ = self.successors()
        for edge in self.edges:
            indegree[edge.dst] += 1

        ready = deque(sorted(uid for uid, degree in indegree.items() if degree == 0))
        order: list[int] = []
        while ready:
            uid = ready.popleft()
            order.append(uid)
            for dst in succ.get(uid, ()):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)

        if len(order) != len(self.uops):
            raise ValueError("EDMGraph contains a cycle or dangling dependency")
        return order

    def critical_path(self) -> tuple[list[int], float]:
        """Compatibility wrapper.  Prefer ir.analysis.analyze_critical_path."""

        from .analysis import analyze_critical_path

        result = analyze_critical_path(self)
        return list(result.path), result.duration_us

    def phase_duration_us(self) -> float:
        from .analysis import phase_duration_us

        return phase_duration_us(self)

    def total_movement_us(self) -> float:
        return self.total_data_movement_us()

    def total_data_movement_us(self) -> float:
        from .analysis import total_data_movement_us

        return total_data_movement_us(self)

    def exposed_movement_us(self) -> float:
        return self.exposed_data_movement_us()

    def exposed_data_movement_us(self) -> float:
        from .analysis import exposed_data_movement_us

        return exposed_data_movement_us(self)

    def ecr(self) -> float:
        from .analysis import analyze_exposure

        return analyze_exposure(self).ecr

    def edmr(self) -> float:
        from .analysis import analyze_exposure

        return analyze_exposure(self).edmr

    def total_comm_us(self) -> float:
        from .analysis import total_comm_us

        return total_comm_us(self)

    def exposed_comm_us(self) -> float:
        from .analysis import exposed_comm_us

        return exposed_comm_us(self)

    def dump_uops(self) -> str:
        return "\n".join(self.uops[uid].short() for uid in sorted(self.uops))

    def dump_critical_path(self) -> str:
        path, duration = self.critical_path()
        rendered = " -> ".join(f"%{uid}:{self.uops[uid].op}" for uid in path)
        return f"{rendered}\ncritical_path_us={duration:.3f}"
