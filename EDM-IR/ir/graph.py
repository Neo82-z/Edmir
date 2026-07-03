from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .ops import EdgeKind, check_edge_kind, is_data_movement
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
        """Return ids on the longest dependency path and its duration."""
        order = self.topological_order()
        pred = self.predecessors()

        best: dict[int, float] = {}
        parent: dict[int, int | None] = {}
        for uid in order:
            uop = self.uops[uid]
            if not pred.get(uid):
                best[uid] = uop.dur_us
                parent[uid] = None
                continue

            prev = max(pred[uid], key=lambda p: best[p])
            best[uid] = best[prev] + uop.dur_us
            parent[uid] = prev

        if not best:
            return [], 0.0

        end = max(best, key=lambda uid: (best[uid], self.uops[uid].end_us, uid))
        path: list[int] = []
        cur: int | None = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path, best[end]

    def phase_duration_us(self) -> float:
        if not self.uops:
            return 0.0
        start = min(uop.start_us for uop in self.uops.values())
        end = max(uop.end_us for uop in self.uops.values())
        return end - start

    def total_movement_us(self) -> float:
        return sum(uop.dur_us for uop in self.uops.values() if is_data_movement(uop.op))
    
    def total_data_movement_us(graph)  -> float:
        return sum(u.dur_us for u in graph.uops.values() if is_data_movement(u.op))

    def exposed_movement_us(self) -> float:
        path, _ = self.critical_path()
        return sum(self.uops[uid].dur_us for uid in path if is_data_movement(self.uops[uid].op))
    
    def exposed_data_movement_us(graph) -> float:
        cp = set(critical_path(graph).path)
        return sum(u.dur_us for u in graph.uops.values() if is_data_movement(u.op) and u.id in cp)

    def ecr(self) -> float:
        phase = self.phase_duration_us()
        return 0.0 if phase == 0 else self.exposed_movement_us() / phase

    def total_comm_us(graph) -> float:
        return sum(u.dur_us for u in graph.uops.values() if is_comm(u.op))

    def exposed_comm_us(graph) -> float:
        cp = set(critical_path(graph).path)
        return sum(u.dur_us for u in graph.uops.values() if is_comm(u.op) and u.id in cp)

    def dump_uops(self) -> str:
        return "\n".join(self.uops[uid].short() for uid in sorted(self.uops))

    def dump_critical_path(self) -> str:
        path, duration = self.critical_path()
        rendered = " -> ".join(f"%{uid}:{self.uops[uid].op}" for uid in path)
        return f"{rendered}\ncritical_path_us={duration:.3f}"
    

