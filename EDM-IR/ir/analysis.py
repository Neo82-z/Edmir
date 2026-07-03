from __future__ import annotations

from dataclasses import dataclass

from .ops import PHASE_BEGIN, PHASE_END, is_comm, is_data_movement


@dataclass(frozen=True, slots=True)
class CriticalPathResult:
    path: tuple[int, ...]
    duration_us: float


@dataclass(frozen=True, slots=True)
class ExposureResult:
    phase_time_us: float
    critical_path_us: float
    total_comm_us: float
    exposed_comm_us: float
    total_data_movement_us: float
    exposed_data_movement_us: float
    ecr: float
    edmr: float
    critical_path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OverlapGainResult:
    before_us: float
    after_us: float
    gain_us: float
    gain_ratio: float
    label: str


def analyze_critical_path(graph) -> CriticalPathResult:
    """Return the longest dependency path in a trace-derived EDM graph."""

    order = graph.topological_order()
    pred = graph.predecessors()

    best: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    for uid in order:
        uop = graph.uops[uid]
        if not pred.get(uid):
            best[uid] = uop.dur_us
            parent[uid] = None
            continue

        prev = max(pred[uid], key=lambda p: best[p])
        best[uid] = best[prev] + uop.dur_us
        parent[uid] = prev

    if not best:
        return CriticalPathResult(path=(), duration_us=0.0)

    end = max(best, key=lambda uid: (best[uid], graph.uops[uid].end_us, uid))
    path: list[int] = []
    cur: int | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return CriticalPathResult(path=tuple(path), duration_us=best[end])


def phase_duration_us(graph) -> float:
    """Return phase duration, preferring explicit phase boundaries.

    Imported profiler traces often contain unrelated runtime noise.  Phase
    markers are therefore first-class boundaries; min/max timestamps are only
    a fallback for synthetic or partially imported graphs.
    """

    if not graph.uops:
        return 0.0

    begins = [uop.start_us for uop in graph.uops.values() if uop.op == PHASE_BEGIN]
    ends = [uop.end_us for uop in graph.uops.values() if uop.op == PHASE_END]
    if begins and ends:
        return max(ends) - min(begins)

    start = min(uop.start_us for uop in graph.uops.values())
    end = max(uop.end_us for uop in graph.uops.values())
    return end - start


def total_comm_us(graph) -> float:
    return sum(uop.dur_us for uop in graph.uops.values() if is_comm(uop.op))


def exposed_comm_us(graph, critical_path: tuple[int, ...] | None = None) -> float:
    path = set(critical_path if critical_path is not None else analyze_critical_path(graph).path)
    return sum(uop.dur_us for uop in graph.uops.values() if uop.id in path and is_comm(uop.op))


def total_data_movement_us(graph) -> float:
    return sum(uop.dur_us for uop in graph.uops.values() if is_data_movement(uop.op))


def exposed_data_movement_us(graph, critical_path: tuple[int, ...] | None = None) -> float:
    path = set(critical_path if critical_path is not None else analyze_critical_path(graph).path)
    return sum(uop.dur_us for uop in graph.uops.values() if uop.id in path and is_data_movement(uop.op))


def analyze_exposure(graph) -> ExposureResult:
    critical = analyze_critical_path(graph)
    phase = phase_duration_us(graph)
    exposed_comm = exposed_comm_us(graph, critical.path)
    exposed_movement = exposed_data_movement_us(graph, critical.path)

    return ExposureResult(
        phase_time_us=phase,
        critical_path_us=critical.duration_us,
        total_comm_us=total_comm_us(graph),
        exposed_comm_us=exposed_comm,
        total_data_movement_us=total_data_movement_us(graph),
        exposed_data_movement_us=exposed_movement,
        ecr=0.0 if phase == 0 else exposed_comm / phase,
        edmr=0.0 if phase == 0 else exposed_movement / phase,
        critical_path=critical.path,
    )


def analyze_overlap(before, after) -> OverlapGainResult:
    before_phase = phase_duration_us(before)
    after_phase = phase_duration_us(after)
    gain = before_phase - after_phase
    return OverlapGainResult(
        before_us=before_phase,
        after_us=after_phase,
        gain_us=gain,
        gain_ratio=0.0 if before_phase == 0 else gain / before_phase,
        label=classify_gain(gain, before_phase),
    )


def classify_gain(gain_us: float, baseline_us: float) -> str:
    if baseline_us <= 0:
        return "unknown"
    ratio = gain_us / baseline_us
    if ratio > 0.05:
        return "material"
    if ratio > 0.0:
        return "small"
    if ratio == 0.0:
        return "none"
    return "regression"


__all__ = [
    "CriticalPathResult",
    "ExposureResult",
    "OverlapGainResult",
    "analyze_critical_path",
    "analyze_exposure",
    "analyze_overlap",
    "classify_gain",
    "exposed_comm_us",
    "exposed_data_movement_us",
    "phase_duration_us",
    "total_comm_us",
    "total_data_movement_us",
]
