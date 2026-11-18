from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .ops import PHASE_BEGIN, PHASE_END, is_comm, is_data_movement


@dataclass(frozen=True, slots=True)
class CriticalPathResult:
    path: tuple[int, ...]
    duration_us: float


@dataclass(frozen=True, slots=True)
class ExposureResult:
    phase_time_us: float
    critical_path_us: float
    graph_coverage: float
    total_comm_us: float
    path_comm_us: float
    exposed_comm_us: float
    total_data_movement_us: float
    path_data_movement_us: float
    exposed_data_movement_us: float
    ecr: float
    edmr: float
    critical_path: tuple[int, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OverlapGainResult:
    before_us: float
    after_us: float
    gain_us: float
    gain_ratio: float
    label: str


def analyze_critical_path(
    graph,
    *,
    duration_overrides: Mapping[int, float] | None = None,
) -> CriticalPathResult:
    """Return a deterministic longest dependency path.

    ``duration_overrides`` is the primitive used by counterfactual analysis:
    callers can set a category of uops to zero without mutating the observed
    trace. Ties are broken by end timestamp and id for reproducible dumps, but
    exposure metrics never depend on the selected tie.
    """

    order = graph.topological_order()
    pred = graph.predecessors()
    overrides = duration_overrides or {}

    best: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    for uid in order:
        duration = float(overrides.get(uid, graph.uops[uid].dur_us))
        if duration < 0:
            raise ValueError(f"negative overridden duration for uop %{uid}: {duration}")

        predecessors = pred.get(uid, ())
        if not predecessors:
            best[uid] = duration
            parent[uid] = None
            continue

        prev = max(
            predecessors,
            key=lambda candidate: (
                best[candidate],
                graph.uops[candidate].end_us,
                candidate,
            ),
        )
        best[uid] = best[prev] + duration
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
    """Return observed phase duration, preferring explicit phase boundaries."""

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


def path_comm_us(graph, critical_path: tuple[int, ...] | None = None) -> float:
    path = set(critical_path if critical_path is not None else analyze_critical_path(graph).path)
    return sum(graph.uops[uid].dur_us for uid in path if is_comm(graph.uops[uid].op))


def total_data_movement_us(graph) -> float:
    return sum(uop.dur_us for uop in graph.uops.values() if is_data_movement(uop.op))


def path_data_movement_us(graph, critical_path: tuple[int, ...] | None = None) -> float:
    path = set(critical_path if critical_path is not None else analyze_critical_path(graph).path)
    return sum(graph.uops[uid].dur_us for uid in path if is_data_movement(graph.uops[uid].op))


def counterfactual_exposure_us(graph, predicate: Callable[[str], bool]) -> float:
    """Return the critical-path reduction after zeroing a category of uops.

    This group counterfactual is robust to tied critical paths. If an equally
    long path remains after removing the selected category, the category has
    zero exposed benefit even if it appeared on one diagnostic path.
    """

    baseline = analyze_critical_path(graph).duration_us
    overrides = {
        uid: 0.0
        for uid, uop in graph.uops.items()
        if predicate(uop.op)
    }
    if not overrides:
        return 0.0
    ablated = analyze_critical_path(graph, duration_overrides=overrides).duration_us
    return max(0.0, baseline - ablated)


def exposed_comm_us(graph) -> float:
    return counterfactual_exposure_us(graph, is_comm)


def exposed_data_movement_us(graph) -> float:
    return counterfactual_exposure_us(graph, is_data_movement)


def analyze_exposure(graph) -> ExposureResult:
    critical = analyze_critical_path(graph)
    phase = phase_duration_us(graph)
    selected_comm = path_comm_us(graph, critical.path)
    selected_movement = path_data_movement_us(graph, critical.path)
    exposed_comm = exposed_comm_us(graph)
    exposed_movement = exposed_data_movement_us(graph)
    coverage = 0.0 if phase == 0 else critical.duration_us / phase

    warnings: list[str] = []
    if phase <= 0 and graph.uops:
        warnings.append("non-positive phase duration")
    if phase > 0 and critical.duration_us > phase + 1e-6:
        warnings.append("critical path exceeds observed phase duration")
    if phase > 0 and critical.duration_us + 1e-6 < phase:
        warnings.append(f"unexplained phase time: {phase - critical.duration_us:.3f}us")
    for edge in graph.edges:
        src = graph.uops[edge.src]
        dst = graph.uops[edge.dst]
        if src.end_us > dst.start_us + 1e-6:
            warnings.append(
                f"temporal overlap on dependency %{edge.src}->%{edge.dst} ({edge.kind})"
            )

    return ExposureResult(
        phase_time_us=phase,
        critical_path_us=critical.duration_us,
        graph_coverage=coverage,
        total_comm_us=total_comm_us(graph),
        path_comm_us=selected_comm,
        exposed_comm_us=exposed_comm,
        total_data_movement_us=total_data_movement_us(graph),
        path_data_movement_us=selected_movement,
        exposed_data_movement_us=exposed_movement,
        ecr=0.0 if phase == 0 else exposed_comm / phase,
        edmr=0.0 if phase == 0 else exposed_movement / phase,
        critical_path=critical.path,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def analyze_overlap(before, after, *, min_gain_ratio: float = 0.0) -> OverlapGainResult:
    before_phase = phase_duration_us(before)
    after_phase = phase_duration_us(after)
    gain = before_phase - after_phase
    return OverlapGainResult(
        before_us=before_phase,
        after_us=after_phase,
        gain_us=gain,
        gain_ratio=0.0 if before_phase == 0 else gain / before_phase,
        label=classify_gain(gain, before_phase, min_gain_ratio=min_gain_ratio),
    )


def classify_gain(gain_us: float, baseline_us: float, *, min_gain_ratio: float = 0.0) -> str:
    if baseline_us <= 0:
        return "unknown"
    threshold = baseline_us * min_gain_ratio
    if gain_us > threshold:
        return "helpful"
    if gain_us < -threshold:
        return "harmful"
    return "neutral"


__all__ = [
    "CriticalPathResult",
    "ExposureResult",
    "OverlapGainResult",
    "analyze_critical_path",
    "analyze_exposure",
    "analyze_overlap",
    "classify_gain",
    "counterfactual_exposure_us",
    "exposed_comm_us",
    "exposed_data_movement_us",
    "path_comm_us",
    "path_data_movement_us",
    "phase_duration_us",
    "total_comm_us",
    "total_data_movement_us",
]
