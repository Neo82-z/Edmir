from __future__ import annotations

from collections.abc import Iterable

from .graph import EDMGraph
from .ops import is_data_movement
from .uop import DataRef, UOp


def format_uop(uop: UOp, *, highlight: bool = False) -> str:
    mark = "*" if highlight else " "
    tags = _format_tags(uop)
    return (
        f"{mark}%{uop.id:03d} {uop.op:<20} {uop.name:<24} "
        f"{uop.start_us:9.3f}->{uop.end_us:9.3f} "
        f"dur={uop.dur_us:9.3f}us{tags}"
    )


def format_uops(graph: EDMGraph, *, highlight: Iterable[int] = ()) -> str:
    highlighted = set(highlight)
    return "\n".join(
        format_uop(graph.uops[uid], highlight=uid in highlighted)
        for uid in sorted(graph.uops)
    )


def format_critical_path(graph: EDMGraph) -> str:
    path, duration_us = graph.critical_path()
    if not path:
        return "critical_path=<empty>\ncritical_path_us=0.000"
    rendered = " -> ".join(f"%{uid}:{graph.uops[uid].op}" for uid in path)
    return f"critical_path={rendered}\ncritical_path_us={duration_us:.3f}"


def format_summary(graph: EDMGraph) -> str:
    path, critical_path_us = graph.critical_path()
    exposed_ids = [uid for uid in path if is_data_movement(graph.uops[uid].op)]
    lines = [
        f"phase_us={graph.phase_duration_us():.3f}",
        f"critical_path_us={critical_path_us:.3f}",
        f"total_movement_us={graph.total_movement_us():.3f}",
        f"exposed_movement_us={graph.exposed_movement_us():.3f}",
        f"ecr={graph.ecr():.3f}",
    ]
    if exposed_ids:
        lines.append("exposed_movement=" + ", ".join(f"%{uid}:{graph.uops[uid].op}" for uid in exposed_ids))
    else:
        lines.append("exposed_movement=<none>")
    return "\n".join(lines)


def _format_tags(uop: UOp) -> str:
    tags: list[str] = []
    for key in ("phase", "stream", "device"):
        value = uop.attr(key)
        if value is not None:
            tags.append(f"{key}={value}")

    rank = uop.attr("global_rank")
    if rank is not None:
        tags.append(f"rank={rank}")

    movement = uop.movement
    if movement is not None:
        if movement.size_bytes is not None:
            tags.append(f"bytes={movement.size_bytes}")
        if movement.group is not None:
            tags.append(f"group={movement.group}")

    if uop.reads:
        tags.append("reads=" + _format_data_refs(uop.reads))
    if uop.writes:
        tags.append("writes=" + _format_data_refs(uop.writes))

    return "" if not tags else "  " + " ".join(tags)


def _format_data_refs(refs: tuple[DataRef, ...]) -> str:
    return ",".join(f"{ref.kind}:{ref.data_id}" for ref in refs)


__all__ = [
    "format_critical_path",
    "format_summary",
    "format_uop",
    "format_uops",
]
