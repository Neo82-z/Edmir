from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .graph import EDMGraph, Edge
from .uop import DataRef, MovementDesc, TensorDesc, UOp


TRACE_SCHEMA = "edmir.trace.v1"


def graph_to_dict(graph: EDMGraph) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "uops": [asdict(graph.uops[uid]) for uid in sorted(graph.uops)],
        "edges": [asdict(edge) for edge in graph.edges],
    }


def graph_from_dict(value: dict[str, Any]) -> EDMGraph:
    if value.get("schema") not in (None, TRACE_SCHEMA):
        raise ValueError(f"unsupported trace schema: {value.get('schema')}")
    graph = EDMGraph()
    for raw in value.get("uops", ()):
        payload = dict(raw)
        if payload.get("tensor") is not None:
            tensor = dict(payload["tensor"])
            if tensor.get("shape") is not None:
                tensor["shape"] = tuple(tensor["shape"])
            payload["tensor"] = TensorDesc(**tensor)
        if payload.get("movement") is not None:
            movement = dict(payload["movement"])
            for key in ("group_ranks", "src_ranks", "dst_ranks"):
                movement[key] = tuple(movement.get(key, ()))
            payload["movement"] = MovementDesc(**movement)
        payload["reads"] = tuple(DataRef(**item) for item in payload.get("reads", ()))
        payload["writes"] = tuple(DataRef(**item) for item in payload.get("writes", ()))
        payload["raw_events"] = tuple(payload.get("raw_events", ()))
        graph.add_uop(UOp(**payload))
    for raw in value.get("edges", ()):
        graph.edges.append(Edge(src=int(raw["src"]), dst=int(raw["dst"]), kind=raw["kind"], attrs=dict(raw.get("attrs", {}))))
    graph.topological_order()
    return graph


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["TRACE_SCHEMA", "graph_from_dict", "graph_to_dict", "read_json", "write_json"]
