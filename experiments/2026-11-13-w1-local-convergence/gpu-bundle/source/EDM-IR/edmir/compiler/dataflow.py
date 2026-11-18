from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .model import DTYPE_BYTES, _check_schema


SCHEMA_DATAFLOW = "edmir.dataflow.v1"
DATAFLOW_OPS = frozenset({"all_to_all", "gemm"})
DISTRIBUTIONS = frozenset({"uniform_shard", "replicated"})


@dataclass(frozen=True, slots=True)
class TensorType:
    """Per-rank logical tensor type used by the high-level dataflow IR."""

    shape: tuple[int, ...]
    dtype: str = "float16"
    layout: str = "row_major"
    distribution: str = "uniform_shard"

    def validate(self) -> None:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported tensor dtype {self.dtype!r}")
        if self.layout != "row_major":
            raise ValueError("v0.1 supports only row_major tensors")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"unsupported distribution {self.distribution!r}")

    @property
    def size_bytes(self) -> int:
        size = DTYPE_BYTES[self.dtype]
        for dimension in self.shape:
            size *= dimension
        return size

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TensorType":
        tensor_type = cls(
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value.get("dtype", "float16")),
            layout=str(value.get("layout", "row_major")),
            distribution=str(value.get("distribution", "uniform_shard")),
        )
        tensor_type.validate()
        return tensor_type


@dataclass(frozen=True, slots=True)
class Value:
    id: str
    type: TensorType


@dataclass(frozen=True, slots=True)
class Node:
    """One pure dataflow operation. Ordering comes only from SSA values."""

    id: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataflowModule:
    name: str
    world_size: int
    values: tuple[Value, ...]
    nodes: tuple[Node, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def value_map(self) -> dict[str, Value]:
        return {value.id: value for value in self.values}

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def producer_map(self) -> dict[str, Node]:
        return {output: node for node in self.nodes for output in node.outputs}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_DATAFLOW,
            "name": self.name,
            "world_size": self.world_size,
            "values": [
                {"id": value.id, "type": value.type.to_dict()} for value in self.values
            ],
            "nodes": [
                {
                    "id": node.id,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "attrs": node.attrs,
                }
                for node in self.nodes
            ],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataflowModule":
        _check_schema(value, SCHEMA_DATAFLOW)
        return cls(
            name=str(value["name"]),
            world_size=int(value["world_size"]),
            values=tuple(
                Value(id=str(item["id"]), type=TensorType.from_dict(item["type"]))
                for item in value.get("values", ())
            ),
            nodes=tuple(
                Node(
                    id=str(item["id"]),
                    op=str(item["op"]),
                    inputs=tuple(str(name) for name in item.get("inputs", ())),
                    outputs=tuple(str(name) for name in item.get("outputs", ())),
                    attrs=dict(item.get("attrs", {})),
                )
                for item in value.get("nodes", ())
            ),
            inputs=tuple(str(name) for name in value.get("inputs", ())),
            outputs=tuple(str(name) for name in value.get("outputs", ())),
            metadata=dict(value.get("metadata", {})),
        )


__all__ = [
    "DATAFLOW_OPS",
    "DISTRIBUTIONS",
    "DataflowModule",
    "Node",
    "SCHEMA_DATAFLOW",
    "TensorType",
    "Value",
]
