from __future__ import annotations

from .dataflow import DataflowModule, Node, TensorType, Value
from .model import A2AGemmProblem


class GraphBuilder:
    """Tiny frontend builder that emits typed SSA-like Dataflow IR."""

    def __init__(self, name: str, *, world_size: int) -> None:
        if world_size < 2:
            raise ValueError("world_size must be >= 2")
        self.name = name
        self.world_size = world_size
        self._values: dict[str, Value] = {}
        self._nodes: list[Node] = []
        self._inputs: list[str] = []
        self._outputs: list[str] = []

    def _define_value(self, name: str, tensor_type: TensorType) -> Value:
        if name in self._values:
            raise ValueError(f"duplicate frontend value {name!r}")
        tensor_type.validate()
        value = Value(name, tensor_type)
        self._values[name] = value
        return value

    def input(self, name: str, tensor_type: TensorType) -> Value:
        value = self._define_value(name, tensor_type)
        self._inputs.append(value.id)
        return value

    def all_to_all(self, source: Value, *, rows_per_peer: int, name: str = "a2a") -> Value:
        if source.id not in self._values:
            raise ValueError("all_to_all source does not belong to this builder")
        if rows_per_peer <= 0 or source.type.shape[0] != self.world_size * rows_per_peer:
            raise ValueError("all_to_all rows_per_peer does not match the local row dimension")
        output = self._define_value(
            name,
            TensorType(
                shape=source.type.shape,
                dtype=source.type.dtype,
                layout=source.type.layout,
                distribution="uniform_shard",
            ),
        )
        self._nodes.append(
            Node(
                id=f"{name}.all_to_all",
                op="all_to_all",
                inputs=(source.id,),
                outputs=(output.id,),
                attrs={"rows_per_peer": rows_per_peer},
            )
        )
        return output

    def gemm(self, lhs: Value, rhs: Value, *, name: str = "gemm") -> Value:
        if lhs.id not in self._values or rhs.id not in self._values:
            raise ValueError("GEMM operands must belong to this builder")
        if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
            raise ValueError("GEMM requires rank-2 tensors")
        m, k = lhs.type.shape
        rhs_k, n = rhs.type.shape
        if k != rhs_k or lhs.type.dtype != rhs.type.dtype:
            raise ValueError("GEMM operand shape or dtype mismatch")
        output = self._define_value(
            name,
            TensorType(
                shape=(m, n),
                dtype=lhs.type.dtype,
                layout="row_major",
                distribution=lhs.type.distribution,
            ),
        )
        self._nodes.append(
            Node(
                id=f"{name}.gemm",
                op="gemm",
                inputs=(lhs.id, rhs.id),
                outputs=(output.id,),
                attrs={"transpose_a": False, "transpose_b": False},
            )
        )
        return output

    def output(self, value: Value) -> None:
        if value.id not in self._values:
            raise ValueError("output value does not belong to this builder")
        if value.id not in self._outputs:
            self._outputs.append(value.id)

    def build(self, *, metadata: dict | None = None) -> DataflowModule:
        if not self._outputs:
            raise ValueError("frontend module requires at least one output")
        return DataflowModule(
            name=self.name,
            world_size=self.world_size,
            values=tuple(self._values.values()),
            nodes=tuple(self._nodes),
            inputs=tuple(self._inputs),
            outputs=tuple(self._outputs),
            metadata=dict(metadata or {}),
        )


def build_a2a_gemm_module(problem: A2AGemmProblem) -> DataflowModule:
    problem.validate()
    builder = GraphBuilder(problem.name, world_size=problem.world_size)
    activations = builder.input(
        "activations",
        TensorType(
            shape=(problem.local_rows, problem.k),
            dtype=problem.dtype,
            distribution="uniform_shard",
        ),
    )
    weight = builder.input(
        "weight",
        TensorType(
            shape=(problem.k, problem.n),
            dtype=problem.dtype,
            distribution="replicated",
        ),
    )
    exchanged = builder.all_to_all(
        activations,
        rows_per_peer=problem.rows_per_peer,
        name="exchanged",
    )
    output = builder.gemm(exchanged, weight, name="output")
    builder.output(output)
    return builder.build(metadata={"frontend": "a2a_gemm", "layout": problem.layout})


__all__ = ["GraphBuilder", "build_a2a_gemm_module"]
