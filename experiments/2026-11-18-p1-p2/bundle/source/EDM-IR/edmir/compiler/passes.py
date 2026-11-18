from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .dataflow import DATAFLOW_OPS, DataflowModule, Node


@dataclass(frozen=True, slots=True)
class DataflowVerificationReport:
    valid: bool
    errors: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("invalid dataflow module: " + "; ".join(self.errors))


def verify_dataflow(module: DataflowModule) -> DataflowVerificationReport:
    errors: list[str] = []
    if module.world_size < 2:
        errors.append("dataflow world_size must be >= 2")
    values = module.value_map()
    if len(values) != len(module.values):
        errors.append("duplicate dataflow value id")
    nodes = module.node_map()
    if len(nodes) != len(module.nodes):
        errors.append("duplicate dataflow node id")

    for value in module.values:
        try:
            value.type.validate()
        except ValueError as exc:
            errors.append(f"value {value.id}: {exc}")

    input_ids = set(module.inputs)
    if len(input_ids) != len(module.inputs):
        errors.append("duplicate module input")
    for name in module.inputs:
        if name not in values:
            errors.append(f"unknown module input {name}")

    available = set(input_ids)
    produced: set[str] = set()
    for node in module.nodes:
        if node.op not in DATAFLOW_OPS:
            errors.append(f"node {node.id} has unsupported op {node.op!r}")
        for name in node.inputs:
            if name not in values:
                errors.append(f"node {node.id} reads unknown value {name}")
            elif name not in available:
                errors.append(f"node {node.id} reads {name} before its definition")
        for name in node.outputs:
            if name not in values:
                errors.append(f"node {node.id} writes unknown value {name}")
            if name in input_ids or name in produced:
                errors.append(f"value {name} has multiple definitions")
            produced.add(name)
            available.add(name)

        if node.op == "all_to_all":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                errors.append(f"all_to_all node {node.id} requires one input and one output")
                continue
            source = values.get(node.inputs[0])
            output = values.get(node.outputs[0])
            rows_per_peer = node.attrs.get("rows_per_peer")
            if not isinstance(rows_per_peer, int) or rows_per_peer <= 0:
                errors.append(f"all_to_all node {node.id} has invalid rows_per_peer")
            elif source is not None and (
                len(source.type.shape) != 2
                or source.type.shape[0] != module.world_size * rows_per_peer
            ):
                errors.append(f"all_to_all node {node.id} local rows do not match world size")
            if source is not None and output is not None and source.type != output.type:
                errors.append(f"all_to_all node {node.id} must preserve its tensor type")
            if node.attrs.get("peer_order", "cyclic") != "cyclic":
                errors.append(f"all_to_all node {node.id} uses an unsupported peer order")

        elif node.op == "gemm":
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                errors.append(f"GEMM node {node.id} requires two inputs and one output")
                continue
            lhs = values.get(node.inputs[0])
            rhs = values.get(node.inputs[1])
            output = values.get(node.outputs[0])
            if lhs is None or rhs is None or output is None:
                continue
            if len(lhs.type.shape) != 2 or len(rhs.type.shape) != 2:
                errors.append(f"GEMM node {node.id} requires rank-2 operands")
                continue
            m, k = lhs.type.shape
            rhs_k, n = rhs.type.shape
            expected = (m, n)
            if k != rhs_k or output.type.shape != expected:
                errors.append(f"GEMM node {node.id} has incompatible shapes")
            if not (lhs.type.dtype == rhs.type.dtype == output.type.dtype):
                errors.append(f"GEMM node {node.id} has incompatible dtypes")
            if node.attrs.get("transpose_a", False) or node.attrs.get("transpose_b", False):
                errors.append(f"GEMM node {node.id} transpose is outside v0.1")
            if node.attrs.get("epilogue", "identity") != "identity":
                errors.append(f"GEMM node {node.id} epilogue is outside v0.1")

    for name in module.outputs:
        if name not in values:
            errors.append(f"unknown module output {name}")
        elif name not in available:
            errors.append(f"module output {name} is not defined")
    if not module.outputs:
        errors.append("dataflow module requires at least one output")

    return DataflowVerificationReport(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
    )


@dataclass(frozen=True, slots=True)
class PassTrace:
    name: str
    changed: bool


class DataflowPass(Protocol):
    name: str

    def run(self, module: DataflowModule) -> DataflowModule: ...


class CanonicalizeDataflowPass:
    name = "canonicalize-dataflow"

    def run(self, module: DataflowModule) -> DataflowModule:
        canonical_nodes: list[Node] = []
        for node in module.nodes:
            attrs = dict(node.attrs)
            if node.op == "all_to_all":
                attrs.setdefault("peer_order", "cyclic")
            elif node.op == "gemm":
                attrs.setdefault("transpose_a", False)
                attrs.setdefault("transpose_b", False)
                attrs.setdefault("epilogue", "identity")
            canonical_nodes.append(replace(node, attrs=attrs))
        return replace(module, nodes=tuple(canonical_nodes))


class VerifyDataflowPass:
    name = "verify-dataflow"

    def run(self, module: DataflowModule) -> DataflowModule:
        verify_dataflow(module).require_valid()
        return module


class PassManager:
    def __init__(self, passes: tuple[DataflowPass, ...] | None = None) -> None:
        self.passes = passes or (CanonicalizeDataflowPass(), VerifyDataflowPass())

    def run(self, module: DataflowModule) -> tuple[DataflowModule, tuple[PassTrace, ...]]:
        current = module
        trace: list[PassTrace] = []
        for compiler_pass in self.passes:
            updated = compiler_pass.run(current)
            trace.append(PassTrace(compiler_pass.name, updated != current))
            current = updated
        return current, tuple(trace)


__all__ = [
    "CanonicalizeDataflowPass",
    "DataflowPass",
    "DataflowVerificationReport",
    "PassManager",
    "PassTrace",
    "VerifyDataflowPass",
    "verify_dataflow",
]
