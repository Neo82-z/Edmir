from __future__ import annotations

from dataclasses import dataclass

from .dataflow import DataflowModule, Node
from .model import A2AGemmProblem
from .passes import verify_dataflow


@dataclass(frozen=True, slots=True)
class A2AGemmMatch:
    problem: A2AGemmProblem
    all_to_all: Node
    gemm: Node
    activation: str
    weight: str
    output: str


def match_a2a_gemm(module: DataflowModule) -> A2AGemmMatch:
    """Match the single high-level pattern supported by compiler v0.1."""

    verify_dataflow(module).require_valid()
    if len(module.nodes) != 2:
        raise ValueError("v0.1 lowering requires exactly all_to_all followed by GEMM")
    all_to_all, gemm = module.nodes
    if all_to_all.op != "all_to_all" or gemm.op != "gemm":
        raise ValueError("v0.1 lowering requires all_to_all followed by GEMM")
    if gemm.inputs[0] != all_to_all.outputs[0]:
        raise ValueError("GEMM lhs must consume the all_to_all result")
    if all_to_all.inputs[0] not in module.inputs or gemm.inputs[1] not in module.inputs:
        raise ValueError("activation and weight must be frontend inputs")
    if module.outputs != gemm.outputs:
        raise ValueError("GEMM result must be the only module output")

    values = module.value_map()
    activation = values[all_to_all.inputs[0]]
    weight = values[gemm.inputs[1]]
    if weight.type.distribution != "replicated":
        raise ValueError("GEMM weight must be replicated across ranks in v0.1")
    rows_per_peer = int(all_to_all.attrs["rows_per_peer"])
    _local_rows, k = activation.type.shape
    weight_k, n = weight.type.shape
    if k != weight_k:
        raise ValueError("lowering observed inconsistent GEMM reduction dimensions")
    problem = A2AGemmProblem(
        name=module.name,
        world_size=module.world_size,
        rows_per_peer=rows_per_peer,
        k=k,
        n=n,
        dtype=activation.type.dtype,
        layout=activation.type.layout,
    )
    problem.validate()
    return A2AGemmMatch(
        problem=problem,
        all_to_all=all_to_all,
        gemm=gemm,
        activation=activation.id,
        weight=weight.id,
        output=gemm.outputs[0],
    )


__all__ = ["A2AGemmMatch", "match_a2a_gemm"]
