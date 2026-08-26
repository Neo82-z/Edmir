from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    DataflowModule,
    GraphBuilder,
    Node,
    PassManager,
    TensorType,
    build_a2a_gemm_module,
    lower_fused_a2a_gemm,
    match_a2a_gemm,
    verify_dataflow,
)


class DataflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = A2AGemmProblem(world_size=4, rows_per_peer=32, k=128, n=256)

    def test_frontend_builds_typed_a2a_gemm_graph(self) -> None:
        module = build_a2a_gemm_module(self.problem)
        self.assertEqual([node.op for node in module.nodes], ["all_to_all", "gemm"])
        self.assertTrue(verify_dataflow(module).valid)
        self.assertEqual(match_a2a_gemm(module).problem, self.problem)

    def test_dataflow_json_round_trip(self) -> None:
        module = build_a2a_gemm_module(self.problem)
        self.assertEqual(DataflowModule.from_dict(module.to_dict()), module)

    def test_pattern_has_explicit_fused_program_lowering(self) -> None:
        match = match_a2a_gemm(build_a2a_gemm_module(self.problem))
        program = lower_fused_a2a_gemm(match, chunk_rows=16)
        self.assertEqual(program.problem, self.problem)
        self.assertEqual(program.strategy, "fused")
        self.assertTrue(all(op.kind == "fused_transfer_gemm" for op in program.ops))

    def test_pass_manager_canonicalizes_backend_relevant_attributes(self) -> None:
        module = build_a2a_gemm_module(self.problem)
        normalized, trace = PassManager().run(module)
        self.assertEqual([item.name for item in trace], ["canonicalize-dataflow", "verify-dataflow"])
        self.assertEqual(normalized.nodes[0].attrs["peer_order"], "cyclic")
        self.assertEqual(normalized.nodes[1].attrs["epilogue"], "identity")

    def test_verifier_rejects_use_before_definition(self) -> None:
        module = build_a2a_gemm_module(self.problem)
        broken = replace(module, nodes=tuple(reversed(module.nodes)))
        report = verify_dataflow(broken)
        self.assertFalse(report.valid)
        self.assertTrue(any("before its definition" in error for error in report.errors))

    def test_pattern_lowering_rejects_extra_operation(self) -> None:
        module = build_a2a_gemm_module(self.problem)
        extra = Node("extra", "gemm", ("exchanged", "weight"), ("output",), {})
        with self.assertRaisesRegex(ValueError, "invalid dataflow|exactly"):
            match_a2a_gemm(replace(module, nodes=module.nodes + (extra,)))

    def test_frontend_rejects_shape_mismatch(self) -> None:
        builder = GraphBuilder("broken", world_size=4)
        source = builder.input("x", TensorType((64, 128)))
        with self.assertRaisesRegex(ValueError, "rows_per_peer"):
            builder.all_to_all(source, rows_per_peer=32)


if __name__ == "__main__":
    unittest.main()
