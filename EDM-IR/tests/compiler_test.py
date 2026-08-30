from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    CalibrationProfile,
    CommSample,
    CompilerOptions,
    ExecutionOverhead,
    EXPLICIT_EPOCH,
    Program,
    PairwiseCommCurve,
    TopologySnapshot,
    build_fused_program,
    build_pipelined_program,
    build_sequential_program,
    compile_a2a_gemm,
    derive_dependencies,
    simulate_program,
    verify_program,
    validate_remote_publication,
)


def full_topology(world_size: int = 4) -> TopologySnapshot:
    return TopologySnapshot(
        world_size=world_size,
        devices=tuple("NVIDIA A800" for _ in range(world_size)),
        p2p=tuple(tuple(True for _ in range(world_size)) for _ in range(world_size)),
    )


def synthetic_profile() -> CalibrationProfile:
    return CalibrationProfile(
        comm_latency_us=1.0,
        comm_bandwidth_gbps=50.0,
        gemm_launch_us=0.5,
        gemm_tflops=100.0,
        sync_us=0.2,
    )


class CompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = A2AGemmProblem(world_size=4, rows_per_peer=64, k=4096, n=4096)

    def test_both_program_families_are_valid_and_simulatable(self) -> None:
        sequential = build_sequential_program(self.problem)
        pipeline = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2)
        self.assertTrue(verify_program(self.problem, sequential).valid)
        self.assertTrue(verify_program(self.problem, pipeline).valid)
        self.assertGreater(simulate_program(sequential, synthetic_profile()).makespan_us, 0.0)
        self.assertGreater(simulate_program(pipeline, synthetic_profile()).makespan_us, 0.0)

    def test_fused_program_has_direct_dataflow_and_combined_resources(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        report = verify_program(self.problem, program)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(program.strategy, "fused")
        self.assertFalse(any(buffer.role == "staging" for buffer in program.buffers))
        self.assertEqual(len(program.ops), 4 * 4 * 4)
        remote = next(
            op for op in program.ops
            if op.peer_rank is not None and op.peer_rank != op.rank
        )
        self.assertEqual(
            set(remote.resources()),
            {
                f"rank{remote.peer_rank}.comm",
                f"rank{remote.rank}.comm",
                f"rank{remote.rank}.compute",
            },
        )

    def test_remote_inputs_declare_immutable_prelaunch_publication(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        report = validate_remote_publication(program)
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(report.executable)
        self.assertEqual(report.mode, "immutable_external")

    def test_remote_inputs_reject_a_missing_publication_contract(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        broken = replace(
            program,
            metadata={
                key: value
                for key, value in program.metadata.items()
                if key != "remote_input_contract"
            },
        )
        report = verify_program(self.problem, broken)
        self.assertFalse(report.valid)
        self.assertTrue(
            any("versioned remote_input_contract" in error for error in report.errors)
        )

    def test_mutable_remote_inputs_fail_closed_until_events_are_visible(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        mutable = replace(
            program,
            metadata={
                **program.metadata,
                "remote_input_contract": {
                    "schema": "edmir.remote_publication.v1",
                    "mode": EXPLICIT_EPOCH,
                    "visibility": "release_acquire",
                    "reuse": "all_consumers_complete",
                },
            },
        )
        publication = validate_remote_publication(mutable)
        self.assertFalse(publication.valid)
        self.assertFalse(publication.executable)
        self.assertEqual(publication.mode, EXPLICIT_EPOCH)
        self.assertTrue(
            any("not implemented" in error for error in publication.errors)
        )
        self.assertFalse(verify_program(self.problem, mutable).valid)

    def test_fused_program_rejects_fake_unfused_cost_model(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        with self.assertRaisesRegex(ValueError, "measured fused-kernel cost model"):
            simulate_program(program, synthetic_profile())

    def test_fused_verifier_rejects_missing_remote_input(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        op = next(item for item in program.ops if item.peer_rank != item.rank)
        op_index = program.ops.index(op)
        weight = next(
            name for name in op.reads if program.buffer_map()[name].role == "weight"
        )
        broken = replace(
            program,
            ops=program.ops[:op_index] + (replace(op, reads=(weight,)),) + program.ops[op_index + 1 :],
        )
        report = verify_program(self.problem, broken)
        self.assertFalse(report.valid)
        self.assertTrue(any("source buffer" in error for error in report.errors))

    def test_runtime_overhead_uses_exact_plan_feature_counts(self) -> None:
        problem = A2AGemmProblem(world_size=2, rows_per_peer=16, k=64, n=64)
        program = build_pipelined_program(problem, chunk_rows=8, buffer_count=1)
        profile = replace(
            synthetic_profile(),
            execution_overhead=ExecutionOverhead(
                iteration_us=1.0,
                op_dispatch_us=2.0,
                event_record_us=3.0,
                event_wait_us=4.0,
                buffer_reuse_us=5.0,
            ),
        )
        result = simulate_program(program, profile)
        self.assertEqual(result.features.op_dispatch_count, 16)
        self.assertEqual(result.features.event_record_count, 20)
        self.assertEqual(result.features.event_wait_count, 42)
        self.assertEqual(result.features.buffer_reuse_count, 6)
        self.assertEqual(result.overhead_us, 1 + 16 * 2 + 20 * 3 + 42 * 4 + 6 * 5)
        self.assertAlmostEqual(result.makespan_us, result.device_makespan_us + result.overhead_us)

    def test_simulator_uses_rank_pair_communication_curve(self) -> None:
        problem = A2AGemmProblem(world_size=2, rows_per_peer=16, k=64, n=64)
        program = build_pipelined_program(problem, chunk_rows=16, buffer_count=2)
        analytic = synthetic_profile()
        measured = replace(
            analytic,
            pairwise_comm_curves=(
                PairwiseCommCurve(0, 1, (CommSample(1024, 100.0),)),
                PairwiseCommCurve(1, 0, (CommSample(1024, 100.0),)),
            ),
        )
        self.assertGreater(
            simulate_program(program, measured).makespan_us,
            simulate_program(program, analytic).makespan_us,
        )

    def test_program_serializes_ops_and_buffers_but_not_edges(self) -> None:
        value = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2).to_dict()
        self.assertEqual(value["schema"], "edmir.program.v1")
        self.assertIn("ops", value)
        self.assertIn("buffers", value)
        self.assertNotIn("edges", value)

    def test_data_readiness_and_buffer_reuse_are_derived(self) -> None:
        program = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=1)
        dependencies = derive_dependencies(program).dependencies
        keys = {(dep.src, dep.dst, dep.kind, dep.buffer) for dep in dependencies}
        self.assertIn(
            (
                "transfer.s0.d0.c0",
                "gemm.s0.d0.c0",
                "read_after_write",
                "stage.r0.slot0",
            ),
            keys,
        )
        self.assertIn(
            (
                "gemm.s0.d0.c0",
                "transfer.s0.d0.c1",
                "buffer_reuse",
                "stage.r0.slot0",
            ),
            keys,
        )

    def test_verifier_rejects_read_before_transfer(self) -> None:
        program = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2)
        gemm = next(op for op in program.ops if op.kind == "gemm")
        broken = replace(program, ops=(gemm,) + tuple(op for op in program.ops if op != gemm))
        report = verify_program(self.problem, broken)
        self.assertFalse(report.valid)
        self.assertTrue(any("uninitialized read" in error for error in report.errors))

    def test_verifier_rejects_missing_transfer(self) -> None:
        program = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2)
        first_transfer = next(op for op in program.ops if op.kind == "transfer")
        broken = replace(program, ops=tuple(op for op in program.ops if op != first_transfer))
        report = verify_program(self.problem, broken)
        self.assertFalse(report.valid)
        self.assertTrue(any("coverage" in error for error in report.errors))

    def test_verifier_rejects_unknown_op_kind(self) -> None:
        program = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2)
        first = replace(program.ops[0], kind="mystery")
        report = verify_program(self.problem, replace(program, ops=(first,) + program.ops[1:]))
        self.assertFalse(report.valid)
        self.assertTrue(any("unknown kind" in error for error in report.errors))

    def test_verifier_rejects_gemm_without_local_weight(self) -> None:
        program = build_pipelined_program(self.problem, chunk_rows=16, buffer_count=2)
        gemm_index = next(index for index, op in enumerate(program.ops) if op.kind == "gemm")
        gemm = program.ops[gemm_index]
        broken_gemm = replace(gemm, reads=(gemm.reads[0],))
        broken_ops = program.ops[:gemm_index] + (broken_gemm,) + program.ops[gemm_index + 1 :]
        report = verify_program(self.problem, replace(program, ops=broken_ops))
        self.assertFalse(report.valid)
        self.assertTrue(any("weight buffer" in error for error in report.errors))

    def test_verifier_rejects_incomplete_sequential_barrier(self) -> None:
        program = build_sequential_program(self.problem)
        barrier_index = next(index for index, op in enumerate(program.ops) if op.kind == "barrier")
        barrier = program.ops[barrier_index]
        broken_barrier = replace(barrier, reads=barrier.reads[:-1])
        broken_ops = program.ops[:barrier_index] + (broken_barrier,) + program.ops[barrier_index + 1 :]
        report = verify_program(self.problem, replace(program, ops=broken_ops))
        self.assertFalse(report.valid)
        self.assertTrue(any("every incoming tile" in error for error in report.errors))

    def test_compiler_selects_pipeline_when_predicted_faster(self) -> None:
        result = compile_a2a_gemm(
            self.problem,
            synthetic_profile(),
            full_topology(),
            CompilerOptions(chunk_rows=(16, 32), buffer_counts=(2,)),
        )
        self.assertEqual(result.chosen.program.strategy, "pipelined")

    def test_default_search_covers_divisor_chunks_and_buffer_depths(self) -> None:
        problem = replace(self.problem, rows_per_peer=96)
        result = compile_a2a_gemm(problem, synthetic_profile(), full_topology())
        pipeline = {
            (candidate.program.chunk_rows, candidate.program.buffer_count)
            for candidate in result.candidates
            if candidate.program.strategy == "pipelined"
        }
        self.assertEqual(
            pipeline,
            {
                (chunk_rows, buffer_count)
                for chunk_rows in (16, 32, 48, 96)
                for buffer_count in (1, 2, 4)
            },
        )

    def test_profile_selection_margin_controls_default_fallback(self) -> None:
        profile = replace(synthetic_profile(), selection_min_gain_ratio=1.0)
        fallback = compile_a2a_gemm(
            self.problem,
            profile,
            full_topology(),
            CompilerOptions(chunk_rows=(64,), buffer_counts=(4,)),
        )
        override = compile_a2a_gemm(
            self.problem,
            profile,
            full_topology(),
            CompilerOptions(chunk_rows=(64,), buffer_counts=(4,), min_gain_ratio=0.0),
        )
        self.assertEqual(fallback.chosen.program.strategy, "sequential")
        self.assertEqual(override.chosen.program.strategy, "pipelined")

    def test_equal_prediction_prefers_fewer_ops_then_more_buffers(self) -> None:
        result = compile_a2a_gemm(
            self.problem,
            synthetic_profile(),
            full_topology(),
            CompilerOptions(chunk_rows=(64,), buffer_counts=(2, 4)),
        )
        pipeline = [
            candidate for candidate in result.candidates
            if candidate.program.strategy == "pipelined"
        ]
        self.assertEqual(pipeline[0].makespan_us, pipeline[1].makespan_us)
        self.assertEqual(result.chosen.program.strategy, "pipelined")
        self.assertEqual(result.chosen.program.buffer_count, 4)
        self.assertIsNone(result.fallback_reason)

    def test_compiler_falls_back_without_full_p2p(self) -> None:
        p2p = tuple(
            tuple(src == dst for dst in range(self.problem.world_size))
            for src in range(self.problem.world_size)
        )
        topology = TopologySnapshot(
            world_size=4,
            devices=tuple("NVIDIA A800" for _ in range(4)),
            p2p=p2p,
        )
        result = compile_a2a_gemm(self.problem, synthetic_profile(), topology)
        self.assertEqual(result.chosen.program.strategy, "sequential")
        self.assertIn("peer-to-peer", result.fallback_reason or "")

    def test_program_rejects_unknown_schema(self) -> None:
        value = build_sequential_program(self.problem).to_dict()
        value["schema"] = "edmir.program.v999"
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            Program.from_dict(value)

    def test_legacy_plan_schema_rebuilds_program(self) -> None:
        legacy = {
            "schema": "edmir.plan.v1",
            "problem": self.problem.to_dict(),
            "strategy": "pipelined",
            "chunk_rows": 16,
            "buffer_count": 2,
            "tasks": [],
            "edges": [],
            "metadata": {"old": True},
        }
        program = Program.from_dict(legacy)
        self.assertEqual(program.strategy, "pipelined")
        self.assertEqual(program.metadata["loaded_from"], "edmir.plan.v1")
        self.assertTrue(verify_program(self.problem, program).valid)


if __name__ == "__main__":
    unittest.main()
