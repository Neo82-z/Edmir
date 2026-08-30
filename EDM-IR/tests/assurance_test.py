from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    AdmissionPolicy,
    EvidenceRecord,
    MeasurementInterval,
    admit_evidence,
    build_fused_program,
    build_pipelined_program,
    build_sequential_program,
    compare_reference_outputs,
    make_reference_fixture,
    reference_gemm,
    validate_program_semantics,
)


class ExecutableSemanticsTest(unittest.TestCase):
    def test_all_program_families_match_the_reference_relation(self) -> None:
        checked = 0
        for world_size in (2, 3):
            for rows_per_peer in (2, 4):
                for k in (2, 3):
                    for n in (2, 3):
                        problem = A2AGemmProblem(
                            world_size=world_size,
                            rows_per_peer=rows_per_peer,
                            k=k,
                            n=n,
                        )
                        programs = [build_sequential_program(problem)]
                        for chunk_rows in range(1, rows_per_peer + 1):
                            if rows_per_peer % chunk_rows:
                                continue
                            programs.extend(
                                (
                                    build_pipelined_program(
                                        problem,
                                        chunk_rows=chunk_rows,
                                        buffer_count=1,
                                    ),
                                    build_pipelined_program(
                                        problem,
                                        chunk_rows=chunk_rows,
                                        buffer_count=2,
                                    ),
                                    build_fused_program(problem, chunk_rows=chunk_rows),
                                )
                            )
                        for index, program in enumerate(programs):
                            fixture = make_reference_fixture(
                                problem,
                                program.chunk_rows,
                                seed=1000 * world_size + 100 * rows_per_peer + 10 * k + n,
                            )
                            report = validate_program_semantics(program, fixture)
                            self.assertTrue(
                                report.valid,
                                (program.strategy, index, report.errors),
                            )
                            self.assertGreater(report.checked_elements, 0)
                            checked += 1
        self.assertEqual(checked, 136)

    def test_split_k_and_grouped_n_preserve_gemm_semantics(self) -> None:
        left = (
            (0.25, -0.5, 1.25, 0.75),
            (-0.125, 0.625, 0.875, -1.5),
        )
        right = (
            (0.5, -1.0, 0.25),
            (0.75, 0.125, -0.5),
            (-0.25, 0.5, 1.5),
            (1.25, -0.75, 0.375),
        )
        expected = reference_gemm(left, right)
        for split_k_slices in (1, 2, 4):
            for tile_n in (1, 2, 3):
                for n_tiles_per_group in (1, 2, 3):
                    observed = reference_gemm(
                        left,
                        right,
                        split_k_slices=split_k_slices,
                        tile_n=tile_n,
                        n_tiles_per_group=n_tiles_per_group,
                    )
                    report = compare_reference_outputs(
                        {"tile": expected},
                        {"tile": observed},
                    )
                    self.assertTrue(report.valid, report.errors)

    def test_invalid_reduction_and_grouping_parameters_are_rejected(self) -> None:
        left = ((1.0, 2.0),)
        right = ((3.0,), (4.0,))
        for split_k_slices in (0, 3):
            with self.assertRaisesRegex(ValueError, "split_k_slices"):
                reference_gemm(left, right, split_k_slices=split_k_slices)
        with self.assertRaisesRegex(ValueError, "n_tiles_per_group"):
            reference_gemm(left, right, n_tiles_per_group=0)

    def test_wrong_remote_mapping_is_rejected(self) -> None:
        problem = A2AGemmProblem(world_size=2, rows_per_peer=2, k=3, n=2)
        program = build_fused_program(problem, chunk_rows=1)
        op_index = next(
            index
            for index, op in enumerate(program.ops)
            if op.peer_rank is not None and op.peer_rank != op.rank
        )
        op = program.ops[op_index]
        broken = replace(
            program,
            ops=program.ops[:op_index]
            + (replace(op, peer_rank=op.rank),)
            + program.ops[op_index + 1 :],
        )
        fixture = make_reference_fixture(problem, 1, seed=17)
        report = validate_program_semantics(broken, fixture)
        self.assertFalse(report.valid)
        self.assertTrue(any("metadata does not match" in error for error in report.errors))

    def test_corrupted_output_is_detected(self) -> None:
        expected = {"s0.d0.c0": ((1.0, -2.0),)}
        observed = {"s0.d0.c0": ((1.0, -1.5),)}
        report = compare_reference_outputs(expected, observed, atol=1e-12, rtol=1e-12)
        self.assertFalse(report.valid)
        self.assertEqual(report.checked_elements, 2)
        self.assertEqual(report.mismatched_elements, 1)


class EvidenceAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = A2AGemmProblem(world_size=4, rows_per_peer=64, k=4096, n=4096)

    def record(
        self,
        family: str,
        interval: MeasurementInterval,
        *,
        numerical_check_passed: bool = True,
        environment_hash: str = "a" * 64,
        protocol_hash: str = "b" * 64,
        source_hash: str = "c" * 64,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            problem=self.problem,
            family=family,
            environment_hash=environment_hash,
            code_commit="d" * 40,
            protocol_hash=protocol_hash,
            source_hash=source_hash,
            numerical_check_passed=numerical_check_passed,
            interval=interval,
        )

    def test_clear_target_bound_evidence_is_admitted(self) -> None:
        host = self.record("host", MeasurementInterval(100, 105.0, 100.0, 110.0))
        candidate = self.record("fused", MeasurementInterval(100, 75.0, 70.0, 80.0))
        decision = admit_evidence(
            host,
            candidate,
            AdmissionPolicy(minimum_samples=30, minimum_gain_ratio=0.1),
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.selected_family, "fused")
        self.assertAlmostEqual(decision.conservative_gain_ratio or 0.0, 0.2)

    def test_touching_or_overlapping_intervals_fall_back_to_host(self) -> None:
        host = self.record("host", MeasurementInterval(100, 105.0, 100.0, 110.0))
        for candidate_upper in (100.0, 101.0):
            candidate = self.record(
                "fused",
                MeasurementInterval(100, 90.0, 80.0, candidate_upper),
            )
            decision = admit_evidence(host, candidate)
            self.assertFalse(decision.admitted)
            self.assertEqual(decision.selected_family, "host")

    def test_binding_mismatch_falls_back_to_host(self) -> None:
        host = self.record("host", MeasurementInterval(100, 105.0, 100.0, 110.0))
        mismatches = (
            self.record(
                "fused",
                MeasurementInterval(100, 75.0, 70.0, 80.0),
                environment_hash="e" * 64,
            ),
            self.record(
                "fused",
                MeasurementInterval(100, 75.0, 70.0, 80.0),
                protocol_hash="f" * 64,
            ),
            self.record(
                "fused",
                MeasurementInterval(100, 75.0, 70.0, 80.0),
                source_hash="0" * 64,
            ),
        )
        for candidate in mismatches:
            decision = admit_evidence(host, candidate)
            self.assertFalse(decision.admitted)
            self.assertIn("binding mismatch", decision.reason)

    def test_failed_numerics_or_insufficient_samples_fall_back(self) -> None:
        host = self.record("host", MeasurementInterval(100, 105.0, 100.0, 110.0))
        failed = self.record(
            "fused",
            MeasurementInterval(100, 75.0, 70.0, 80.0),
            numerical_check_passed=False,
        )
        sparse = self.record("fused", MeasurementInterval(5, 75.0, 70.0, 80.0))
        self.assertFalse(admit_evidence(host, failed).admitted)
        self.assertFalse(admit_evidence(host, sparse).admitted)

    def test_evidence_round_trip_is_lossless(self) -> None:
        record = self.record("host", MeasurementInterval(100, 105.0, 100.0, 110.0))
        self.assertEqual(EvidenceRecord.from_dict(record.to_dict()), record)


class AssuranceSchemaTest(unittest.TestCase):
    def test_schemas_are_strict_and_versioned(self) -> None:
        schema_root = Path(__file__).resolve().parents[1] / "spec" / "schemas"
        expected = {
            "evidence-record.schema.json": "edmir.evidence_record.v1",
            "experiment-spec.schema.json": "edmir.experiment_spec.v1",
        }
        for filename, schema_name in expected.items():
            value = json.loads((schema_root / filename).read_text())
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertFalse(value["additionalProperties"])
            self.assertEqual(value["properties"]["schema"]["const"], schema_name)


if __name__ == "__main__":
    unittest.main()
