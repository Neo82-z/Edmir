from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    PublicationEvent,
    build_fused_program,
    build_pipelined_program,
    validate_publication_trace,
    validate_slot_correspondence,
)


ROOT = Path(__file__).resolve().parents[1]
TLA_DIR = ROOT / "spec" / "tla"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalModelEvidenceTest(unittest.TestCase):
    def test_model_names_the_stable_assurance_properties(self) -> None:
        model = (TLA_DIR / "SlotLifecycle.tla").read_text()
        for property_name in (
            "ReadAfterPublish",
            "NoPrematureReuse",
            "CompleteReduction",
            "EventualCompletion",
        ):
            self.assertIn(property_name, model)

    def test_retained_tlc_report_matches_model_and_configs(self) -> None:
        summary_path = TLA_DIR / "results" / "summary.json"
        self.assertTrue(summary_path.is_file(), "run tools/run_tlc.py first")
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["schema"], "edmir.tlc_report.v1")
        self.assertTrue(summary["all_expectations_met"])
        self.assertEqual(
            summary["model_sha256"], sha256(TLA_DIR / summary["model"])
        )
        self.assertEqual(len(summary["cases"]), 5)
        for case in summary["cases"]:
            self.assertTrue(case["expectation_met"], case)
            self.assertEqual(
                case["config_sha256"], sha256(TLA_DIR / case["config"])
            )
            self.assertTrue((TLA_DIR / "results" / case["raw_output"]).is_file())

    def test_mutations_have_distinct_expected_counterexamples(self) -> None:
        summary = json.loads((TLA_DIR / "results" / "summary.json").read_text())
        by_config = {case["config"]: case for case in summary["cases"]}
        self.assertEqual(
            by_config["SlotLifecycleUnsafeReuse.cfg"]["expected_violation"],
            "NoPrematureReuse",
        )
        self.assertEqual(
            by_config["SlotLifecycleUnsafeRead.cfg"]["expected_violation"],
            "ReadAfterPublish",
        )
        self.assertEqual(
            by_config["SlotLifecycleUnsafeRelease.cfg"]["expected_violation"],
            "CompleteReduction",
        )


class RemotePublicationEvidenceTest(unittest.TestCase):
    def test_model_names_the_publication_properties(self) -> None:
        model = (TLA_DIR / "RemotePublication.tla").read_text()
        for property_name in (
            "AcquireAfterPublish",
            "ConsumeAfterAcquire",
            "CompleteConsumption",
            "NoOverwriteBeforeConsume",
            "EventualCompletion",
        ):
            self.assertIn(property_name, model)

    def test_retained_publication_report_matches_model_and_configs(self) -> None:
        result_dir = TLA_DIR / "publication_results"
        summary_path = result_dir / "summary.json"
        self.assertTrue(
            summary_path.is_file(), "run tools/run_publication_tlc.py first"
        )
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["schema"], "edmir.publication_tlc_report.v1")
        self.assertTrue(summary["all_expectations_met"])
        self.assertEqual(
            summary["model_sha256"], sha256(TLA_DIR / summary["model"])
        )
        self.assertEqual(len(summary["cases"]), 6)
        for case in summary["cases"]:
            self.assertTrue(case["expectation_met"], case)
            self.assertEqual(
                case["config_sha256"], sha256(TLA_DIR / case["config"])
            )
            self.assertTrue((result_dir / case["raw_output"]).is_file())

    def test_publication_mutations_have_distinct_counterexamples(self) -> None:
        summary = json.loads(
            (TLA_DIR / "publication_results" / "summary.json").read_text()
        )
        by_config = {case["config"]: case for case in summary["cases"]}
        expected = {
            "RemotePublicationUnsafeAcquire.cfg": "AcquireAfterPublish",
            "RemotePublicationUnsafeConsume.cfg": "ConsumeAfterAcquire",
            "RemotePublicationUnsafeRetire.cfg": "CompleteConsumption",
            "RemotePublicationUnsafeOverwrite.cfg": "NoOverwriteBeforeConsume",
        }
        self.assertEqual(
            {name: by_config[name]["expected_violation"] for name in expected},
            expected,
        )


class RemotePublicationCorrespondenceTest(unittest.TestCase):
    @staticmethod
    def event(action: str, epoch: int, consumer: int | None = None) -> PublicationEvent:
        return PublicationEvent(action, epoch, consumer)

    def test_complete_trace_projects_to_tla_actions(self) -> None:
        events = (
            self.event("BeginWrite", 1),
            self.event("Publish", 1),
            self.event("Acquire", 1, 1),
            self.event("Consume", 1, 1),
            self.event("Acquire", 1, 2),
            self.event("Consume", 1, 2),
            self.event("Retire", 1),
            self.event("BeginWrite", 2),
            self.event("Publish", 2),
            self.event("Acquire", 2, 1),
            self.event("Consume", 2, 1),
            self.event("Acquire", 2, 2),
            self.event("Consume", 2, 2),
            self.event("Retire", 2),
        )
        report = validate_publication_trace(
            events,
            epoch_count=2,
            consumer_count=2,
        )
        self.assertTrue(report.valid, report.errors)
        self.assertTrue(report.complete)
        self.assertEqual(report.retired_epochs, (1, 2))
        self.assertEqual(report.overwritten_epochs, (1,))
        self.assertEqual(report.actions, tuple(event.action for event in events))

    def test_tla_unsafe_acquire_maps_to_pub1(self) -> None:
        report = validate_publication_trace(
            (
                self.event("BeginWrite", 1),
                self.event("Acquire", 1, 1),
            ),
            epoch_count=1,
            consumer_count=1,
        )
        self.assertFalse(report.valid)
        self.assertIn("[PUB-1]", report.errors[0])

    def test_tla_unsafe_consume_maps_to_pub2(self) -> None:
        report = validate_publication_trace(
            (
                self.event("BeginWrite", 1),
                self.event("Publish", 1),
                self.event("Consume", 1, 1),
            ),
            epoch_count=1,
            consumer_count=1,
        )
        self.assertFalse(report.valid)
        self.assertIn("[PUB-2]", report.errors[0])

    def test_tla_unsafe_retire_maps_to_pub3(self) -> None:
        report = validate_publication_trace(
            (
                self.event("BeginWrite", 1),
                self.event("Publish", 1),
                self.event("Retire", 1),
            ),
            epoch_count=1,
            consumer_count=2,
        )
        self.assertFalse(report.valid)
        self.assertIn("[PUB-3]", report.errors[0])

    def test_tla_unsafe_overwrite_maps_to_pub4(self) -> None:
        report = validate_publication_trace(
            (
                self.event("BeginWrite", 1),
                self.event("Publish", 1),
                self.event("BeginWrite", 2),
            ),
            epoch_count=2,
            consumer_count=2,
        )
        self.assertFalse(report.valid)
        self.assertIn("[PUB-4]", report.errors[0])

    def test_safe_prefix_is_distinct_from_complete_execution(self) -> None:
        report = validate_publication_trace(
            (self.event("BeginWrite", 1), self.event("Publish", 1)),
            epoch_count=1,
            consumer_count=1,
            require_complete=False,
        )
        self.assertTrue(report.valid, report.errors)
        self.assertFalse(report.complete)


class SlotCorrespondenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = A2AGemmProblem(world_size=2, rows_per_peer=4, k=8, n=8)

    def test_reused_pipeline_slots_project_to_complete_epochs(self) -> None:
        program = build_pipelined_program(
            self.problem,
            chunk_rows=1,
            buffer_count=1,
        )
        report = validate_slot_correspondence(program)
        self.assertTrue(report.applicable)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(len(report.epochs), 16)
        self.assertTrue(all(epoch.actions[-1] == "Release" for epoch in report.epochs))
        self.assertEqual(
            {epoch.buffer for epoch in report.epochs},
            {"stage.r0.slot0", "stage.r1.slot0"},
        )

    def test_read_before_publication_breaks_correspondence(self) -> None:
        program = build_pipelined_program(
            self.problem,
            chunk_rows=2,
            buffer_count=1,
        )
        gemm = next(op for op in program.ops if op.kind == "gemm")
        broken = replace(
            program,
            ops=(gemm,) + tuple(op for op in program.ops if op != gemm),
        )
        report = validate_slot_correspondence(broken)
        self.assertTrue(report.applicable)
        self.assertFalse(report.valid)
        self.assertTrue(any("uninitialized read" in error for error in report.errors))

    def test_fused_internal_semaphore_is_explicitly_out_of_scope(self) -> None:
        report = validate_slot_correspondence(
            build_fused_program(self.problem, chunk_rows=2)
        )
        self.assertFalse(report.applicable)
        self.assertTrue(report.valid)
        self.assertEqual(report.epochs, ())


if __name__ == "__main__":
    unittest.main()
