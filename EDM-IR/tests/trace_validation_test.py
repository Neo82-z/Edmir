from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import A2AGemmProblem, build_pipelined_program
from edmir.trace_validation import ObservedGpuEvent, read_observed_gpu_events, validate_trace


class TraceValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.program = build_pipelined_program(
            A2AGemmProblem(world_size=2, rows_per_peer=16, k=32, n=64),
            chunk_rows=16,
            buffer_count=2,
        )

    def events(self) -> tuple[ObservedGpuEvent, ...]:
        events = []
        clock = {0: 0.0, 1: 0.0}
        for op in self.program.ops:
            devices = {op.rank}
            if op.kind == "transfer" and op.peer_rank != op.rank:
                devices.add(int(op.peer_rank))
            for device in sorted(devices):
                start = clock[device]
                end = start + 1.0
                clock[device] = end + 1.0
                events.append(
                    ObservedGpuEvent(
                        iteration=3,
                        op_id=op.id,
                        kind="memcpy" if op.kind == "transfer" and op.peer_rank == op.rank else "kernel",
                        device_id=device,
                        start_us=start,
                        end_us=end,
                        copy_kind="Device-to-Device" if op.peer_rank == op.rank else "",
                    )
                )
        return tuple(events)

    def test_complete_ordered_trace_is_valid(self) -> None:
        report = validate_trace(self.program, self.events())
        self.assertTrue(report.valid, report.violations)
        self.assertEqual(report.iteration, 3)
        self.assertEqual(report.covered_ops, report.expected_ops)
        self.assertGreater(report.dependency_device_checks, 0)
        self.assertGreater(report.gpu_span_us, 0.0)
        self.assertGreaterEqual(report.min_dependency_slack_us or 0.0, 0.0)

    def test_missing_device_and_dependency_overlap_are_rejected(self) -> None:
        events = list(self.events())
        remote = next(
            op for op in self.program.ops if op.kind == "transfer" and op.peer_rank != op.rank
        )
        events = [
            event
            for event in events
            if not (event.op_id == remote.id and event.device_id == remote.peer_rank)
        ]
        first = events[0]
        events[-1] = replace(
            events[-1],
            start_us=first.start_us,
            end_us=first.end_us,
        )
        report = validate_trace(self.program, tuple(events))
        kinds = {violation.kind for violation in report.violations}
        self.assertFalse(report.valid)
        self.assertIn("missing_device_event", kinds)
        self.assertIn("dependency_order", kinds)

    def test_reader_filters_unmapped_and_non_d2d_memcpy(self) -> None:
        fields = (
            "iteration", "edm_op", "kind", "device_id", "start_us", "end_us", "copy_kind"
        )
        rows = (
            ("3", "op.x", "kernel", "0", "1", "2", ""),
            ("3", "op.x", "memcpy", "0", "2", "3", "Host-to-Device"),
            ("", "", "kernel", "0", "3", "4", ""),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(rows)
            events = read_observed_gpu_events(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].op_id, "x")


if __name__ == "__main__":
    unittest.main()
