from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    CalibrationProfile,
    ExecutionOverhead,
    RuntimeMeasurement,
    evaluate_cost_model,
    fit_execution_overhead,
    simulate_program,
)


class EvaluationTest(unittest.TestCase):
    def test_nonnegative_fit_and_heldout_metrics(self) -> None:
        base = CalibrationProfile(
            comm_latency_us=1.0,
            comm_bandwidth_gbps=50.0,
            gemm_launch_us=0.5,
            gemm_tflops=100.0,
        )
        truth = replace(
            base,
            execution_overhead=ExecutionOverhead(
                iteration_us=2.0,
                op_dispatch_us=0.4,
                event_record_us=0.2,
                event_wait_us=0.1,
                buffer_reuse_us=0.3,
            ),
        )
        specs = (
            ("train", 32, "sequential", 32, 4),
            ("train", 32, "pipelined", 16, 1),
            ("train", 32, "pipelined", 16, 2),
            ("train", 64, "sequential", 64, 4),
            ("train", 64, "pipelined", 16, 1),
            ("train", 64, "pipelined", 32, 4),
            ("heldout", 96, "sequential", 96, 4),
            ("heldout", 96, "pipelined", 32, 2),
            ("heldout", 96, "pipelined", 48, 4),
        )
        measurements = []
        for split, rows, strategy, chunk, buffers in specs:
            placeholder = RuntimeMeasurement(
                split=split,
                world_size=4,
                rows_per_peer=rows,
                k=256,
                n=256,
                dtype="float16",
                strategy=strategy,
                chunk_rows=chunk,
                buffer_count=buffers,
                measured_us=1.0,
            )
            measured = simulate_program(placeholder.program(), truth).makespan_us
            measurements.append(replace(placeholder, measured_us=measured))
        fitted = fit_execution_overhead(base, tuple(measurements), ridge=1e-10)
        report = evaluate_cost_model(fitted, tuple(measurements))
        self.assertLess(report["train"]["mape"], 1e-3)
        self.assertLess(report["heldout"]["mape"], 1e-3)
        self.assertEqual(report["direction_accuracy"], 1.0)
        self.assertEqual(report["heldout"]["policy_direction_accuracy"], 1.0)
        self.assertGreaterEqual(fitted.selection_min_gain_ratio, 0.0)
        self.assertGreaterEqual(fitted.execution_overhead.buffer_reuse_us, 0.0)


if __name__ == "__main__":
    unittest.main()
