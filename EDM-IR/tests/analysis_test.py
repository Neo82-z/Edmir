from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir import EDMBuilder, MovementDesc  # noqa: E402
from ir.analysis import analyze_exposure, phase_duration_us  # noqa: E402
from ir.ops import COMM_ALL_GATHER, COMPUTE_GEMM, MOVEMENT_COPY, STREAM_ORDER  # noqa: E402


class AnalysisTest(unittest.TestCase):
    def test_phase_duration_prefers_phase_boundaries(self) -> None:
        builder = EDMBuilder(default_phase="decode")
        begin = builder.phase_begin("decode", 10.0)
        compute = builder.compute("decode_gemm", 20.0, 40.0, op=COMPUTE_GEMM)
        end = builder.phase_end("decode", 60.0)
        noise = builder.compute("unrelated_profiler_noise", 1000.0, 2000.0, op=COMPUTE_GEMM)

        builder.sequence(begin, compute, end, kind=STREAM_ORDER)
        graph = builder.build()

        self.assertEqual(noise.name, "unrelated_profiler_noise")
        self.assertEqual(phase_duration_us(graph), 50.0)

    def test_ecr_and_edmr_are_split(self) -> None:
        builder = EDMBuilder(default_phase="prefill")
        begin = builder.phase_begin("prefill", 0.0)
        copy = builder.movement(
            "h2d_activation",
            0.0,
            20.0,
            op=MOVEMENT_COPY,
            movement=MovementDesc(kind="h2d_copy", backend="cuda", size_bytes=1024),
        )
        comm = builder.comm(
            "tp_all_gather",
            20.0,
            30.0,
            op=COMM_ALL_GATHER,
            movement=MovementDesc(kind="all_gather", backend="nccl", size_bytes=2048),
        )
        compute = builder.compute("gemm", 30.0, 100.0, op=COMPUTE_GEMM)
        end = builder.phase_end("prefill", 100.0)
        builder.sequence(begin, copy, comm, compute, end, kind=STREAM_ORDER)

        exposure = analyze_exposure(builder.build())

        self.assertEqual(exposure.phase_time_us, 100.0)
        self.assertEqual(exposure.total_comm_us, 10.0)
        self.assertEqual(exposure.exposed_comm_us, 10.0)
        self.assertEqual(exposure.total_data_movement_us, 30.0)
        self.assertEqual(exposure.exposed_data_movement_us, 30.0)
        self.assertAlmostEqual(exposure.ecr, 0.1)
        self.assertAlmostEqual(exposure.edmr, 0.3)


if __name__ == "__main__":
    unittest.main()
