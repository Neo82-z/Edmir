from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    CalibrationProfile,
    CommSample,
    GemmSample,
    PairwiseCommCurve,
    fit_gemm_profile,
    fit_local_copy_profile,
    fit_pairwise_nccl_profile,
    parse_nccl_tests_log,
    parse_gemm_benchmark_log,
    parse_local_copy_benchmark_log,
)


NCCL_LOG = """\
# nccl-tests synthetic fixture
#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
        1024           256     float    none      -1     4.00    0.26    0.26       0     4.10    0.25    0.25       0
        2048           512     float    none      -1     6.00    0.34    0.34       0     6.10    0.34    0.34       0
        4096          1024     float    none      -1    10.00    0.41    0.41       0    10.10    0.41    0.41       0
"""


def base_profile() -> CalibrationProfile:
    return CalibrationProfile(
        comm_latency_us=1.0,
        comm_bandwidth_gbps=50.0,
        gemm_launch_us=0.5,
        gemm_tflops=100.0,
        source="test-base",
    )


class CalibrationTest(unittest.TestCase):
    def test_old_profile_defaults_to_zero_runtime_overhead(self) -> None:
        restored = CalibrationProfile.from_dict(base_profile().to_dict())
        self.assertEqual(restored.execution_overhead.iteration_us, 0.0)
        self.assertEqual(restored.gemm_samples, ())

    def test_parse_nccl_tests_uses_out_of_place_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(NCCL_LOG, encoding="utf-8")
            samples = parse_nccl_tests_log(path)
        self.assertEqual(samples[0], CommSample(size_bytes=1024, latency_us=4.0))
        self.assertEqual(samples[-1], CommSample(size_bytes=4096, latency_us=10.0))

    def test_fit_profile_creates_both_pair_directions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_0_2_default_sendrecv.txt"
            path.write_text(NCCL_LOG, encoding="utf-8")
            profile = fit_pairwise_nccl_profile(base_profile(), directory)
        self.assertEqual(
            [(curve.src_rank, curve.dst_rank) for curve in profile.pairwise_comm_curves],
            [(0, 2), (2, 0)],
        )
        self.assertEqual(profile.comm_us(1536, src_rank=0, dst_rank=2), 5.0)
        self.assertEqual(profile.comm_us(1536, src_rank=1, dst_rank=2), 1.03072)

    def test_profile_curve_json_round_trip(self) -> None:
        profile = replace(
            base_profile(),
            pairwise_comm_curves=(
                PairwiseCommCurve(
                    src_rank=0,
                    dst_rank=1,
                    samples=(CommSample(1024, 3.0), CommSample(2048, 5.0)),
                ),
            ),
        )
        # Build through the public JSON form; tuples are accepted by json.dumps.
        restored = CalibrationProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)

    def test_curve_validation_rejects_unsorted_samples(self) -> None:
        profile = base_profile()
        curve = PairwiseCommCurve(0, 1, (CommSample(2048, 4.0), CommSample(1024, 3.0)))
        with self.assertRaisesRegex(ValueError, "increasing"):
            CalibrationProfile(
                comm_latency_us=profile.comm_latency_us,
                comm_bandwidth_gbps=profile.comm_bandwidth_gbps,
                gemm_launch_us=profile.gemm_launch_us,
                gemm_tflops=profile.gemm_tflops,
                pairwise_comm_curves=(curve,),
            ).validate()

    def test_gemm_log_aggregates_devices_and_profile_interpolates_m(self) -> None:
        log = "\n".join(
            (
                "torch preamble",
                '{"device": 0, "m": 16, "n": 4096, "k": 4096, "median_us": 28.0}',
                '{"device": 1, "m": 16, "n": 4096, "k": 4096, "median_us": 30.0}',
                '{"device": 0, "m": 32, "n": 4096, "k": 4096, "median_us": 40.0}',
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemm.txt"
            path.write_text(log, encoding="utf-8")
            samples = parse_gemm_benchmark_log(path)
            profile = fit_gemm_profile(base_profile(), path)
        self.assertEqual(samples[0], GemmSample("float16", 16, 4096, 4096, 29.0))
        self.assertEqual(profile.gemm_us(16, 4096, 4096), 29.0)
        self.assertEqual(profile.gemm_us(24, 4096, 4096), 34.5)
        self.assertNotEqual(
            profile.gemm_us(16, 4096, 4096, dtype="float32"),
            29.0,
        )

    def test_local_copy_log_aggregates_and_interpolates_size(self) -> None:
        log = "\n".join(
            (
                '{"device":0,"size_bytes":1024,"median_us":2.0}',
                '{"device":1,"size_bytes":1024,"median_us":4.0}',
                '{"device":0,"size_bytes":2048,"median_us":5.0}',
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.txt"
            path.write_text(log, encoding="utf-8")
            samples = parse_local_copy_benchmark_log(path)
            profile = fit_local_copy_profile(base_profile(), path)
        self.assertEqual(samples[0], CommSample(1024, 3.0))
        self.assertEqual(profile.comm_us(1024, local=True), 3.0)
        self.assertEqual(profile.comm_us(1536, local=True), 4.0)


if __name__ == "__main__":
    unittest.main()
