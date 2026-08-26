from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    FusedMeasurement,
    fit_and_evaluate_fused_boundary,
    read_fused_measurements,
)


def measurement(family: str, k: int, measured_ms: float, *, m: int = 64, n: int = 64):
    return FusedMeasurement(
        family=family,
        rows_per_peer=m,
        k=k,
        n=n,
        measured_ms=measured_ms,
        remote_input_bytes=12 * m * k * 2,
        remote_a_replay=(n + 127) // 128,
        split_k_slices=8 if family == "splitk8" else 1,
        estimated_ctas_per_op=8 if family == "splitk8" else 1,
    )


def shape(k: int, winner: str, *, m: int = 64, n: int = 64):
    times = {
        "host": 0.6,
        "direct": 0.3 if winner == "direct" else 0.8,
        "splitk8": 0.3 if winner == "splitk8" else 0.5,
    }
    return tuple(measurement(family, k, value, m=m, n=n) for family, value in times.items())


class FusedEvaluationTest(unittest.TestCase):
    def test_fit_reports_heldout_direction_and_keeps_admission_closed(self) -> None:
        train = (*shape(256, "direct"), *shape(1024, "direct"), *shape(2048, "splitk8"))
        heldout = (
            *shape(512, "direct", m=32),
            *shape(4096, "splitk8", n=256),
        )
        report = fit_and_evaluate_fused_boundary(train, heldout)
        self.assertEqual(report["rule"]["max_direct_k"], 1024)
        self.assertEqual(report["rule"]["min_split_k"], 2048)
        self.assertFalse(report["rule"]["default_compiler_admission"])
        self.assertEqual(report["train"]["direction_accuracy"], 1.0)
        self.assertEqual(report["heldout"]["direction_accuracy"], 1.0)
        self.assertEqual(report["heldout"]["mean_selection_regret"], 0.0)

    def test_csv_reader_rejects_incomplete_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "family,rows_per_peer,k,n,median_ms,remote_input_bytes,"
                "remote_a_replay,split_k_slices,estimated_ctas_per_op\n"
                "direct,64,256,64,0.3,393216,1,1,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing families"):
                read_fused_measurements((path,))


if __name__ == "__main__":
    unittest.main()
