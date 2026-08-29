from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    AnalyticalScheduleFeatures,
    ScheduleMeasurement,
    TwoStageSelector,
    evaluate_group_heldout,
    evaluate_two_stage_ablations,
    parse_schedule_family,
    read_schedule_measurements,
)


def measurement(
    group: str,
    family: str,
    m: int,
    k: int,
    n: int,
    measured_ms: float,
) -> ScheduleMeasurement:
    q, g = parse_schedule_family(family)
    replay = max(1, (n + 127) // 128 // g)
    ctas = 0 if family == "host" else max(1, (m + 63) // 64 * replay * q)
    return ScheduleMeasurement(
        group_id=group,
        family=family,
        world_size=4,
        rows_per_peer=m,
        k=k,
        n=n,
        measured_ms=measured_ms,
        remote_input_bytes=4 * 3 * m * k * 2,
        remote_a_replay=replay,
        split_k_slices=q,
        n_tiles_per_cta=g,
        estimated_ctas_per_op=ctas,
    )


def synthetic_surface() -> tuple[ScheduleMeasurement, ...]:
    rows = []
    groups = (("m32-n512", 32, 512), ("m64-n512", 64, 512), ("m64-n1024", 64, 1024))
    families = ("host", "direct", "splitk2", "splitk4", "splitk8", "splitk8_groupedn2")
    for group, m, n in groups:
        for k in (512, 1024, 2048, 4096, 8192, 16384):
            for family in families:
                q, g = parse_schedule_family(family)
                work = m * n * k / 1e9
                if family == "host":
                    latency = 0.50 + 0.10 * work
                else:
                    supply = ((m + 63) // 64) * max(1, (n + 127) // 128 // g) * q
                    latency = (
                        0.25
                        + 0.04 * work
                        + 0.12 * max(0.0, 24.0 / supply - 1.0)
                        + 0.008 * (q - 1)
                        + 0.02 * (g - 1)
                    )
                rows.append(measurement(group, family, m, k, n, latency))
    return tuple(rows)


class TwoStageSelectionTest(unittest.TestCase):
    def test_family_parser_and_analytical_features(self) -> None:
        self.assertEqual(parse_schedule_family("direct"), (1, 1))
        self.assertEqual(parse_schedule_family("splitk8_groupedn2"), (8, 2))
        row = measurement("g", "splitk8_groupedn2", 64, 8192, 1024, 0.7)
        features = AnalyticalScheduleFeatures.from_measurement(row)
        self.assertEqual(features.rho_n, 4)
        self.assertEqual(features.gamma, 32)
        self.assertEqual(features.split_k_depth, 1024)
        self.assertGreater(features.reduction_bytes, 0)

    def test_selector_is_explicitly_two_stage(self) -> None:
        rows = synthetic_surface()
        selector = TwoStageSelector.fit(rows, ridge=1.0)
        candidates = {
            row.family: row
            for row in rows
            if row.group_id == "m64-n512" and row.k == 4096
        }
        selected, report = selector.select(candidates)
        self.assertIn(selected.family, candidates)
        self.assertIn("predicted_host_ms", report)
        self.assertIn("predicted_fused_ms", report)
        self.assertIn("predicted_fused_family", report)

    def test_group_heldout_and_ablation_suite(self) -> None:
        rows = synthetic_surface()
        report = evaluate_group_heldout(rows)
        self.assertEqual(report["protocol"], "leave-one-declared-sweep-group-out")
        self.assertEqual(report["metrics"]["problems"], 18)
        self.assertEqual(len(report["fold_models"]), 3)
        suite = evaluate_two_stage_ablations(rows)
        self.assertEqual(
            set(suite["reports"]),
            {"full", "no_admission", "no_gamma", "no_rho", "no_reduction", "k_only"},
        )

    def test_csv_reader_uses_parent_directory_as_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory) / "heldout-group"
            group.mkdir()
            path = group / "results.csv"
            fields = (
                "family,rows_per_peer,k,n,median_ms,remote_input_bytes,"
                "remote_a_replay,split_k_slices,n_tiles_per_cta,estimated_ctas_per_op\n"
            )
            path.write_text(
                fields
                + "host,64,512,512,0.5,786432,1,1,1,0\n"
                + "direct,64,512,512,0.3,786432,4,1,1,4\n",
                encoding="utf-8",
            )
            rows = read_schedule_measurements((path,), world_size=4)
            self.assertEqual({row.group_id for row in rows}, {"heldout-group"})


if __name__ == "__main__":
    unittest.main()
