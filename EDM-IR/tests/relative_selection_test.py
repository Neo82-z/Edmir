from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from edmir.compiler.measurement_dataset import load_selector_dataset, sha256
from edmir.compiler.relative_selection import RelativeCostSelector, ScheduleCandidate
from prepare_selector_dataset import build_dataset, nccl_records, positive_ms
from two_stage_selection_test import synthetic_surface

REPO = Path(__file__).resolve().parents[2]


class RelativeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_surface()
        self.model = RelativeCostSelector.fit(self.rows, environment_id="source")
        self.candidates = {
            r.family: ScheduleCandidate.from_measurement(r, "source")
            for r in self.rows if r.group_id == "m64-n512" and r.k == 4096
        }

    def test_inference_has_no_latency_and_rejects_measurement_objects(self):
        self.assertNotIn("measured_ms", {f.name for f in fields(ScheduleCandidate)})
        row = self.rows[0]
        self.assertEqual(
            ScheduleCandidate.from_measurement(row, "source"),
            ScheduleCandidate.from_measurement(replace(row, measured_ms=999.0), "source"),
        )
        with self.assertRaises(TypeError):
            self.model.predict_log_ratio(row)
        with self.assertRaises(TypeError):
            self.model.select({r.family: r for r in self.rows[:6]})

    def test_common_workload_latency_scale_does_not_change_relative_model(self):
        # Different positive scales for different workloads cancel only when
        # pairs are formed within workloads; they must not affect the scaler.
        scaled = tuple(replace(r, measured_ms=r.measured_ms * (
            (r.k / 512) ** 2 * r.rows_per_peer / 32)) for r in self.rows)
        other = RelativeCostSelector.fit(scaled, environment_id="source")
        self.assertEqual(self.model.means, other.means)
        for candidate in self.candidates.values():
            self.assertAlmostEqual(self.model.predict_log_ratio(candidate),
                                   other.predict_log_ratio(candidate), places=10)

    def test_offset_changes_admission_level_but_preserves_schedule_differences(self):
        calibration = tuple(replace(r, group_id="cal-" + r.group_id,
                                    measured_ms=r.measured_ms * (1 if r.is_host else 0.5))
                            for r in self.rows if r.k == 1024)
        calibrated = self.model.calibrate_offset(calibration, environment_id="target")
        target = {name: replace(c, environment_id="target") for name, c in self.candidates.items()}
        before = {name: self.model.predict_log_ratio(c, allow_environment_transfer=True)
                  for name, c in target.items()}
        after = {name: calibrated.predict_log_ratio(c) for name, c in target.items()}
        self.assertEqual(before["host"], after["host"])
        differences = [after[name] - before[name] for name in target if name != "host"]
        self.assertTrue(all(math.isclose(v, calibrated.calibration_offset) for v in differences))
        self.assertEqual(min((n for n in target if n != "host"), key=before.get),
                         min((n for n in target if n != "host"), key=after.get))
        with self.assertRaises(ValueError):
            self.model.calibrate_offset(self.rows, environment_id="source")

    def test_transfer_and_mixed_candidate_guards(self):
        target = {n: replace(c, environment_id="target") for n, c in self.candidates.items()}
        with self.assertRaises(ValueError):
            self.model.select(target)
        selected, _ = self.model.select(target, allow_environment_transfer=True)
        self.assertIn(selected.family, target)
        mixed = {**self.candidates, "host": target["host"]}
        with self.assertRaises(ValueError):
            self.model.select(mixed, allow_environment_transfer=True)
        mixed = {**self.candidates, "host": replace(self.candidates["host"], n=1024)}
        with self.assertRaises(ValueError):
            self.model.select(mixed)

    def test_invalid_latency_and_duplicate_workload_groups_rejected(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(ValueError):
                replace(self.rows[0], measured_ms=value).validate()
            with self.assertRaises(ValueError):
                positive_ms(value, unit="us")
        duplicates = tuple(replace(r, group_id="other") for r in self.rows[:6])
        with self.assertRaises(ValueError):
            RelativeCostSelector.fit(self.rows + duplicates, environment_id="source")
        with self.assertRaises(ValueError):
            RelativeCostSelector.fit(self.rows + self.rows[:1], environment_id="source")

    def test_nccl_units_and_unknown_validation_count(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nccl.txt"
            path.write_text("# header\n1024 64 float none -1 62.23 0.02 0.01 0 29.94 0.03 0.03 N/A\n")
            rows = nccl_records(path, environment_id="p4", world_size=4, source={"path": "nccl.txt"})
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["latency_ms"], 0.06223)
            self.assertEqual(rows[0]["wrong"], 0)
            self.assertIsNone(rows[1]["wrong"])
            self.assertEqual(rows[0]["size_bytes_per_rank_including_self"], 1024)
        self.assertAlmostEqual(positive_ms(16.55328, unit="us"), 0.01655328)

    def test_archive_normalization_is_reproducible_and_excludes_components(self):
        archived = REPO / "experiments/2026-10-20-offline-predictor/data"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dataset"
            manifest = build_dataset(REPO, out)
            for name in ("manifest.json", *manifest["files"]):
                self.assertEqual(sha256(out / name), sha256(archived / name))
            p4, calibration, scoring, _ = load_selector_dataset(out)
            self.assertEqual((len(p4), len(calibration), len(scoring)), (346, 44, 209))
            self.assertFalse({r.shape for r in calibration} & {r.shape for r in scoring})
            self.assertEqual(manifest["counts"]["component_rows"], 156)
            self.assertEqual(manifest["counts"]["scoring_overrides"], 10)
            records = [json.loads(line) for line in (out / "selector_measurements.jsonl").read_text().splitlines()]
            repeated_cal = [r for r in records if r["historical_role"] == "calibration"
                            and r["repeat_observation"]]
            self.assertEqual(len(repeated_cal), 2)
            self.assertTrue(all(r["scoring_override"] is None for r in repeated_cal))
            path = out / "selector_measurements.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_selector_dataset(out)


if __name__ == "__main__":
    unittest.main()
