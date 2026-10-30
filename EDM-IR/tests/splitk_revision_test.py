from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from edmir.compiler.factorized_selection import FactorizedSelector
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.splitk_revision import FEATURES, SplitKFit, SplitKRevision, split_features
from evaluate_splitk_revision import q_decisions
from factorized_selection_test import synthetic_ratios


class SplitKRevisionTest(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_ratios()

    def test_reference_control_matches_original_fit(self):
        original = FactorizedSelector.fit(self.data)
        fit = SplitKFit.fit(self.data, basis="original", loss="reference")
        for a, b in zip(fit.coefficients, original.split.coefficients):
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(fit.scales, original.split.scales):
            self.assertAlmostEqual(a, b, places=12)

    def test_labels_scaled_per_workload_do_not_change_fit(self):
        other = replace(self.data, rows=tuple(
            replace(r, measured_ms=r.measured_ms*r.k*r.rows_per_peer/r.n)
            for r in self.data.rows))
        for basis in FEATURES:
            for loss in ("reference", "pairwise"):
                a = SplitKFit.fit(self.data, basis=basis, loss=loss)
                b = SplitKFit.fit(other, basis=basis, loss=loss)
                for x, y in zip(a.coefficients, b.coefficients):
                    self.assertAlmostEqual(x, y, places=11)

    def test_non_split_components_and_scaler_are_fixed_controls(self):
        original = FactorizedSelector.fit(self.data)
        for basis in FEATURES:
            a = SplitKRevision.fit(self.data, basis=basis, loss="reference")
            b = SplitKRevision.fit(self.data, basis=basis, loss="pairwise")
            self.assertEqual(a.split.scales, b.split.scales)
            for name in ("gate", "group", "grouped_support", "calibration_offset"):
                self.assertEqual(getattr(a.effective(), name), getattr(original, name))
                self.assertEqual(getattr(b.effective(), name), getattr(original, name))
            target = replace(self.data, environment_id="calibration")
            calibrated = a.calibrate_offset(target)
            self.assertEqual(a.split, calibrated.split)
            self.assertEqual(calibrated.original, original.calibrate_offset(target))

    def test_host_grouped_labels_do_not_affect_split(self):
        other = replace(self.data, rows=tuple(
            replace(r, measured_ms=r.measured_ms*17) if r.is_host or r.n_tiles_per_cta == 2 else r
            for r in self.data.rows))
        for basis in FEATURES:
            self.assertEqual(SplitKFit.fit(self.data, basis=basis, loss="pairwise"),
                             SplitKFit.fit(other, basis=basis, loss="pairwise"))

    def test_zero_anchor_and_label_free_prediction(self):
        fit = SplitKFit.fit(self.data, basis="waves", loss="pairwise")
        for r in self.data.rows:
            c = ScheduleCandidate.from_measurement(r, self.data.environment_id)
            self.assertEqual(fit.predict(r), fit.predict(c))
            if r.family == "direct":
                self.assertEqual(split_features(c, "waves"), (0., 0., 0., 0.))
                self.assertEqual(fit.predict(c), 0.)

    def test_wave_basis_can_rank_an_interior_q_best(self):
        # Independent synthetic latency: diminishing parallel work plus serial overhead.
        data = replace(self.data, rows=tuple(
            replace(r, measured_ms=1/r.split_k_slices+0.08*r.split_k_slices)
            if not r.is_host else r for r in self.data.rows))
        fit = SplitKFit.fit(data, basis="waves", loss="pairwise", ridge=0.01)
        rows = [r for r in data.rows if r.rows_per_peer == 64 and r.k == 2048
                and r.n == 512 and not r.is_host and r.n_tiles_per_cta == 1]
        chosen = min(rows, key=fit.predict)
        self.assertIn(chosen.split_k_slices, (2, 4))
        self.assertGreater(fit.predict(max(rows, key=lambda r: r.split_k_slices)), fit.predict(chosen))

    def test_heldout_leakage_and_invalid_state_rejected(self):
        model = SplitKRevision.fit(self.data, basis="waves", loss="pairwise")
        with self.assertRaises(ValueError):
            q_decisions(model, self.data, self.data)
        for fit in (replace(model.split, coefficients=()), replace(model.split, scales=(0.,)*4),
                    replace(model.split, coefficients=(1., *model.split.coefficients[1:])),
                    replace(model.split, ridge=math.nan), replace(model.split, basis="unknown")):
            with self.assertRaises(ValueError):
                fit.validate()
        with self.assertRaises(ValueError):
            SplitKFit.fit(self.data, basis="waves", loss="unknown")


if __name__ == "__main__":
    unittest.main()
