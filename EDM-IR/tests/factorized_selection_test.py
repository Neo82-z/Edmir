from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from edmir.compiler.factorized_selection import FactorizedSelector, IdentifiedMeasurements
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS, PER_SOURCE_CUTLASS, MERGED_CUTLASS
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.relative_selection import ScheduleCandidate, strict_surfaces
from evaluate_factorized_selector import design_audit, decisions, fit_models
from independent_selector_validation import candidate
from two_stage_selection_test import measurement

REPO = Path(__file__).resolve().parents[2]


def synthetic_ratios():
    """Independent generating process, with only M=64 paired grouping labels."""
    rows = []
    for m in (32, 64, 128):
        for k in (512, 1024, 2048, 4096):
            for n in (256, 512, 1024):
                host = 1.0 + m*k*n/1e8
                direct_ratio = math.exp(-0.2+0.02*math.log2(m)-0.01*math.log2(k))
                for q in (1, 2, 4, 8):
                    base = "direct" if q == 1 else f"splitk{q}"
                    split_ratio = math.exp(-0.07*math.log2(q)*math.log2(k/512)+0.03*(q-1)*512/k)
                    value = host*direct_ratio*split_ratio
                    rows.append(measurement(f"m{m}-n{n}", base, m, k, n, value))
                    if m == 64:
                        rows.append(measurement(f"m{m}-n{n}", base+"_groupedn2", m, k, n,
                                                value*math.exp(0.02*math.log2(k/q)-0.15)))
                rows.append(measurement(f"m{m}-n{n}", "host", m, k, n, host))
    return IdentifiedMeasurements(tuple(rows), "synthetic", PER_SOURCE_CUTLASS)


class FactorizedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_ratios()
        self.model = FactorizedSelector.fit(self.data)

    def test_label_ratio_identity_and_reference_anchors(self):
        for surface in strict_surfaces(self.data.rows).values():
            for row in surface.values():
                if row.is_host:
                    continue
                base = surface[row.family.removesuffix("_groupedn2")]
                y = math.log(surface["direct"].measured_ms/surface["host"].measured_ms)
                y += math.log(base.measured_ms/surface["direct"].measured_ms)
                y += math.log(row.measured_ms/base.measured_ms)
                self.assertAlmostEqual(y, math.log(row.measured_ms/surface["host"].measured_ms), places=12)
                if row.family == "direct":
                    self.assertEqual(self.model.split.predict(row), 0.0)

    def test_group_labels_do_not_change_gate_or_split_fit(self):
        changed = replace(self.data, rows=tuple(
            replace(r, measured_ms=r.measured_ms*7) if r.n_tiles_per_cta == 2 else r
            for r in self.data.rows))
        fitted = FactorizedSelector.fit(changed)
        self.assertEqual(self.model.gate, fitted.gate)
        self.assertEqual(self.model.split, fitted.split)
        self.assertNotEqual(self.model.group, fitted.group)

    def test_common_workload_scaling_cancels_in_all_components(self):
        scaled = replace(self.data, rows=tuple(replace(
            r, measured_ms=r.measured_ms*(r.rows_per_peer*r.k/r.n)) for r in self.data.rows))
        other = FactorizedSelector.fit(scaled)
        for stage in ("gate", "split", "group"):
            a, b = getattr(self.model, stage), getattr(other, stage)
            for x, y in zip(a.coefficients, b.coefficients):
                self.assertAlmostEqual(x, y, places=11)

    def test_group_support_is_exact_m_not_same_cta_row_tile(self):
        # M32 and M64 both have one 64-row tile, but paired evidence is M64 only.
        a = candidate((4, 64, 2048, 512), "splitk4_groupedn2", "synthetic")
        b = candidate((4, 32, 2048, 512), "splitk4_groupedn2", "synthetic")
        self.assertEqual(a.estimated_ctas_per_op, b.estimated_ctas_per_op)
        self.assertIsNone(self.model.exclusion_reason(a))
        self.assertIsNotNone(self.model.exclusion_reason(b))
        self.assertIsNotNone(self.model.exclusion_reason(candidate(
            (4, 64, 8192, 512), "splitk4_groupedn2", "synthetic")))
        result = self.model.explain(b, baseline=PER_SOURCE_CUTLASS)
        self.assertFalse(result["eligible"])
        self.assertIsNone(result["log_ratio"])

    def test_calibration_never_expands_group_support_or_changes_rank_components(self):
        target = replace(self.data, environment_id="other")
        calibrated = self.model.calibrate_offset(target)
        self.assertEqual(calibrated.grouped_support, self.model.grouped_support)
        self.assertEqual(calibrated.split, self.model.split)
        self.assertEqual(calibrated.group, self.model.group)
        changed = replace(target, rows=tuple(
            replace(r, measured_ms=r.measured_ms*100) if r.family not in ("host", "direct") else r
            for r in target.rows))
        self.assertEqual(calibrated, self.model.calibrate_offset(changed))

    def test_baseline_identity_required_at_training_and_inference(self):
        c = candidate((4, 64, 2048, 512), "direct", "synthetic")
        for baseline in (PER_SOURCE_CUBLAS, MERGED_CUTLASS):
            with self.assertRaises(ValueError):
                self.model.explain(c, baseline=baseline)
            with self.assertRaises(ValueError):
                FactorizedSelector.fit(replace(self.data, baseline=baseline))
        with self.assertRaises(ValueError):
            replace(PER_SOURCE_CUTLASS, partition="one_PM_gemm").validate()

    def test_invalid_inputs_and_model_state_fail_closed(self):
        c = candidate((4, 64, 2048, 512), "direct", "synthetic")
        with self.assertRaises(TypeError):
            self.model.explain(self.data.rows[0], baseline=PER_SOURCE_CUTLASS)
        with self.assertRaises(ValueError):
            self.model.explain(replace(c, environment_id="new"), baseline=PER_SOURCE_CUTLASS)
        with self.assertRaises(ValueError):
            self.model.calibrate_offset(self.data)
        for broken in (
            replace(self.model, gate=replace(self.model.gate, scales=(0., 1., 1.))),
            replace(self.model, split=replace(self.model.split, coefficients=(1., *self.model.split.coefficients[1:]))),
            replace(self.model, minimum_fused_advantage=float("nan")),
            replace(self.model, group=None),
        ):
            with self.assertRaises(ValueError):
                broken.explain(c, baseline=PER_SOURCE_CUTLASS)
        with self.assertRaises(ValueError):
            self.model.select({"host": candidate((4, 32, 2048, 512), "host", "synthetic"), "direct": c},
                              baseline=PER_SOURCE_CUTLASS)
        no_pair = replace(self.data, rows=tuple(
            r for r in self.data.rows if not (r.family == "splitk4" and r.rows_per_peer == 64)))
        with self.assertRaisesRegex(ValueError, "same-q"):
            FactorizedSelector.fit(no_pair)

    def test_inference_decisions_do_not_read_scoring_labels(self):
        models = fit_models(self.data)
        test = replace(self.data, environment_id="new")
        before = decisions(models, test)
        after = decisions(models, replace(test, rows=tuple(
            replace(r, measured_ms=r.measured_ms*(100 if r.family == "direct" else 0.1))
            for r in test.rows)))
        self.assertEqual([d["choices"] for d in before], [d["choices"] for d in after])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            decisions(models, self.data)

    def test_design_audit_detects_exact_dependence(self):
        audit = design_audit(("a", "b", "sum"), [(x, y, x+y) for x in (1, 2, 4) for y in (2, 7)])
        self.assertEqual(audit["columns_including_intercept"], 4)
        self.assertEqual(audit["rank"], 3)

    def test_archived_baseline_and_grouped_support_are_not_relabelled(self):
        p4, cal, p8, manifest = load_identified_selector_dataset(
            REPO/"experiments/2026-10-20-offline-predictor/data")
        self.assertEqual({d.baseline for d in (p4, cal, p8)}, {PER_SOURCE_CUTLASS})
        model = FactorizedSelector.fit(p4)
        self.assertEqual({c.m for c in model.grouped_support}, {64})
        self.assertEqual([c.pairs for c in model.grouped_support], [20]*5)
        self.assertEqual(model.calibrate_offset(cal).grouped_support, model.grouped_support)
        self.assertEqual(manifest["identified_host_baseline"], PER_SOURCE_CUTLASS.to_dict())
        json.dumps(model.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
