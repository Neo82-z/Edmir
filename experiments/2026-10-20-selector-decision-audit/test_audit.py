import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np
import run_audit
from candidate_model import DecisionModel, split_vector
from edmir.compiler.relative_selection import ScheduleCandidate

sys.path.insert(0, str(run_audit.ROOT/"EDM-IR/tests"))
from factorized_selection_test import synthetic_ratios


class DecisionAuditTest(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_ratios()

    def test_old_feature_alias_and_new_reference_anchor(self):
        from edmir.compiler.two_stage_selection import ScheduleMeasurement
        def row(m, q):
            return ScheduleMeasurement("test", "direct" if q == 1 else f"splitk{q}",
                                       4, m, 8192, 512, 1., 4*3*m*8192*2,
                                       4, q, 1, ((m+63)//64)*4*q)
        self.assertEqual(split_vector(row(64, 4), False), split_vector(row(256, 4), False))
        self.assertNotEqual(split_vector(row(64, 4), True), split_vector(row(256, 4), True))
        for m in (32, 64, 128, 256):
            self.assertEqual(split_vector(row(m, 1), True), (0.,)*6)

    def test_numerical_stationarity_and_workload_scale_invariance(self):
        scaled = replace(self.data, rows=tuple(
            replace(r, measured_ms=r.measured_ms*r.k*r.rows_per_peer/r.n)
            for r in self.data.rows))
        for joint in (False, True):
            for interactions in (False, True):
                with self.subTest(joint=joint, interactions=interactions):
                    a = DecisionModel(self.data, joint=joint, interactions=interactions)
                    b = DecisionModel(scaled, joint=joint, interactions=interactions)
                    self.assertLess(a.fit.gradient_max_abs, 1e-9)
                    np.testing.assert_allclose(a.fit.coefficients, b.fit.coefficients, atol=1e-10, rtol=1e-10)

    def test_candidate_interface_cannot_observe_eval_labels(self):
        surface = next(iter(self.data.validate().values()))
        def candidates(rows):
            return {n: ScheduleCandidate.from_measurement(r, self.data.environment_id) for n, r in rows.items()}
        poisoned = {n: replace(r, measured_ms=r.measured_ms*(100 if n == "direct" else .001))
                    for n, r in surface.items()}
        for joint in (False, True):
            model = DecisionModel(self.data, joint=joint, interactions=True)
            self.assertEqual(model.choose(candidates(surface)), model.choose(candidates(poisoned)))
            with self.assertRaises(TypeError):
                model.choose(surface)
            wrong = candidates(surface)
            wrong["host"] = replace(wrong["host"], environment_id="other-device")
            with self.assertRaises(ValueError):
                model.choose(wrong)

    def test_gate_rejects_worst_case_even_if_mean_improves(self):
        report = {"metrics": {
            "candidate": {"mean_regret": .01, "max_regret": .3, "host_regressions": 0},
            "relative_supported": {"mean_regret": .04, "max_regret": .29, "host_regressions": 2},
            "train_best_fixed": {"mean_regret": .05, "max_regret": .18, "host_regressions": 3},
        }, "decisions": []}
        self.assertFalse(run_audit.gate(report, "candidate")["passed"])


if __name__ == "__main__":
    unittest.main()
