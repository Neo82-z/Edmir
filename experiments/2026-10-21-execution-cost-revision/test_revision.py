import math
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import review_revision
from review_revision import response_map
from edmir.compiler.execution_cost_selection import ExecutionCostSelector
from evaluate_execution_cost_selector import decision_gate, PRIORITY

sys.path.insert(0, str(review_revision.ROOT/"EDM-IR/tests"))
from factorized_selection_test import synthetic_ratios


class RevisionReviewTest(unittest.TestCase):
    def test_independent_response_predicts_actual_refit(self):
        data = synthetic_ratios()
        for balanced in (False, True):
            model = ExecutionCostSelector.fit(data, balanced_host=balanced)
            rows, response, checks = response_map(data, model)
            delta = np.cos(np.arange(len(rows)))*.02
            altered = replace(data, rows=tuple(replace(r, measured_ms=r.measured_ms*math.exp(float(d)))
                                               for r, d in zip(rows, delta)))
            refit = ExecutionCostSelector.fit(altered, balanced_host=balanced)
            np.testing.assert_allclose(refit.coefficients, np.asarray(model.coefficients)+response@delta,
                                       atol=1e-10, rtol=1e-10)
            self.assertLess(checks["coefficient_error"], 1e-10)

    def test_gate_checks_both_directions_and_missing_priority(self):
        metrics = {n: {"mean_regret": .03, "max_regret": .2, "host_regressions": 2}
                   for n in ("proposed", "train_best_fixed", "relative_supported", "interactions_joint")}
        records = [{"shape": list(s), "diagnostics": {"proposed": {
            "false_host": False, "false_fused": False, "regret_all_observed": 0.}}} for s in sorted(PRIORITY)]
        primary = {"metrics": metrics, "decisions": records}
        stress = {"metrics": metrics}
        self.assertTrue(decision_gate(primary, stress, "proposed")["passed"])
        for direction in ("false_host", "false_fused"):
            records[0]["diagnostics"]["proposed"][direction] = True
            self.assertFalse(decision_gate(primary, stress, "proposed")["passed"])
            records[0]["diagnostics"]["proposed"][direction] = False
        records[0]["diagnostics"]["proposed"]["regret_all_observed"] = .06
        self.assertFalse(decision_gate(primary, stress, "proposed")["passed"])
        with self.assertRaises(ValueError):
            decision_gate({"metrics": metrics, "decisions": records[:-1]}, stress, "proposed")


if __name__ == "__main__":
    unittest.main()
