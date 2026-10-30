import math
import unittest
from dataclasses import replace

import numpy as np
from test_audit import synthetic_ratios
from candidate_model import DecisionModel
from robustness_review import linear_response, force_choice


class RobustnessMathTest(unittest.TestCase):
    def test_response_is_exact_after_refitting_changed_labels(self):
        data = synthetic_ratios()
        model = DecisionModel(data, joint=True, interactions=True)
        rows, response, _, checks = linear_response(data, model)
        delta = np.sin(np.arange(len(rows)))*.03
        altered = replace(data, rows=tuple(
            replace(r, measured_ms=r.measured_ms*math.exp(float(d)))
            for r, d in zip(rows, delta)))
        refit = DecisionModel(altered, joint=True, interactions=True)
        np.testing.assert_allclose(refit.fit.coefficients,
                                   model.fit.coefficients+response@delta, atol=1e-10, rtol=1e-10)
        self.assertLess(checks["coefficient_error_against_svd"], 1e-10)

    def test_lp_matches_hand_solved_global_winner_constraints(self):
        scores = {"host": 0., "a": -.2, "b": .1}
        influence = {"host": np.array([0., 0.]), "a": np.array([1., 0.]), "b": np.array([0., 1.])}
        for target, expected in (("host", .2000001), ("a", 0.), ("b", .15000005)):
            certificate, delta = force_choice(scores, influence, target)
            self.assertTrue(certificate["success"])
            self.assertAlmostEqual(certificate["epsilon_log"], expected, places=8)
            changed = {n: s+float(influence[n]@delta) for n, s in scores.items()}
            self.assertEqual(min(changed, key=changed.get), target)
            self.assertLess(certificate["duality_gap"], 1e-8)
            self.assertLess(certificate["dual_sign_or_free_bound_violation"], 1e-8)
            self.assertLess(certificate["complementarity_max_abs"], 1e-8)


if __name__ == "__main__":
    unittest.main()
