import math
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"tools")]
from edmir.compiler.execution_cost_selection import (
    ExecutionCostSelector, ExecutionGeometry, execution_vector, feature_vector,
    group_vector, training_pairs, direction_diagnostics,
)
from edmir.compiler.factorized_selection import FactorizedSelector
from edmir.compiler.host_baseline import PER_SOURCE_CUTLASS, PER_SOURCE_CUBLAS
from edmir.compiler.relative_selection import ScheduleCandidate
from factorized_selection_test import synthetic_ratios
from independent_selector_validation import candidate


class ExecutionCostTest(unittest.TestCase):
    def test_output_access_counts_against_explicit_predicated_tile_enumeration(self):
        # Includes partial M/N tiles. Count actual tensor accesses, not padded
        # elements or bytes fetched from an unknown memory-cache hierarchy.
        for g in (1, 2):
            c = candidate((4, 33, 512, 264), "splitk2"+("_groupedn2" if g == 2 else ""), "synthetic")
            geo = ExecutionGeometry.from_candidate(c)
            reads = writes = ctas = 0
            for mi in range(0, 33, 64):
                for ni in range(0, 264, 128*g):
                    for partition in range(2):
                        ctas += 1
                        for m in range(mi, mi+64):
                            for n in range(ni, ni+128*g):
                                if m < 33 and n < 264:
                                    writes += 2
                                    reads += 2 if partition else 0
            self.assertEqual((geo.total_ctas, geo.logical_output_read_bytes,
                              geo.logical_output_write_bytes), (ctas, reads, writes))
            self.assertEqual(geo.extra_output_bytes, reads+writes-2*33*264)
            self.assertEqual(geo.threads_per_cta, 128*g)

    def test_partition_rounding_and_q1_anchor(self):
        for g in (1, 2):
            suffix = "_groupedn2" if g == 2 else ""
            row = candidate((4, 33, 40, 264), "direct"+suffix, "synthetic")
            self.assertEqual(ExecutionGeometry.from_candidate(row).partition_k_span, 64)
            self.assertEqual(execution_vector(row), (0.,)*9)
        for q in (2, 4, 8, 16):
            row = candidate((4, 64, 8192, 512), f"splitk{q}", "synthetic")
            geo = ExecutionGeometry.from_candidate(row)
            self.assertEqual(geo.partition_k_span*q, 8192)
            self.assertEqual(geo.ordered_epilogue_stages, q)

    def test_group_depth_alias_and_partial_M_alias_are_removed(self):
        a = candidate((4, 64, 1024, 1024), "direct_groupedn2", "synthetic")
        b = candidate((4, 64, 8192, 1024), "splitk8_groupedn2", "synthetic")
        self.assertEqual(a.k/a.split_k_slices, b.k/b.split_k_slices)
        self.assertNotEqual(group_vector(a), group_vector(b))
        a = candidate((4, 32, 8192, 512), "splitk4", "synthetic")
        b = candidate((4, 64, 8192, 512), "splitk4", "synthetic")
        self.assertEqual(a.estimated_ctas_per_op, b.estimated_ctas_per_op)
        self.assertNotEqual(execution_vector(a), execution_vector(b))

    def test_balanced_pair_mass_is_independent_of_candidate_count(self):
        data = synthetic_ratios()
        for surface in data.validate().values():
            pairs = training_pairs(surface, True)
            self.assertAlmostEqual(sum(w for a, b, w in pairs if a.is_host or b.is_host), .5)
            self.assertAlmostEqual(sum(w for a, b, w in pairs if not a.is_host and not b.is_host), .5)

    def test_bidirectional_diagnostics_reject_both_directions_and_count_q_errors(self):
        a = direction_diagnostics({"host": 1.2, "a": 1.}, "host", ("host", "a"))
        b = direction_diagnostics({"host": 1., "a": 1.2}, "a", ("host", "a"))
        self.assertTrue(a["false_host"])
        self.assertFalse(a["false_fused"])
        self.assertTrue(b["false_fused"])
        self.assertFalse(b["false_host"])
        self.assertAlmostEqual(a["regret_all_observed"], b["regret_all_observed"])
        c = direction_diagnostics({"host": 2., "a": 1.2, "b": 1.}, "a", ("host", "a", "b"))
        self.assertFalse(c["false_host"] or c["false_fused"])
        self.assertAlmostEqual(c["regret_all_observed"], .2)

    def test_scaling_invariance_stationarity_and_training_order(self):
        data = synthetic_ratios()
        altered = replace(data, rows=tuple(reversed([
            replace(r, measured_ms=r.measured_ms*r.k*r.rows_per_peer/r.n) for r in data.rows])))
        for execution, balanced in ((False, False), (True, False), (True, True)):
            a = ExecutionCostSelector.fit(data, execution_terms=execution, balanced_host=balanced)
            b = ExecutionCostSelector.fit(altered, execution_terms=execution, balanced_host=balanced)
            np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-10, rtol=1e-10)
            self.assertLess(a.gradient_max_abs, 1e-9)

    def test_input_baseline_environment_and_fitted_state_rejections(self):
        data = synthetic_ratios()
        model = ExecutionCostSelector.fit(data)
        surface = next(iter(data.validate().values()))
        rows = {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in surface.items()}
        with self.assertRaises(TypeError):
            model.select(surface, baseline=data.baseline)
        with self.assertRaises(ValueError):
            model.select(rows, baseline=PER_SOURCE_CUBLAS)
        with self.assertRaises(ValueError):
            model.select({**rows, "host": replace(rows["host"], environment_id="other")}, baseline=data.baseline)
        for broken in (replace(model, coefficients=(float("nan"), *model.coefficients[1:])),
                       replace(model, scales=(0., *model.scales[1:])),
                       replace(model, means=(1., *model.means[1:])),
                       replace(model, trained_q=(1, 3)), replace(model, ridge=0.)):
            with self.assertRaises(ValueError):
                broken.select(rows, baseline=data.baseline)

    def test_support_matches_existing_contract_and_heldout_labels_are_invisible(self):
        data = synthetic_ratios()
        model = ExecutionCostSelector.fit(data)
        self.assertEqual(model.grouped_support, FactorizedSelector.fit(data).grouped_support)
        surface = next(iter(data.validate().values()))
        def static(rows):
            return {n: ScheduleCandidate.from_measurement(r, data.environment_id) for n, r in rows.items()}
        poisoned = {n: replace(r, measured_ms=r.measured_ms*(1000 if n == "host" else .001))
                    for n, r in surface.items()}
        self.assertEqual(model.select(static(surface), baseline=data.baseline),
                         model.select(static(poisoned), baseline=data.baseline))
        c = candidate((4, 32, 2048, 512), "splitk4_groupedn2", "synthetic")
        self.assertFalse(model.explain(c, baseline=data.baseline)["eligible"])


if __name__ == "__main__":
    unittest.main()
