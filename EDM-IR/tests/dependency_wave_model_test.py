from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EDM-IR'))
from edmir.compiler import dependency_wave_model as model


def case(**kw):
    return replace(model.KernelCase('synthetic-sm80', 4, 2, 512, 8, 1, 1), **kw)


class DependencyWaveTests(unittest.TestCase):
    def test_all_fit_is_span_not_serial_mainloops(self):
        service = model.ServiceScenario(11., 3., 2., 'test_units', 'constant service fixture')
        for q in (1, 2, 4, 8, 16):
            c = case(q=q)
            for order in model.ADMISSION_ORDERS:
                result = model.simulate(c, service, slots=108, order=order)
                self.assertEqual(result['scenario_makespan'], 11 + 3 + 2*(q-1))
                # Summed slot waiting differs from the critical-path duration.
                expected_wait = (q-1)*3 + (q-1)*(q-2)/2*2
                self.assertEqual(result['resident_wait_slot_time'], expected_wait)

    def test_one_slot_serializes_all_service(self):
        service = model.ServiceScenario(7., 2., 3., 'test_units', 'one slot fixture')
        for q in (1, 2, 4):
            c = case(m=130, q=q)
            expected = c.ctas*7 + c.chains*(2+(q-1)*3)
            for order in model.ADMISSION_ORDERS:
                result = model.simulate(c, service, slots=1, order=order)
                self.assertEqual(result['scenario_makespan'], expected)
                self.assertEqual(result['resident_wait_slot_time'], 0)

    def test_q_one_exact_wave_formula_and_boundary(self):
        service = model.ServiceScenario(4., 1., 3., 'test_units', 'q1 fixture')
        for chains in (1, 2, 3, 4, 5, 17):
            c = case(m=chains*64)
            for slots in (1, 2, 4, 8):
                actual = model.simulate(c, service, slots=slots)['scenario_makespan']
                self.assertEqual(actual, math.ceil(chains/slots)*5)

    def test_scaling_units_preserves_time_ratios(self):
        c = case(m=130, q=4)
        a = model.ServiceScenario(7, 2, 3, 'test_units', 'fixture')
        b = model.ServiceScenario(70, 20, 30, 'scaled_test_units', 'scaled fixture')
        for order in model.ADMISSION_ORDERS:
            x = model.simulate(c, a, slots=3, order=order)
            y = model.simulate(c, b, slots=3, order=order)
            for key in ('scenario_makespan', 'resident_wait_slot_time', 'lower_bound'):
                self.assertAlmostEqual(y[key], 10*x[key])

    def test_bounds_against_independent_exhaustive_admission_recurrence(self):
        # Enumerate all topological CTA admission orders for two 2-part chains.
        from itertools import permutations
        service = model.ServiceScenario(5., 2., 3., 'test_units', 'fixture')
        c = case(m=66, q=2)
        bound = model.conditional_bounds(c, service, slots=2)['lower_bound']
        observed = []
        for order in permutations([(0, 0), (0, 1), (1, 0), (1, 1)]):
            if any(order.index((t, 0)) > order.index((t, 1)) for t in (0, 1)):
                continue
            free = [0., 0.]; finishes = {}
            for tile, partition in order:
                lane = min(range(2), key=lambda i: free[i])
                ready = free[lane]+5
                end = max(ready, finishes.get((tile, partition-1), 0)) + (2 if partition == 0 else 3)
                free[lane] = end; finishes[tile, partition] = end
            observed.append(max(finishes.values()))
        self.assertTrue(all(x >= bound for x in observed))
        for order in model.ADMISSION_ORDERS:
            self.assertIn(model.simulate(c, service, slots=2, order=order)['scenario_makespan'], observed)

    def test_grouping_same_ctas_different_chains_and_resources(self):
        a, b = case(m=64, n=1024, k=8192, q=4), case(m=64, n=1024, k=8192, q=8, group_n=2)
        self.assertEqual(a.ctas, b.ctas)
        self.assertEqual(a.chains, 2*b.chains)
        self.assertEqual(a.counts()['threads_per_cta']*2, b.counts()['threads_per_cta'])
        self.assertEqual(model.resource_envelope(a)['resident_ctas_per_sm_upper_bound'], 4)
        self.assertEqual(model.resource_envelope(b)['resident_ctas_per_sm_upper_bound'], 2)
        self.assertIsNone(model.resource_envelope(a)['actual_resident_ctas_per_sm'])

    def test_tail_counts_use_valid_bytes_padded_tasks(self):
        c = case(m=3, n=136, q=2, group_n=2)
        counts = c.counts()
        self.assertEqual(c.chains, 1)
        self.assertEqual(counts['logical_output_write_bytes'], 2*3*136*2)
        self.assertEqual(counts['logical_extra_output_read_bytes'], 2*3*136)
        self.assertLess(counts['valid_output_fraction'], 1)

    def test_missing_service_never_fabricates_prediction_or_choice(self):
        result = model.unresolved_prediction(case())
        self.assertIsNone(result['prediction_us'])
        self.assertIsNone(result['chosen'])
        self.assertFalse(result['compiler_admission'])

    def test_reject_bad_precision_partitions_and_unbounded_traversal(self):
        for c in (case(precision='bf16'), case(k=513), case(n=7), case(q=3),
                  case(m=1_048_576, n=1_048_576), case(m=True), case(environment_id='')):
            with self.assertRaises(ValueError):
                c.validate()
        with self.assertRaises(TypeError):
            model.KernelCase.from_candidate({}, precision='fp16_fp32acc_fp16out')

    def test_service_validation(self):
        c = case()
        for bad in (-1, 0, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                model.simulate(c, model.ServiceScenario(bad, 1, 1, 'u', 'fixture'), slots=1)
        service = model.ServiceScenario(1, 1, 1, 'u', 'fixture')
        for slots in (0, -1, True, 32769):
            with self.assertRaises(ValueError): model.simulate(c, service, slots=slots)
        with self.assertRaises(ValueError): model.simulate(c, service, slots=1, order='CUDA_order')


if __name__ == '__main__':
    unittest.main()
