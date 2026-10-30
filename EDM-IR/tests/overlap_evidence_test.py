from copy import deepcopy
from pathlib import Path
import math
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.overlap_evidence import (decision_margin, kernel_overlap, paired_overlap_cost,
                                            analyze_deepseek_trace, CONTEXT_FIELDS, union_intervals)


def profile():
    context = dict(zip(CONTEXT_FIELDS, ('A100-SXM4', 'NVLink', 4, 'fixed-build', 'fp16',
                      'nccl-cta', 'same-launch-policy', [4,512,4096,512], 'matched-work-wall')))
    return {'equivalent_work_verified': True, 'stage_completion_verified': True,
        'isolated': {'context': context, 'source_id': 'synthetic-unit-test', 'compute_us': 100,
                     'communication_us': 40, 'other_us': 0},
        'concurrent': {'context': deepcopy(context), 'source_id': 'synthetic-unit-test',
                       'compute_us': 150, 'communication_us': 60, 'other_us': 0,
                       'stage_intersection_us': 60}}


class OverlapEvidenceTests(unittest.TestCase):
    def test_common_timing_bias_preserves_choice(self):
        d = decision_margin(100, 110, 120, 132)
        self.assertAlmostEqual(d['differential_log_error'], 0)
        self.assertFalse(d['direction_wrong'])

    def test_small_mape_can_still_flip_close_candidates(self):
        d = decision_margin(100, 101, 101, 100)
        self.assertTrue(d['direction_wrong'])
        self.assertAlmostEqual(d['predicted_log_host_over_fused']-d['measured_log_host_over_fused'], d['differential_log_error'])
        self.assertIsNone(d['mechanism_cause'])

    def test_pair_accounting_does_not_imply_overlap_wins(self):
        r = paired_overlap_cost(profile())
        self.assertEqual(r['stage_intersection_us'], 60)
        self.assertEqual(r['saved_us'], -10)
        self.assertFalse(r['overlap_is_beneficial'])
        self.assertEqual(r['saved_us'], r['stage_intersection_us']-r['service_inflation_us']-r['extra_other_us'])

    def test_other_gaps_change_total_even_with_same_overlap(self):
        p = profile();p['concurrent'].update(compute_us=100, communication_us=40, stage_intersection_us=40, other_us=50)
        self.assertEqual(paired_overlap_cost(p)['saved_us'], -10)

    def test_incompatible_hardware_precision_or_unverified_work_rejected(self):
        for field, value in [('hardware', 'H800'), ('precision', 'bf16'), ('shape', [8,512,4096,512]), ('resource_partition', 'different')]:
            p = profile();p['concurrent']['context'][field] = value
            with self.assertRaises(ValueError):paired_overlap_cost(p)
        p = profile();p['equivalent_work_verified'] = False
        with self.assertRaises(ValueError):paired_overlap_cost(p)
        p = profile();p['stage_completion_verified'] = False
        with self.assertRaises(ValueError):paired_overlap_cost(p)

    def test_missing_provenance_or_impossible_intersection_rejected(self):
        p = profile();p['concurrent']['stage_intersection_us'] = 61
        with self.assertRaises(ValueError):paired_overlap_cost(p)
        p = profile();p['isolated']['source_id'] = ''
        with self.assertRaises(ValueError):paired_overlap_cost(p)
        with self.assertRaises(ValueError):decision_margin(1, 2, float('nan'), 3)
        self.assertTrue(math.isfinite(decision_margin(1e300, 1e-300, 1e-300, 1e300)['differential_log_error']))
        p = profile();p['concurrent']['stage_intersection_us'] = True
        with self.assertRaises(ValueError):paired_overlap_cost(p)

    def test_intersection_matches_discrete_oracle_and_is_order_invariant(self):
        rng = random.Random(52)
        for _ in range(40):
            a = [(v, v+rng.randrange(0, 5)) for v in rng.choices(range(20), k=12)]
            b = [(v, v+rng.randrange(0, 5)) for v in rng.choices(range(20), k=12)]
            aa = {i for lo, hi in a for i in range(lo, hi)};bb = {i for lo, hi in b for i in range(lo, hi)}
            r = kernel_overlap(a,b)
            self.assertEqual(r['kernel_intersection_us'], len(aa & bb))
            self.assertEqual(r['kernel_union_us'], len(aa | bb))
            self.assertEqual(r, kernel_overlap(list(reversed(a)), list(reversed(b))))
            self.assertIsNone(r['overlap_speedup'])

    def test_gpu_clocks_and_barriers_never_merged(self):
        def e(device, name, start, duration):
            return {'cat':'kernel','ph':'X','args':{'device':device,'context':1},'name':name,'ts':start,'dur':duration}
        events = [e(0,'void dpsk::ep::internode::dispatch()',0,10), e(1,'gemm',0,10), e(0,'ncclBarrier',0,10)]
        r = analyze_deepseek_trace({'traceEvents':events})
        self.assertEqual(len(r['groups']), 2)
        self.assertTrue(all(g['kernel_intersection_us']==0 for g in r['groups']))
        self.assertEqual(r['groups'][0]['event_counts']['excluded_nccl'], 1)

    def test_bad_and_oversized_intervals_fail_fast(self):
        with self.assertRaises(ValueError):union_intervals([(4,3)])
        with self.assertRaises(ValueError):union_intervals([(0,1)]*200001)


if __name__ == '__main__':unittest.main()
