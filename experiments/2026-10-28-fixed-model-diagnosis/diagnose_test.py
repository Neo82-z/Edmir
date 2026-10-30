from copy import deepcopy
import math
import unittest

import diagnose as d


class FrozenDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.folds, cls.models = d.load_inputs()

    def surface(self, shape):
        return [r for r in self.rows if tuple(r['shape'])==shape]

    def test_fitting_is_blocked(self):
        with d.forbid_training() as calls:
            with self.assertRaisesRegex(RuntimeError,'forbidden'):
                d.scorer.stage.additive._solve(None,None,None,None,None)
            self.assertEqual(calls,['training_attempt'])

    def test_exact_saved_contrasts_without_training(self):
        with d.forbid_training() as calls:
            for shape,a,b in d.TARGETS:
                item=d.contrast(self.models[self.folds[shape]],self.surface(shape),a,b)
                self.assertLess(item['reconstruction_error_us'],1e-8)
                self.assertEqual(item['local_curves'],{a:None,b:None})
                self.assertEqual(item['term_differences_us']['fixed'],0.)
            self.assertEqual(calls,[])

    def test_query_labels_cannot_change_scores_or_components(self):
        shape=d.TARGETS[0][0];model=self.models[self.folds[shape]]
        before=self.surface(shape);after=deepcopy(before)
        for r in after:r['latency_us']*=1000
        self.assertEqual(d.scorer.predict(model,before),d.scorer.predict(model,after))
        self.assertEqual([d.components(model,r) for r in before if not r['is_host']],
                         [d.components(model,r) for r in after if not r['is_host']])

    def test_local_contrast_has_unidentified_serial_depth_direction(self):
        result=d.local_structure(self.rows,self.models)
        for fold in result['per_fold']:
            self.assertEqual(fold['local_contrast_rank'],2)
            self.assertEqual(fold['serial_depth_null_residual'],0.)
        self.assertAlmostEqual(result['nondecreasing_gap_minimum_max_error_us'],5.)

    def test_equal_cta_count_does_not_imply_same_block_resources(self):
        rr={r['candidate_id']:r for r in self.surface((4,64,8192,1024))}
        a=d.execution_counts(rr['splitk4'],32);b=d.execution_counts(rr['splitk8_groupedn2'],32)
        self.assertEqual(a['split_ctas'],b['split_ctas'])
        self.assertEqual((a['threads_per_cta'],b['threads_per_cta']),(128,256))
        self.assertEqual((a['depth_k_iterations'],b['depth_k_iterations']),(64,32))
        self.assertEqual(a['logical_mnk_flops_per_gemm'],b['logical_mnk_flops_per_gemm'])
        self.assertEqual(a['padded_mnk_flops_per_gemm'],b['padded_mnk_flops_per_gemm'])

    def test_break_even_is_analytic_not_a_model_edit(self):
        shape,a,b=d.TARGETS[-1];model=self.models[self.folds[shape]];saved=deepcopy(model)
        item=d.contrast(model,self.surface(shape),a,b)
        term=item['term_differences_us']['depth_32_64']
        # Decreasing this negative term by the stated fraction reaches zero.
        relative=item['single_coefficient_sensitivity']['depth_32_64']['relative_coefficient_change_to_zero_raw_gap']
        self.assertAlmostEqual(item['predicted_gap_us']-term*relative,0.,places=10)
        self.assertEqual(model,saved)


if __name__=='__main__':unittest.main()
