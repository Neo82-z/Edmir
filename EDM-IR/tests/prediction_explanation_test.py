from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
sys.path.insert(0,str(ROOT/'experiments/2026-10-28-dependency-structure-review'))
from edmir.compiler import prediction_explanation as explanation
import review as previous


class PredictionExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.folds,cls.models = previous.diagnosis.load_inputs()
        cls.stress = json.loads((previous.diagnosis.SAVED.parent/'audit/all_M512-intercept.json').read_text())['model']

    def surface(self, shape): return [r for r in self.rows if tuple(r['shape'])==shape]

    def test_exact_explanation_all_saved_scores_without_fit(self):
        with previous.no_fitting() as calls:
            for shape,fold in self.folds.items():
                rr = self.surface(shape); a = explanation.explain(self.models[fold],rr)
                self.assertEqual(a['prediction'],previous.scorer.predict(self.models[fold],rr))
                for name,detail in a['explanations'].items():
                    total = math.fsum(detail['active_terms_us'].values())+detail['K_anchor_adjustment_us']
                    self.assertAlmostEqual(total,math.exp(a['prediction']['predicted_log_us'][name]),places=8)
            self.assertEqual(calls,[])

    def test_host_fallback_is_visible_and_not_an_active_tile_model(self):
        a = explanation.explain(self.stress,self.surface((4,512,4096,512)))
        h = a['explanations']['host']
        self.assertEqual(h['active_path'],'original_log_M_quadratic_K_host_prior')
        self.assertEqual(h['fallback_reason'],'no_same_M_N_candidate_precision_K_bracket')
        self.assertAlmostEqual(h['prediction_us'],835.998735798408,places=8)
        self.assertAlmostEqual(h['inactive_host_interval_us'],1053.4386835847904,places=8)
        self.assertFalse(h['inactive_value_is_used_for_selection'])

    def test_supported_host_uses_actual_interval_terms(self):
        a = explanation.explain(self.stress,self.surface((4,256,4096,512)))['explanations']['host']
        self.assertEqual(a['active_path'],'host_interval_tiles')
        self.assertIsNone(a['fallback_reason'])
        self.assertIn('additional_M_tiles',a['active_terms_us'])

    def test_query_labels_do_not_change_trace(self):
        rr = self.surface((4,512,4096,512)); changed = deepcopy(rr)
        for r in changed: r['latency_us'] = 1e20
        self.assertEqual(explanation.explain(self.stress,rr),explanation.explain(self.stress,changed))

    def test_order_serialization_and_model_immutability(self):
        model = deepcopy(self.stress); rr = self.surface((4,512,4096,512))
        a = explanation.explain(model,rr)
        self.assertEqual(model,self.stress)
        self.assertEqual(a,explanation.explain(model,list(reversed(rr))))
        self.assertEqual(a,explanation.explain(json.loads(json.dumps(model)),rr))

    def test_precision_and_execution_identity_are_checked(self):
        rr = deepcopy(self.surface((4,512,4096,512))); rr[0]['precision']['input'] = 'bf16'
        with self.assertRaises(ValueError): explanation.explain(self.stress,rr)
        rr = deepcopy(self.surface((4,512,4096,512))); rr[0]['execution_context'] = 'different-run'
        with self.assertRaises(ValueError): explanation.explain(self.stress,rr)

    def test_candidate_removal_does_not_change_other_explanations(self):
        rr = self.surface((4,512,4096,512))
        a = explanation.explain(self.stress,rr)['explanations']
        b = explanation.explain(self.stress,[r for r in rr if r['candidate_id'] in ('host','splitk2')])['explanations']
        self.assertEqual(b,{n:a[n] for n in b})


if __name__=='__main__': unittest.main()
