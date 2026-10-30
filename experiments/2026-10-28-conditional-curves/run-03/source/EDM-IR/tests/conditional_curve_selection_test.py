from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import conditional_curve_selection as m


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads((ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json').read_text())['p4']
        fold={tuple(s['shape']):s['fold'] for s in data['folds']}
        cls.train=[r for r in data['rows'] if fold[tuple(r['shape'])]!=0]
        cls.test=[{**r,'split':'evaluation'} for r in data['rows'] if r['shape']==[4,512,4096,512]]
        cls.prior=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-0.json').read_text())['model']
        cls.model=m.fit(cls.train,cls.prior)

    def test_convex_weights_and_no_extrapolation(self):
        knots=[{'k':2,'v':10.,'source_id':'a'},{'k':8,'v':40.,'source_id':'b'}]
        d=m.interpolate(knots,4,'v')
        self.assertAlmostEqual(d['value_us'],20.)
        self.assertEqual(d['anchor_sources'],['a','b'])
        self.assertAlmostEqual(sum(d['weights']),1.)
        self.assertIsNone(m.interpolate(knots,16,'v'))
        self.assertIsNone(m.interpolate(knots,1,'v'))

    def test_bounded_and_invalid_knots(self):
        with self.assertRaises(ValueError):m.interpolate([{'k':2,'v':1.}]*257,2,'v')
        with self.assertRaises(ValueError):m.interpolate([{'k':2,'v':1.},{'k':2,'v':2.}],2,'v')
        with self.assertRaises(ValueError):m.interpolate([{'k':2,'v':math.nan}],2,'v')

    def test_training_and_candidate_order_invariance(self):
        self.assertEqual(self.model,m.fit(list(reversed(self.train)),self.prior))
        p=m.predict(self.model,self.test)
        self.assertEqual(p,m.predict(self.model,list(reversed(self.test))))
        single=m.predict(self.model,self.test[:1])
        n=self.test[0]['candidate_id']
        self.assertEqual(single['predicted_log_us'][n],p['predicted_log_us'][n])

    def test_query_labels_do_not_change_scores(self):
        p=m.predict(self.model,self.test);r=deepcopy(self.test)
        for row in r:row['latency_us']=1e9
        self.assertEqual(p,m.predict(self.model,r))

    def test_no_cross_candidate_bracket(self):
        model=deepcopy(self.model);row=self.test[0];model['curves'].pop(m.curve_key(row))
        p=m.predict(model,self.test);d=p['curve_details'][row['candidate_id']]
        self.assertFalse(d['curve_applied'])
        self.assertEqual(d['prediction_us'],d['prior_us'])
        self.assertEqual(set(p['predicted_log_us']),{r['candidate_id'] for r in self.test})

    def test_precision_execution_and_correctness_contracts(self):
        rows=deepcopy(self.test);rows[0]['precision']['input']='bf16'
        with self.assertRaisesRegex(ValueError,'precision'):m.predict(self.model,rows)
        rows=deepcopy(self.test);rows[0]['execution_context']='another GPU collection'
        with self.assertRaisesRegex(ValueError,'context'):m.predict(self.model,rows)
        rows=deepcopy(self.test);rows[0]['validity']['status']='unknown'
        self.assertIsNone(m.predict(self.model,rows)['chosen'])

    def test_leakage_and_wrong_training_prior_rejected(self):
        with self.assertRaisesRegex(ValueError,'leakage'):
            m.evaluate(self.model,[{**r,'split':'evaluation'} for r in self.train])
        prior=deepcopy(self.prior);prior['training_shapes']=prior['training_shapes'][:-1]
        with self.assertRaises(ValueError):m.fit(self.train,prior)

    def test_negative_residual_prediction_rejected_not_clipped(self):
        model=deepcopy(self.model)
        for k in model['curves'].values():
            for v in k:v['residual_us']=-1e9
        with self.assertRaisesRegex(ValueError,'nonpositive'):m.predict(model,self.test)

    def test_interval_model_allows_decreasing_slopes_without_negative_time(self):
        r=deepcopy(next(r for r in self.test if r['is_host']));r['shape']=[4,64,2048,512]
        state={'knots':[2048,4096,8192], 'fit':{'coefficients':[1,0,0,2,0,0,.5,0,0]}}
        times=[]
        for k in (2048,4096,8192):
            r['shape'][2]=k;times.append(m.interval_host_prediction(state,r))
        self.assertGreater(times[1],(2*times[0]+times[2])/3)
        self.assertTrue(all(a<b for a,b in zip(times,times[1:])))
        r['shape'][2]=16384;self.assertIsNone(m.interval_host_prediction(state,r))

    def test_host_only_change_preserves_every_fused_score(self):
        control=m.predict(m.fit(self.train,self.prior,mode='prior'),self.test)
        changed=m.predict(m.fit(self.train,self.prior,mode='host_interval'),self.test)
        for r in self.test:
            if not r['is_host']:
                n=r['candidate_id'];self.assertEqual(control['predicted_log_us'][n],changed['predicted_log_us'][n])


if __name__=='__main__':unittest.main()
