from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import curvature_transfer_selection as m


class CurvatureTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads((ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json').read_text())['p4']
        folds={tuple(s['shape']):s['fold'] for s in data['folds']}
        cls.train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=0]
        cls.test=[{**r,'split':'evaluation'} for r in data['rows'] if r['shape']==[4,64,512,128]]
        cls.prior=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-0.json').read_text())['model']
        cls.model=m.fit(cls.train,cls.prior)

    def test_non_depth_coefficients_are_frozen(self):
        co=dict(zip(self.prior['fused']['names'],self.prior['fused']['coefficients']))
        self.assertEqual(self.model['stage_prior']['curvature_transfer']['non_depth_coefficients'],
            {n:v for n,v in co.items() if n not in ('fixed','partition_K')})

    def test_depth_curve_is_unchanged(self):
        depth=m.stage.fit(self.train,self.prior)
        self.assertEqual(depth['stage_prior']['fit'],self.model['stage_prior']['fit'])

    def test_intercept_weighted_gradient_is_zero(self):
        ss=m.stage.curves.baseline.old.valid_rows(self.train,'train');gradient=0.
        for surface in ss:
            rows=[r for r in surface if not r['is_host']]
            gradient+=sum((m.transferred_value(self.model['stage_prior'],r)-r['latency_us'])/1000/len(rows) for r in rows)
        self.assertAlmostEqual(gradient,0.,places=10)

    def test_common_intercept_preserves_raw_fused_differences(self):
        fixed=m.fit(self.train,self.prior,'fixed')
        deltas=[m.transferred_value(self.model['stage_prior'],r)-m.transferred_value(fixed['stage_prior'],r)
                for r in self.test if not r['is_host']]
        self.assertLess(max(deltas)-min(deltas),1e-9)

    def test_training_anchor_residuals_are_recomputed(self):
        rr=[r for r in self.train if r['shape']==self.train[0]['shape']]
        p=m.predict(self.model,rr)
        for r in rr:
            if not r['is_host']:self.assertAlmostEqual(math.exp(p['predicted_log_us'][r['candidate_id']]),r['latency_us'],places=9)

    def test_host_unchanged(self):
        before=m.predict(m.stage.fit(self.train,self.prior),self.test)
        self.assertEqual(before['predicted_log_us']['host'],m.predict(self.model,self.test)['predicted_log_us']['host'])

    def test_no_query_label_dependency(self):
        rows=deepcopy(self.test)
        for r in rows:r['latency_us']=1e12
        self.assertEqual(m.predict(self.model,self.test),m.predict(self.model,rows))

    def test_order_subset_and_serialization(self):
        a=m.predict(self.model,self.test)
        self.assertEqual(a,m.predict(self.model,list(reversed(self.test))))
        self.assertEqual(a,m.predict(json.loads(json.dumps(self.model)),self.test))
        self.assertEqual(self.model,m.fit(list(reversed(self.train)),self.prior))
        p=m.predict(self.model,[r for r in self.test if r['candidate_id'] in ('host','direct','splitk2')])
        self.assertEqual(p['predicted_log_us'],{n:a['predicted_log_us'][n] for n in p['predicted_log_us']})

    def test_precision_implementation_and_shape_isolation(self):
        rows=deepcopy(self.test);rows[0]['precision']['input']='bf16'
        with self.assertRaises(ValueError):m.predict(self.model,rows)
        rows=deepcopy(self.test);rows[0]['implementation_id']='different'
        with self.assertRaises(ValueError):m.predict(self.model,rows)
        with self.assertRaisesRegex(ValueError,'leakage'):
            m.evaluate(self.model,[{**r,'split':'evaluation'} for r in self.train])

    def test_invalid_state_rejected(self):
        for key,value in [('intercept_ms',-1.),('intercept_ms',float('nan')),('intercept_kkt_violation',1.)]:
            bad=deepcopy(self.model);bad['stage_prior']['curvature_transfer'][key]=value
            with self.assertRaises(ValueError):m.predict(bad,self.test)
        bad=deepcopy(self.model);bad['stage_prior']['curvature_transfer']['non_depth_coefficients'].pop('ctas')
        with self.assertRaises(ValueError):m.predict(bad,self.test)
        with self.assertRaises(ValueError):m.fit(self.train,self.prior,'unknown')


if __name__=='__main__':unittest.main()
