from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-fixed-model-diagnosis')]
import diagnose
from edmir.compiler import axis_residual_selection as axis


class AxisResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.folds,cls.bases=diagnose.load_inputs()
        cls.training=[r for r in cls.rows if cls.folds[tuple(r['shape'])]!=2]
        cls.query=[r for r in cls.rows if r['shape']==[4,128,512,512]]
        cls.model=axis.assemble(cls.bases[2],cls.training,'M_work_extension')

    def test_labels_not_read_and_candidate_order_invariant(self):
        expected=axis.predict(self.model,self.query)
        query=deepcopy(self.query)
        for r in query:r.pop('latency_us')
        self.assertEqual(expected,axis.predict(self.model,list(reversed(query))))

    def test_no_fitting(self):
        with diagnose.forbid_training() as calls:
            model=axis.assemble(self.bases[2],self.training,'M_work_extension')
            axis.predict(model,self.query)
        self.assertEqual(calls,[])

    def test_training_permutation_serialization_and_immutability(self):
        before=json.dumps([self.training,self.query,self.bases,self.model],sort_keys=True)
        reverse=axis.assemble(self.bases[2],list(reversed(self.training)),'M_work_extension')
        self.assertEqual(axis.predict(reverse,self.query),axis.predict(json.loads(json.dumps(self.model)),self.query))
        self.assertEqual(before,json.dumps([self.training,self.query,self.bases,self.model],sort_keys=True))

    def test_exact_training_set_and_observations_required(self):
        with self.assertRaises(ValueError):axis.assemble(self.bases[2],self.training+self.query)
        with self.assertRaises(ValueError):axis.assemble(self.bases[2],self.training[1:])
        rows=deepcopy(self.training);rows[0]['latency_us']+=1
        with self.assertRaises(ValueError):axis.assemble(self.bases[2],rows)

    def test_contract_isolation(self):
        for field,value in (('execution_context','another-gpu'),('implementation_id','different-kernel')):
            query=deepcopy(self.query)
            for r in query:r[field]=value
            with self.assertRaises(ValueError):axis.predict(self.model,query)
        query=deepcopy(self.query)
        for r in query:r['precision']['input']='bf16'
        with self.assertRaises(ValueError):axis.predict(self.model,query)

    def test_existing_K_route_retained_exactly(self):
        for shape in self.folds:
            if self.folds[shape]!=2:continue
            query=[r for r in self.rows if tuple(r['shape'])==shape]
            old=axis.base.predict(self.bases[2],query);new=axis.predict(self.model,query)
            for name,d in old['curve_details'].items():
                if d['curve_applied'] or d['host_interval_applied']:
                    self.assertEqual(old['predicted_log_us'][name],new['predicted_log_us'][name])

    def test_adding_candidate_does_not_change_other_scores(self):
        full=axis.predict(self.model,self.query)
        short=axis.predict(self.model,self.query[:-1])
        for name,value in short['predicted_log_us'].items():self.assertEqual(value,full['predicted_log_us'][name])

    def test_extrapolation_is_reported_and_not_admitted(self):
        p=axis.predict(self.model,self.query)
        self.assertTrue(any(d.get('applied') for d in p['axis_details'].values()))
        self.assertTrue(p['requires_validation']);self.assertIsNone(p['chosen'])
        self.assertFalse(p['compiler_admission'])
        for n,d in p['axis_details'].items():
            if d.get('applied'):self.assertFalse(p['support'][n]['supported'])

    def test_interpolant_affine_exactness_and_bounded_extension(self):
        points=[{'x':x,'residual_us':2*x+3,'latency_us':10*x,'shape':[4,x,512,128],'source_id':'test'} for x in (64,128)]
        self.assertEqual(axis._bracket(points,96)['residual_us'],195.)
        self.assertIsNone(axis._bracket(points,256))
        self.assertEqual(axis._bracket(points,256,extend=True)['residual_us'],515.)
        self.assertIsNone(axis._bracket(points,257,extend=True))
        self.assertIsNone(axis._bracket(points,32,extend=True))
        with self.assertRaises(ValueError):axis._bracket(points*129,96)
        with self.assertRaises(ValueError):axis._bracket(list(reversed(points)),96)

    def test_each_fused_M_extension_preserves_prior_curvature(self):
        p=axis.predict(self.model,self.query)
        for r in self.query:
            name=r['candidate_id'];d=p['axis_details'][name]
            if r['is_host'] or not d.get('applied'):continue
            t=d['transfers'][0]
            residual=sum(w*a['residual_us'] for w,a in zip(t['weights'],t['anchors']))
            self.assertAlmostEqual(d['prediction_us'],p['curve_details'][name]['prior_us']+residual)

    def test_explanation_times_match_scores(self):
        p=axis.predict(self.model,self.query)
        for name,value in p['predicted_log_us'].items():
            self.assertAlmostEqual(math.exp(value),p['curve_details'][name]['prediction_us'])


if __name__=='__main__':unittest.main()
