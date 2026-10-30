from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from precision_candidate_selection_test import rows
from edmir.compiler import calibrated_candidate_selection as m


class CalibratedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_exact_times_are_stationary_and_zero_loss(self):
        y=torch.log(torch.tensor([[100.,100.7,250.]],dtype=torch.float64))
        s=y.clone().requires_grad_(True);mask=torch.ones_like(y,dtype=torch.bool);w=torch.tensor([[.25,.25,.5]],dtype=torch.float64)
        value,metrics=m.calibrated_loss(s,y,mask,w);value.backward()
        self.assertLess(abs(float(value.detach())),1e-12)
        self.assertLess(float(s.grad.abs().max()),1e-12)
        legacy=s.detach().clone().requires_grad_(True)
        old=m.old.losses(legacy,y,mask,w)[0];old.backward()
        self.assertGreater(float(legacy.grad.abs().max()),.001)

    def test_gradient_permutation_and_masked_nan(self):
        y=torch.tensor([[1.,1.01,float('nan')],[1.,2.,3.]],dtype=torch.float64)
        s=torch.tensor([[1.1,1.2,0.],[1.2,1.8,3.1]],dtype=torch.float64,requires_grad=True)
        mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool);w=mask.double()/mask.sum(1,keepdim=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:m.calibrated_loss(z,y,mask,w)[0],(s,)))
        v=m.calibrated_loss(s,y,mask,w)[0];v.backward();self.assertEqual(float(s.grad[0,2]),0)
        idx=[2,0,1]
        self.assertAlmostEqual(float(v.detach()),float(m.calibrated_loss(s[:,idx],y[:,idx],mask[:,idx],w[:,idx])[0].detach()),places=12)

    def test_pair_loss_is_symmetric_and_translation_invariant(self):
        y=torch.tensor([[1.,1.5,2.]],dtype=torch.float64);w=torch.tensor([[.5,.25,.25]],dtype=torch.float64);mask=torch.ones_like(y,dtype=torch.bool)
        e=torch.tensor([[0.,.1,-.2]],dtype=torch.float64)
        a=m.calibrated_loss(y+e,y,mask,w)[1]['symmetric_log_gap']
        b=m.calibrated_loss(y-e,y,mask,w)[1]['symmetric_log_gap']
        c=m.calibrated_loss(y+e+5,y,mask,w)[1]['symmetric_log_gap']
        self.assertAlmostEqual(float(a),float(b),places=12);self.assertAlmostEqual(float(a),float(c),places=12)

    def test_workload_counts_include_linear_M_and_tail(self):
        r=deepcopy(rows()[0]);r.update(candidate_id='host',is_host=True,shape=[4,64,4096,512])
        a=m.host_workload_features(r);r['shape'][1]=128;b=m.host_workload_features(r)
        self.assertEqual(a[-2:],[1.,1.] if isinstance(a,list) else (1.,1.))
        self.assertEqual(b[-2:],(2.,2.));r['shape'][1]=65;c=m.host_workload_features(r)
        self.assertEqual(c[-2],65/64);self.assertEqual(c[-1],2.)
        r['shape'][0]=8
        with self.assertRaises(ValueError):m.host_workload_features(r)

    def test_actual_muon_hidden_matrix_and_zero_initial_head_updates(self):
        net=m.old._network(40,8);opts=m.make_optimizers(net,'muon')
        self.assertIsInstance(opts[0],torch.optim.Muon);self.assertIsInstance(opts[1],torch.optim.AdamW)
        self.assertEqual([id(p) for p in opts[0].param_groups[0]['params']],[id(net.hidden.weight)])
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):fit=m.fit(rows(),optimizer='muon')
        self.assertEqual(fit['optimizer'],'muon');self.assertIsNone(fit['runs'][0]['history'][0]['parameter_updates']['readout.weight']['relative_update'])
        self.assertGreater(np.linalg.norm(fit['state']['readout.weight']),0.)

    def test_unseen_candidate_not_dropped_to_manufacture_success(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):fit=m.fit(rr)
        sample=deepcopy(rr[:2]);fit['candidate_support'].pop(m.support_key(sample[1]))
        p=m.predict(fit,sample)
        self.assertIn(sample[1]['candidate_id'],p['predicted_log_us']);self.assertIsNone(p['chosen'])
        self.assertIsNotNone(p['provisional_choice']);self.assertTrue(p['requires_validation'])
        self.assertEqual(p,m.predict(fit,list(reversed(sample))))
        for r in sample:r['latency_us']=1e8
        self.assertEqual(p,m.predict(fit,sample))

    def test_unknown_correctness_blocks_admission_and_dtype_is_not_borrowed(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):fit=m.fit(rr)
        sample=deepcopy(rr[:2]);sample[1]['validity']['status']='unknown'
        self.assertIsNone(m.predict(fit,sample)['chosen'])
        sample[1]['precision']['input']='bf16'
        with self.assertRaisesRegex(ValueError,'unsupported precision'):m.predict(fit,sample)

    def test_numpy_score_matches_network_and_selection_has_no_separate_head(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):fit=m.fit(rr)
        sample=rr[:2];p=m.predict(fit,sample);net=m.old._network(40,8)
        net.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in fit['state'].items()})
        x=(np.array([m.old.feature_vector(r) for r in sample])-fit['means'])/fit['scales']
        y=net(torch.tensor(x,dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose([p['predicted_log_us'][r['candidate_id']] for r in sample],y,atol=2e-6)
        self.assertAlmostEqual(sum(p['relative_log_scores'].values()),0.,places=12)
        self.assertEqual(p['chosen'],m.canonical_choice(p['predicted_log_us']))
        with self.assertRaisesRegex(ValueError,'shape leakage'):m.evaluate(fit,[{**r,'split':'evaluation'} for r in rr])

    def test_support_has_to_be_same_candidate_not_global_hull(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):fit=m.fit(rr)
        sample=rr[:2];fit['candidate_support'][m.support_key(sample[1])]=[list(rr[-1]['shape'])]
        p=m.predict(fit,sample)
        self.assertIsNone(p['chosen']);self.assertEqual(p['support'][sample[1]['candidate_id']]['reason'],'one_training_shape_only')


if __name__=='__main__':unittest.main()
