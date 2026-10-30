from copy import deepcopy
from pathlib import Path
import sys, unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from shape_evaluation_test import numerical_rows
from edmir.compiler import precision_candidate_selection as m
from edmir.compiler import shape_decision_model as old


def rows():
    result=numerical_rows()
    for r in result:
        r['precision']={'input':'fp16','weight':'fp16','output':'fp16','accumulator':'fp32',
                        'partial':'none' if r['is_host'] else 'fp16','accumulator_evidence':'source'}
        r['validity']={'status':'passed','source_id':'synthetic_unit_test'}
        r['validity']['contract_sha256']=m.evidence_contract(r)
    return result


class PrecisionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_masked_nan_labels_have_zero_mass_and_gradient(self):
        s=torch.tensor([[1.,2.,float('nan')],[3.,2.,1.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1.,1.01,float('nan')],[3.,2.,1.]],dtype=torch.float64)
        mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool);w=mask.double()/mask.sum(1,keepdim=True)
        value,metrics=m.losses(s,y,mask,w);value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(s.grad).all());self.assertEqual(s.grad[0,2],0.)
        self.assertGreater(float(metrics['cross_entropy'].detach()),0.)

    def test_gradcheck_and_permutation_with_ties(self):
        s=torch.tensor([[1.3,1.1,2.],[1.,2.,3.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1.,1.,2.1],[1.2,2.3,99.]],dtype=torch.float64)
        mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool);w=mask.double()/mask.sum(1,keepdim=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:m.losses(z,y,mask,w)[0],(s,),eps=1e-6,atol=1e-4))
        perm=[2,0,1];a=m.losses(s,y,mask,w)[0];b=m.losses(s[:,perm],y[:,perm],mask[:,perm],w[:,perm])[0]
        self.assertAlmostEqual(float(a.detach()),float(b.detach()),places=10)

    def test_zero_ce_reproduces_existing_data_loss(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(2,3,4));theta=rng.normal(size=5)*.1
        y=np.array([[1.,1.,2.],[1.5,2.,0.]]);mask=np.array([[1,1,1],[1,1,0]],bool);w=np.array([[.5,.25,.25],[.5,.5,0.]])
        before=old.objective(theta,x,y,mask,w,0,1.,1.)[0]
        after=m.losses(torch.tensor(x@theta[:4]+theta[4]),torch.tensor(y),torch.tensor(mask),torch.tensor(w),ce_weight=0.)[0]
        self.assertAlmostEqual(before,float(after),places=10)

    def test_ce_is_soft_cost_target_not_hard_winner(self):
        y=torch.log(torch.tensor([[100.,100.1,200.]],dtype=torch.float64));mask=torch.ones_like(y,dtype=torch.bool)
        _,metrics=m.losses(y,y,mask,torch.ones_like(y)/3)
        self.assertLess(abs(float(metrics['kl_to_cost_target'])),1e-12)
        self.assertGreater(float(metrics['target_entropy']),.69)

    def test_precision_features_separate_range_precision_and_partial_storage(self):
        r=rows()[1];b=deepcopy(r);b['precision'].update(input='bf16',weight='bf16',output='bf16',partial='bf16')
        a=m.feature_vector(r)[-len(m.EXTRA):];bb=m.feature_vector(b)[-len(m.EXTRA):]
        self.assertEqual(a[:2],[10,15]);self.assertEqual(bb[:2],[7,127]);self.assertNotEqual(a,bb)
        changed={**r,'latency_us':1e9};self.assertEqual(m.feature_vector(r),m.feature_vector(changed))

    def test_precision_and_failure_gate_no_label_leak(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':5,'seeds':[0]}):model=m.fit(rr[:4])
        ev=deepcopy(rr[4:]);a=m.predict(model,ev)
        self.assertEqual(a,m.predict(model,list(reversed(ev))))
        ev[0]['latency_us']=1e30;self.assertEqual(a,m.predict(model,ev))
        ev[1]['validity']['status']='failed';pred=m.predict(model,ev)
        self.assertNotIn(ev[1]['candidate_id'],pred['scores']);self.assertTrue(pred['requires_validation'])
        ev[0]['validity']['status']='unknown';self.assertIsNone(m.predict(model,ev)['chosen'])
        with self.assertRaises(ValueError):m.fit(ev)
        wrong=deepcopy(rr[4:]);wrong[0]['precision']['input']='bf16'
        with self.assertRaisesRegex(ValueError,'unsupported precision'):m.predict(model,wrong)
        with self.assertRaises(ValueError):m.evaluate(model,[{**r,'split':'evaluation'} for r in rr[:2]])

    def test_official_optimizer_groups_and_bounded_cpu_run(self):
        net=m._network(40,8);opt=m.make_optimizer(net,'dion3')
        groups={g['algorithm']:[id(p) for p in g['params']] for g in opt.param_groups}
        self.assertEqual(groups['nordion2'],[id(net.hidden.weight)])
        self.assertIn(id(net.readout.weight),groups['adamw'])
        with patch.dict(m.SETTINGS,{'epochs':3,'seeds':[0]}):model=m.fit(rows(),optimizer='dion3')
        self.assertEqual(model['dion_revision'],m.DION_REVISION)
        self.assertFalse(model['compiler_admission'])

    def test_validity_cannot_be_reused_after_input_or_shape_change(self):
        for key,value in [('input_sha256','other-input'),('shape',[4,64,512,128]),
                          ('implementation_id','other-kernel'),('execution_context','other-machine')]:
            row=deepcopy(rows()[0]);row[key]=value
            with self.assertRaisesRegex(ValueError,'evidence does not match'):m.validity_status(row)

    def test_portable_inference_matches_fp32_network(self):
        rr=rows()
        with patch.dict(m.SETTINGS,{'epochs':5,'seeds':[0]}):fitted=m.fit(rr[:4])
        surface=rr[4:];net=m._network(len(fitted['features']),8)
        net.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in fitted['state'].items()})
        xx=(np.array([m.feature_vector(r) for r in surface])-fitted['means'])/fitted['scales']
        expected=net(torch.tensor(xx,dtype=torch.float32)).detach().numpy()
        actual=m.predict(fitted,surface)['scores']
        np.testing.assert_allclose([actual[r['candidate_id']] for r in surface],expected,atol=2e-6,rtol=1e-6)

if __name__=='__main__':unittest.main()
