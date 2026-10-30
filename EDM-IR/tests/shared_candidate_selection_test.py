import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
import numpy as np
from edmir.compiler import shared_candidate_selection as module
from edmir.compiler.shared_candidate_selection import (SharedCandidateSelector,VARIANTS,FEATURE_NAMES,
    objective,forward,parameter_count,training_tensors,static_features,canonical_choice)
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from predict_traffic_schedule import static_candidates


class SharedCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.models={m:SharedCandidateSelector.fit(cls.data,method=m) for m in VARIANTS}

    def test_full_objective_gradients_against_central_differences(self):
        rng=np.random.default_rng(14)
        x=rng.normal(size=(3,4,5));base=rng.normal(size=(3,4))*.1;target=rng.normal(size=(3,4))*.2
        mask=np.ones((3,4),bool);mask[0,3]=False;weights=mask/mask.sum(1,keepdims=True)
        for width in (0,3):
            for cost in (0.,1.):
                t=rng.normal(size=parameter_count(5,width))*.1
                _,g=objective(t,x,base,target,mask,weights,width,cost)
                numerical=[]
                for i in range(len(t)):
                    a,b=t.copy(),t.copy();a[i]+=1e-6;b[i]-=1e-6
                    numerical.append((objective(a,x,base,target,mask,weights,width,cost)[0]-objective(b,x,base,target,mask,weights,width,cost)[0])/2e-6)
                np.testing.assert_allclose(g,numerical,rtol=2e-5,atol=2e-8)

    def test_expected_regret_matches_explicit_enumeration_and_gradient(self):
        # Known latency ratios 1, 2, 4; identity features make scores independent.
        x=np.eye(3)[None,:,:];base=np.zeros((1,3));target=np.log([[1.,2.,4.]])
        mask=np.ones((1,3),bool);w=np.ones((1,3))/3;t=np.array([.1,-.1,.05,0.])
        f0,g0=objective(t,x,base,target,mask,w,0,0.)
        f1,g1=objective(t,x,base,target,mask,w,0,1.)
        z=-t[:3]/module.TEMPERATURE;p=np.exp(z-z.max());p/=p.sum()
        costs=np.array([0.,1.,3.]);expected=sum(p[i]*costs[i] for i in range(3))
        self.assertAlmostEqual(f1-f0,expected,places=12)
        np.testing.assert_allclose((g1-g0)[:3],-p*(costs-expected)/module.TEMPERATURE,atol=1e-12)
        self.assertLess((g1-g0)[2],0.)  # Gradient descent raises the slow candidate's cost.

    def test_padding_never_creates_a_label_or_reward(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(2,3,4));b=np.zeros((2,3));y=np.log([[1.,2.,3.],[2.,3.,4.]])
        mask=np.ones((2,3),bool);w=np.ones((2,3))/3;t=rng.normal(size=5)*.1
        f,g=objective(t,x,b,y,mask,w,0,1.)
        xx=np.pad(x,((0,0),(0,1),(0,0)));bb=np.pad(b,((0,0),(0,1)))
        yy=np.pad(y,((0,0),(0,1)),constant_values=1e6);mm=np.pad(mask,((0,0),(0,1)));ww=np.pad(w,((0,0),(0,1)))
        ff,gg=objective(t,xx,bb,yy,mm,ww,0,1.)
        self.assertAlmostEqual(f,ff,places=12);np.testing.assert_allclose(g,gg,atol=1e-12)

    def test_shared_scorer_permutation_equivariance(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(6,len(FEATURE_NAMES)));order=[4,1,3,0,5,2]
        for model in self.models.values():
            theta=np.asarray(model.parameters);width=VARIANTS[model.method][0]
            np.testing.assert_allclose(forward(theta,x,width)[order],forward(theta,x[order],width),atol=1e-12)
        # Verify the nonlinear scorer can represent cross-coordinate dependence.
        d=2;width=1;t=np.array([0.,0.,0.,1.,1.,0.,1.])
        xx=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        y=forward(t,xx,width)
        self.assertGreater(abs(y[3]-y[2]-y[1]+y[0]),.1)

    def test_every_surface_reordering_and_candidate_subset(self):
        rng=np.random.default_rng(5)
        for model in self.models.values():
            for surface in self.data.validate().values():
                r=surface['host'];all_candidates=static_candidates(r.rows_per_peer,r.k,r.n,self.data.environment_id)
                candidates={n:all_candidates[n] for n in surface}
                expected=model.select(candidates,baseline=self.data.baseline)
                for _ in range(3):
                    names=list(candidates);rng.shuffle(names)
                    self.assertEqual(expected,model.select({n:candidates[n] for n in names},baseline=self.data.baseline))
                keep=['host',next(n for n in expected['predicted_ms'] if n!='host')]
                subset=model.select({n:candidates[n] for n in keep},baseline=self.data.baseline)
                for n in keep:self.assertAlmostEqual(subset['predicted_ms'][n],expected['predicted_ms'][n],places=12)
                self.assertAlmostEqual(sum(expected['policy_weights'].values()),1.,places=12)
                self.assertFalse(expected['eligible_for_compiler_admission'])

    def test_exact_ties_have_documented_order(self):
        for order in itertools.permutations(['host','direct','splitk2']):
            self.assertEqual(canonical_choice({n:1. for n in order}),'host')
        self.assertEqual(canonical_choice({'host':1.000000000001,'splitk2':1.}),'splitk2')
        self.assertEqual(canonical_choice({'splitk4':1.,'splitk2':1.}),'splitk2')

    def test_training_mask_scaler_and_prior_have_no_heldout_rows(self):
        group=self.data.rows[0].group_id
        train=replace(self.data,rows=tuple(r for r in self.data.rows if r.group_id!=group))
        original=module.AdditiveLatencySelector.fit;seen=[]
        def recording(data,**kwargs):
            seen.append({r.shape for r in data.rows});return original(data,**kwargs)
        with patch.object(module.AdditiveLatencySelector,'fit',side_effect=recording):
            model=SharedCandidateSelector.fit(train,method='shared_linear_cost',use_cache=False)
        held={r.shape for r in self.data.rows if r.group_id==group}
        self.assertTrue(seen);self.assertTrue(all(not held&s for s in seen))
        surfaces=[{n:s[n] for n in sorted(s)} for _,s in sorted(train.validate().items())]
        x,base,y,mask,w,means,scales,names=training_tensors(surfaces,model.prior)
        self.assertEqual(tuple(map(float,means)),model.means)
        self.assertEqual(tuple(map(float,scales)),model.scales)
        self.assertEqual(int(mask.sum()),sum(len(ns) for ns in names))
        for i,ns in enumerate(names):
            self.assertEqual(set(ns),set(surfaces[i]))
            self.assertAlmostEqual(w[i].sum(),1.)
            for j,n in enumerate(ns):self.assertAlmostEqual(y[i,j],math.log(surfaces[i][n].measured_ms))

    def test_affine_latency_control_matches_augmented_lstsq(self):
        m=self.models['shared_linear_latency']
        surfaces=[{n:s[n] for n in sorted(s)} for _,s in sorted(self.data.validate().items())]
        x,b,y,mask,w,*_=training_tensors(surfaces,m.prior);d=x.shape[-1]
        a=np.concatenate((x,np.ones((*x.shape[:2],1))),axis=-1).reshape(-1,d+1)
        weight=np.sqrt(w.ravel()/len(surfaces))
        aa=np.vstack((a*weight[:,None],math.sqrt(module.RIDGE)*np.eye(d+1)))
        yy=np.r_[(y-b).ravel()*weight,np.zeros(d+1)]
        alternative=np.linalg.lstsq(aa,yy,rcond=None)[0]
        np.testing.assert_allclose(a@alternative,a@np.asarray(m.parameters),rtol=1e-9,atol=1e-9)

    def test_cache_order_and_changed_labels(self):
        original=self.models['shared_linear_cost']
        repeat=SharedCandidateSelector.fit(replace(self.data,rows=tuple(reversed(self.data.rows))),method=original.method,use_cache=False)
        self.assertEqual(original.to_dict(),repeat.to_dict())
        misses=module._fit_cached.cache_info().misses
        changed=replace(self.data,rows=(replace(self.data.rows[0],measured_ms=self.data.rows[0].measured_ms*1.01),*self.data.rows[1:]))
        altered=SharedCandidateSelector.fit(changed,method=original.method)
        self.assertEqual(module._fit_cached.cache_info().misses,misses+1)
        self.assertNotEqual(original.parameters,altered.parameters)
        original.training_audit['temperature']=100
        self.assertEqual(SharedCandidateSelector.fit(self.data,method=original.method).training_audit['temperature'],module.TEMPERATURE)
        original.training_audit['temperature']=module.TEMPERATURE
        self.assertEqual(module._fit_cached.cache_info().maxsize,128)

    def test_saved_models_replay_without_training(self):
        for model in self.models.values():
            obj=json.loads(json.dumps(model.to_dict(),allow_nan=False))
            restored=SharedCandidateSelector.from_dict(obj)
            c=static_candidates(64,4096,2048,self.data.environment_id)
            with patch.object(SharedCandidateSelector,'fit',side_effect=AssertionError('inference fit')):
                self.assertEqual(model.select(c,baseline=self.data.baseline),restored.select(c,baseline=self.data.baseline))
            for key,value in [('compiler_admission',True),('feature_names',[]),('parameters',[float('nan')]*len(model.parameters))]:
                bad=copy.deepcopy(obj);bad[key]=value
                with self.assertRaises(ValueError):SharedCandidateSelector.from_dict(bad)

    def test_domain_baseline_grouped_support_and_labels_rejected(self):
        model=self.models['shared_mlp_cost'];c=static_candidates(64,4096,512,self.data.environment_id)
        with self.assertRaises(ValueError):model.select(c,baseline=PER_SOURCE_CUBLAS)
        with self.assertRaises(TypeError):model.select(next(iter(self.data.validate().values())),baseline=self.data.baseline)
        with self.assertRaises(TypeError):static_features(self.data.rows[0],1.)
        with self.assertRaises(ValueError):model.select(static_candidates(192,3072,1536,self.data.environment_id),baseline=self.data.baseline)
        mixed=dict(c);mixed['host']=replace(c['host'],environment_id='a100-pcie')
        with self.assertRaises(ValueError):model.select(mixed,baseline=self.data.baseline)
        c=static_candidates(512,4096,512,self.data.environment_id)
        out=model.select(c,baseline=self.data.baseline)
        self.assertFalse(any('groupedn' in n for n in out['predicted_ms']))

    def test_timeout_and_budget_are_enforced(self):
        for bad in (0,-1,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError):SharedCandidateSelector.fit(self.data,timeout_seconds=bad)
        start=time.monotonic()
        with self.assertRaises(TimeoutError):SharedCandidateSelector.fit(self.data,timeout_seconds=1e-12,use_cache=False)
        self.assertLess(time.monotonic()-start,1.)
        for m in self.models.values():
            for run in m.training_audit['starts']:
                self.assertLessEqual(run['iterations'],module.MAX_ITER)
                self.assertLessEqual(run['evaluations'],module.MAX_EVAL+25)
                self.assertLessEqual(run['final_loss'],run['initial_loss']+1e-10)
            best=min(range(len(m.training_audit['starts'])),key=lambda i:(m.training_audit['starts'][i]['final_loss'],m.training_audit['starts'][i]['seed']))
            self.assertEqual(best,m.training_audit['selected_start'])


if __name__=='__main__':unittest.main()
