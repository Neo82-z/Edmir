import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler import decision_loss_selection as mod
from edmir.compiler.measurement_dataset import load_identified_selector_dataset
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from edmir.compiler.relative_selection import ScheduleCandidate
from predict_traffic_schedule import static_candidates


class DecisionLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')
        cls.models={n:mod.DecisionLossSelector.fit(cls.data,method=n) for n in mod.VARIANTS}

    def test_gradients_mask_and_all_loss_settings(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(3,5,4));b=rng.normal(size=(3,5));y=rng.normal(size=(3,5))
        mask=np.ones((3,5),bool);mask[1,-1]=False;w=mask/mask.sum(1,keepdims=True)
        norm=np.array([.2,1.,3.]);t=rng.normal(size=4)
        for loss in mod.LOSS_GRID:
            f,g=mod.objective(t,x,b,y,mask,w,norm,*loss);numeric=[]
            for j in range(len(t)):
                a,c=t.copy(),t.copy();a[j]+=1e-6;c[j]-=1e-6
                numeric.append((mod.objective(a,x,b,y,mask,w,norm,*loss)[0]-mod.objective(c,x,b,y,mask,w,norm,*loss)[0])/2e-6)
            np.testing.assert_allclose(g,numeric,rtol=1e-6,atol=1e-7)
            xx=np.pad(x,((0,0),(0,2),(0,0)),constant_values=17)
            args=[np.pad(z,((0,0),(0,2)),constant_values=fill) for z,fill in ((b,-99),(y,99),(mask,False),(w,0))]
            ff,gg=mod.objective(t,xx,*args,norm,*loss)
            self.assertAlmostEqual(f,ff,places=10);np.testing.assert_allclose(g,gg,atol=1e-10)

    def test_pair_loss_matches_explicit_pairs(self):
        x=np.eye(4)[None,:,:];b=np.zeros((1,4));y=np.array([[1.,2.,4.,7.]])
        mask=np.ones((1,4),bool);w=np.array([[.5,1/6,1/6,1/6]]);norm=np.array([2.]);t=np.array([1.2,3.,5.,6.])
        f0,g0=mod.objective(t,x,b,y,mask,w,norm,0,0,.001)
        f1,g1=mod.objective(t,x,b,y,mask,w,norm,1,0,.001)
        e=(t-y[0])/2
        pairs=[(w[0,i]*w[0,j],(e[i]-e[j])**2) for i in range(4) for j in range(i+1,4)]
        self.assertAlmostEqual(f1-f0,sum(a*b for a,b in pairs)/sum(a for a,b in pairs),places=12)
        self.assertAlmostEqual(sum(g1-g0),0.,places=12)

    def test_margin_bounds_hard_choice_loss_including_ties(self):
        rng=np.random.default_rng(31)
        for space in ('time','log'):
            for _ in range(100):
                times=rng.uniform(.1,3.,6);score=rng.uniform(.1,3.,6)
                target=np.log(times) if space=='log' else times
                norm=np.array([1. if space=='log' else times.min()])
                if _%10==0:score[:]=1.
                x=np.eye(6)[None,:,:];b=np.zeros((1,6));mask=np.ones((1,6),bool);w=np.ones((1,6))/6
                f0,_=mod.objective(score,x,b,target[None,:],mask,w,norm,0,0,0)
                f1,_=mod.objective(score,x,b,target[None,:],mask,w,norm,0,1,0)
                regret=times[np.argmin(score)]/times.min()-1
                bound=(np.log1p(regret) if space=='log' else regret)**2
                self.assertGreaterEqual(f1-f0+1e-10,bound)

    def test_convexity_and_independent_solver(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(8,4,5));b=np.zeros((8,4));y=rng.uniform(.2,2.,(8,4))
        mask=np.ones((8,4),bool);w=np.ones((8,4))/4;norm=y.min(1)
        fun=lambda t:mod.objective(t,x,b,y,mask,w,norm,1,1,.01)
        a,c=rng.normal(size=(2,5));mid=.3*a+.7*c
        self.assertLessEqual(fun(mid)[0],.3*fun(a)[0]+.7*fun(c)[0]+1e-12)
        for bounds in (None,[(0,None)]*5):
            p=minimize(fun,np.ones(5),method='L-BFGS-B',jac=True,bounds=bounds,options={'ftol':1e-14,'gtol':1e-9})
            q=minimize(fun,np.ones(5)*2,method='SLSQP',jac=True,bounds=bounds,options={'ftol':1e-12,'maxiter':1000})
            self.assertTrue(p.success and q.success)
            self.assertAlmostEqual(p.fun,q.fun,places=8)

    def test_order_and_candidate_subset_stability(self):
        rng=np.random.default_rng(5)
        for model in self.models.values():
            for surface in self.data.validate().values():
                c={n:ScheduleCandidate.from_measurement(r,self.data.environment_id) for n,r in surface.items()}
                out=model.select(c,baseline=self.data.baseline);names=list(c);rng.shuffle(names)
                self.assertEqual(out,model.select({n:c[n] for n in names},baseline=self.data.baseline))
                keep=['host',next(n for n in out['predicted_ms'] if n!='host')]
                sub=model.select({n:c[n] for n in keep},baseline=self.data.baseline)
                for n in keep:self.assertAlmostEqual(out['predicted_ms'][n],sub['predicted_ms'][n],places=12)

    def test_label_isolation_and_inner_provenance(self):
        held=self.data.rows[0].group_id
        train=replace(self.data,rows=tuple(r for r in self.data.rows if r.group_id!=held))
        model=mod.DecisionLossSelector.fit(train,method='traffic_nested')
        outside={r.shape for r in self.data.rows if r.group_id==held};known={r.shape for r in train.rows}
        self.assertFalse(set(model.prior.training_shapes)&outside)
        self.assertEqual(set(model.prior.training_shapes),known)
        for entry in model.cv:
            for fold in entry['folds']:
                inner_held={r.shape for r in train.rows if r.group_id==fold['heldout']}
                self.assertEqual(set(fold['training_shapes']),known-inner_held)
                self.assertFalse(set(fold['training_shapes'])&outside)
        x,b,y,mask,w,norm,means,scales,names=mod.tensors(train,model.prior,'time')
        surfaces=[s for _,s in sorted(train.validate().items())]
        for i,ns in enumerate(names):
            self.assertEqual(set(ns),set(surfaces[i]))
            for j,n in enumerate(ns):self.assertEqual(y[i,j],surfaces[i][n].measured_ms)
        self.assertEqual(tuple(scales),model.scales)

    def test_serialization_and_invalid_states(self):
        for m in self.models.values():
            obj=json.loads(json.dumps(m.to_dict(),allow_nan=False));restored=mod.DecisionLossSelector.from_dict(obj)
            c=static_candidates(64,4096,2048,self.data.environment_id)
            with patch.object(mod.DecisionLossSelector,'fit',side_effect=AssertionError('inference training')):
                self.assertEqual(m.select(c,baseline=self.data.baseline),restored.select(c,baseline=self.data.baseline))
            for key,val in [('compiler_admission',True),('parameters',[float('nan')]*len(m.parameters)),('scales',[0.]*len(m.scales)),('training_shapes',[])]:
                bad=copy.deepcopy(obj);bad[key]=val
                with self.assertRaises(ValueError):mod.DecisionLossSelector.from_dict(bad)

    def test_guards_baseline_grouped_and_extrapolation(self):
        for m in self.models.values():
            c=static_candidates(512,4096,512,self.data.environment_id)
            out=m.select(c,baseline=self.data.baseline)
            self.assertFalse(any('groupedn' in n for n in out['predicted_ms']))
            with self.assertRaises(ValueError):m.select(c,baseline=PER_SOURCE_CUBLAS)
            with self.assertRaises(TypeError):m.select(next(iter(self.data.validate().values())),baseline=self.data.baseline)
            with self.assertRaises(ValueError):m.select(static_candidates(192,3072,1536,self.data.environment_id),baseline=self.data.baseline)
            mixed=dict(c);mixed['host']=replace(c['host'],environment_id='a100-pcie')
            with self.assertRaises(ValueError):m.select(mixed,baseline=self.data.baseline)
        train=replace(self.data,rows=tuple(r for r in self.data.rows if r.rows_per_peer<512))
        m=mod.DecisionLossSelector.fit(train,method='traffic_gap')
        with self.assertRaises(ValueError):m.select(c,baseline=self.data.baseline)
        self.assertIn('choice',m.select(c,baseline=self.data.baseline,allow_extrapolation=True))

    def test_cache_isolation_and_determinism(self):
        m=self.models['traffic_gap'];again=mod.DecisionLossSelector.fit(replace(self.data,rows=tuple(reversed(self.data.rows))),method=m.method,use_cache=False)
        self.assertEqual(m.to_dict(),again.to_dict())
        misses=mod._cached.cache_info().misses
        changed=replace(self.data,rows=(replace(self.data.rows[0],measured_ms=self.data.rows[0].measured_ms*1.01),*self.data.rows[1:]))
        other=mod.DecisionLossSelector.fit(changed,method=m.method)
        self.assertEqual(mod._cached.cache_info().misses,misses+1);self.assertNotEqual(m.parameters,other.parameters)
        clean=m.training_audit['final_loss'];m.training_audit['final_loss']=-1
        self.assertEqual(mod.DecisionLossSelector.fit(self.data,method=m.method).training_audit['final_loss'],clean)
        m.training_audit['final_loss']=clean
        self.assertEqual(mod._cached.cache_info().maxsize,128);self.assertEqual(mod._prior.cache_info().maxsize,256)

    def test_timeout_and_convergence(self):
        for bad in (0,-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):mod.DecisionLossSelector.fit(self.data,timeout_seconds=bad)
        start=time.monotonic()
        with self.assertRaises(TimeoutError):mod.DecisionLossSelector.fit(self.data,timeout_seconds=1e-12,use_cache=False)
        self.assertLess(time.monotonic()-start,1.)
        for m in self.models.values():
            a=m.training_audit
            self.assertTrue(a['success']);self.assertLess(a['projected_gradient'],1e-5)
            self.assertLessEqual(a['final_loss'],a['initial_loss']+1e-10)
            self.assertLessEqual(a['iterations'],mod.MAX_ITER)


if __name__=='__main__':unittest.main()
