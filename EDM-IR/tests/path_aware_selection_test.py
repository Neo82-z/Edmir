from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from shape_evaluation_test import numerical_rows
from edmir.compiler import shape_decision_model as old
from edmir.compiler import path_aware_selection as new
from edmir.compiler.shared_candidate_selection import parameter_count


class BoundaryTests(unittest.TestCase):
    def test_objective_reproduces_original_without_interventions(self):
        rows=numerical_rows();x,y,mask,w,h,m,_,_,_,_=new.tensors(rows,'original',False)
        theta=np.random.default_rng(3).normal(0,.1,parameter_count(x.shape[-1],8))
        a,ga=old.objective(theta,x,y,mask,w,8,1.,1.)
        b,gb=new.objective(theta,x,y,mask,w,h,m,8,cross_path=False)
        self.assertAlmostEqual(a,b,places=10);np.testing.assert_allclose(ga,gb,atol=1e-11)

    def test_cross_path_gradient_and_mask(self):
        rng=np.random.default_rng(47);x=rng.normal(size=(2,3,4))
        y=np.array([[1.,2.,3.],[1.5,1.,9999.]])
        mask=np.array([[1,1,1],[1,1,0]],bool);h=np.array([[1,0,0],[1,0,0]],bool)
        w=np.array([[.5,.25,.25],[.5,.5,0.]]);mass=np.array([2.,.7])
        for width in (0,2):
            theta=rng.normal(0,.2,parameter_count(4,width))
            value,grad=new.objective(theta,x,y,mask,w,h,mass,width,cross_path=True)
            for i in range(len(theta)):
                d=np.zeros_like(theta);d[i]=1e-6
                a=new.objective(theta+d,x,y,mask,w,h,mass,width,cross_path=True)[0]
                b=new.objective(theta-d,x,y,mask,w,h,mass,width,cross_path=True)[0]
                self.assertAlmostEqual(grad[i],(a-b)/2e-6,places=5)
            perm=[2,0,1]
            vp,gp=new.objective(theta,x[:,perm],y[:,perm],mask[:,perm],w[:,perm],h[:,perm],mass,width,cross_path=True)
            self.assertAlmostEqual(value,vp,places=10);np.testing.assert_allclose(grad,gp,atol=1e-10)

    def test_path_interactions_use_metadata_not_label(self):
        rows=numerical_rows();r=rows[0];bad={**r,'latency_us':1e9}
        self.assertEqual(new.feature_vector(r,'path_geometry'),new.feature_vector(bad,'path_geometry'))
        a=new.feature_vector(rows[0],'path_geometry')[-6:];b=new.feature_vector(rows[-2],'path_geometry')[-6:]
        self.assertNotEqual(a,b)
        self.assertEqual(new.feature_vector(rows[1],'path_geometry')[-6:],[0.]*6)

    def test_frontier_gap_gradient_with_fixed_prior(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(2,3,4))
        y=np.array([[1.,2.,3.],[1.5,1.,2.]]);mask=np.ones((2,3),bool)
        h=np.array([[1,0,0],[1,0,0]],bool);w=np.array([[.5,.25,.25]]*2)
        mass=np.array([1.,2.]);theta=rng.normal(0,.2,parameter_count(4,2));offset=y*.9
        _,grad=new.objective(theta,x,y,mask,w,h,mass,2,cross_path='frontier',offset=offset)
        for i in range(len(theta)):
            d=np.zeros_like(theta);d[i]=1e-6
            a=new.objective(theta+d,x,y,mask,w,h,mass,2,cross_path='frontier',offset=offset)[0]
            b=new.objective(theta-d,x,y,mask,w,h,mass,2,cross_path='frontier',offset=offset)[0]
            self.assertAlmostEqual(grad[i],(a-b)/2e-6,places=5)

    def test_support_is_joint_geometry_not_axis_range(self):
        points=((4,32,512,128),(4,64,1024,128),(4,32,1024,128))
        self.assertTrue(new.in_training_hull(points,(4,32,512,128)))
        self.assertTrue(new.in_training_hull(points,(4,32,768,128)))
        self.assertFalse(new.in_training_hull(points,(4,64,512,128)))
        with self.assertRaises(ValueError):new.in_training_hull(points,(8,32,768,128))

    def test_prior_cannot_use_more_training_shapes_than_residual(self):
        rows=numerical_rows()
        metadata={'training_shapes':[r['shape'] for r in rows[::2]],'environment_id':'unit_test'}
        with patch('edmir.compiler.additive_latency_selection.AdditiveLatencySelector.from_dict',
                   return_value=SimpleNamespace(to_dict=lambda:metadata)):
            with self.assertRaises(ValueError):new.fit(rows[:4],prior=metadata)

    def test_train_scaler_and_initialization_weight_controls(self):
        rows=numerical_rows()
        a=new.tensors(rows,'path_geometry',False);b=new.tensors(rows,'path_geometry',True)
        np.testing.assert_array_equal(a[6],b[6]);np.testing.assert_array_equal(a[7],b[7])
        self.assertNotEqual(a[5].tolist(),b[5].tolist())

    def test_fit_order_and_no_heldout_leakage(self):
        rows=numerical_rows();a=new.fit(rows[:4],width=0);b=new.fit(list(reversed(rows[:4])),width=0)
        self.assertEqual(a['theta'],b['theta'])
        ev=[{**r,'split':'evaluation'} for r in rows[4:]]
        self.assertTrue(new.evaluate(a,ev)['decisions'][0]['order_invariant'])
        with self.assertRaises(ValueError):new.evaluate(a,[{**r,'split':'evaluation'} for r in rows[:2]])
        wrong=deepcopy(ev);wrong[0]['execution_context']='changed';
        with self.assertRaises(ValueError):new.predict(a,wrong)


if __name__=='__main__':unittest.main()
