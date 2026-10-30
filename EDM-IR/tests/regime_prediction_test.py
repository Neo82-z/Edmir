"""Independent numerical checks and evidence boundaries of the max hypothesis."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"EDM-IR"),str(ROOT/"EDM-IR/tools")]
from edmir.compiler.mechanism_prediction import HOSTS,RIDGE,MechanismModel,select_prediction
from edmir.compiler.regime_prediction import RegimeModel,max_branch_fit,regime_basis
from evaluate_regime_prediction import grouped_folds


class SolverTests(unittest.TestCase):
    def test_recover_known_two_regimes_with_common_floor(self):
        x=[[1.,l,w] for l in (1.,2.,3.) for w in (.5,1.,2.,4.,8.)]
        y=[7+max(5*l,2*w) for _,l,w in x]
        f=max_branch_fit(x,y,[1.]*len(y))
        for a,b in zip(f["coefficients_us"],(7,5,2)):
            self.assertAlmostEqual(a,b,places=4)
        self.assertEqual(set(f["training_branch_counts"]),{"latency","throughput"})
        self.assertLess(f["kkt_relative_violation"],1e-9)
        self.assertEqual(f["interval_count"],len({w/l for _,l,w in x})+1)
        self.assertAlmostEqual(f["objective_us2"],min(c["objective_us2"] for c in f["interval_certificates"]))

    def test_global_objective_against_independent_dense_ratio_profile(self):
        # For a fixed raw ratio t=a/b, solve a 2x2 quadratic in common,b.
        # This calculation does not use the cone reparameterization or solver.
        x=[[1.,l,w] for l in (1.,2.,4.) for w in (.5,1.,3.,9.)]
        y=[4+max(3.1*l,1.7*w)+(i%3-.8)*.5 for i,(_,l,w) in enumerate(x)]
        weights=[1.+(i%2) for i in range(len(x))]
        f=max_branch_fit(x,y,weights)
        sl2=sum(r[1]**2 for r in x)/len(x)
        sw2=sum(r[2]**2 for r in x)/len(x)
        best=math.inf
        for j in range(1001):
            t=j/100
            v=[max(t*l,w) for _,l,w in x]
            aa=sum(weights)+RIDGE
            ab=sum(w*r for w,r in zip(weights,v))
            bb=sum(w*r*r for w,r in zip(weights,v))+RIDGE*(t*t*sl2+sw2)
            ra=sum(w*r for w,r in zip(weights,y))
            rb=sum(w*r*s for w,r,s in zip(weights,v,y))
            det=aa*bb-ab*ab
            candidates=[(max(ra/aa,0),0),(0,max(rb/bb,0)),((ra*bb-rb*ab)/det,(rb*aa-ra*ab)/det)]
            for c,b in candidates:
                if min(c,b)<0:continue
                obj=sum(w*(c+b*v0-y0)**2 for w,v0,y0 in zip(weights,v,y))
                obj+=RIDGE*(c*c+b*b*(t*t*sl2+sw2))
                best=min(best,obj)
        self.assertLessEqual(f["objective_us2"],best+1e-7)
        self.assertLess(best-f["objective_us2"],.02)

    def test_units_and_permutation_preserve_prediction_and_ridge(self):
        x=[[1.,l,w] for l in (1.,2.,3.) for w in (1.,2.,4.,8.)]
        y=[11+max(5*l,2*w) for _,l,w in x]
        a=max_branch_fit(x,y,[1.]*len(x))
        b=max_branch_fit([[r[0],r[1]*1024,r[2]/1024] for r in x[::-1]],y[::-1],[1.]*len(x))
        self.assertAlmostEqual(a["objective_us2"],b["objective_us2"],places=8)
        self.assertAlmostEqual(a["coefficients_us"][0],b["coefficients_us"][0],places=7)
        self.assertAlmostEqual(a["coefficients_us"][1],b["coefficients_us"][1]*1024,places=7)
        self.assertAlmostEqual(a["coefficients_us"][2],b["coefficients_us"][2]/1024,places=7)

    def test_all_latency_and_all_throughput_boundaries(self):
        x=[[l,w] for l in (1.,2.,3.) for w in (1.,2.,4.)]
        for index in (0,1):
            f=max_branch_fit(x,[5*r[index] for r in x],[1.]*len(x))
            self.assertAlmostEqual(f["coefficients_us"][index],5,places=5)
            self.assertLess(f["coefficients_us"][1-index],1e-6)
            self.assertEqual(sum(f["training_branch_counts"].values()),len(x))

    def test_tied_ratio_is_one_boundary_and_prediction_not_doubled(self):
        x=[[1.,1.,2.],[1.,2.,4.],[1.,3.,6.]]
        f=max_branch_fit(x,[9.,17.,25.],[1.,1.,1.])
        self.assertEqual(f["interval_count"],2)
        self.assertLess(f["unregularized_rank"],f["feature_count"])
        for r,y in zip(x,(9.,17.,25.)):
            c=f["coefficients_us"]
            self.assertAlmostEqual(c[0]+max(c[1]*r[1],c[2]*r[2]),y,places=3)

    def test_reject_invalid_or_unbounded_solver_inputs(self):
        for raw,y,w in (([],[],[]),([[1,0]],[1],[1]),([[1,2]],[0],[1]),
                        ([[1,math.nan]],[1],[1]),([[1,2]]*513,[1]*513,[1]*513),
                        ([[1]*9],[1],[1]),([[1,2],[1,2,3]],[1,2],[1,1])):
            with self.assertRaises(ValueError):max_branch_fit(raw,y,w)

    def test_deadline_fails_before_unbounded_traversal(self):
        with patch("edmir.compiler.regime_prediction.time.monotonic",side_effect=(0.,2.)):
            with self.assertRaises(TimeoutError):max_branch_fit([[1.,2.]],[3.],[1.],timeout_seconds=1.)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs=json.loads((ROOT/"experiments/2026-10-25-fine-grained-prediction/final/configurations.json").read_text())
        cls.primary=[r for r in cls.configs if r["primary_scope"]]
        cls.stable=[r for r in cls.primary if r["eligible_for_calibration"]]
        cls.model=RegimeModel.fit(cls.stable)

    def test_ratio_cancels_K_and_retains_N_q_grouping(self):
        for m,n,q,g in ((64,512,2,1),(192,1536,4,2),(512,2048,8,1)):
            for k in (1024,4096,8192):
                _,v=regime_basis((4,m,k,n),f"splitk{q}"+("_groupedn2" if g==2 else ""))
                self.assertEqual(v[-1]/v[-2],m*q*((n+128*g-1)//(128*g))/256)

    def test_controlled_sum_and_max_share_eight_inputs_and_hosts(self):
        models=[RegimeModel.fit(self.stable,variant=v) for v in ("regime_sum","regime_max")]
        base=MechanismModel.fit(self.stable,variant="cutlass_depth")
        for model in models:
            self.assertEqual(model.fused["feature_count"],8)
            self.assertEqual(model.fused["names"],models[0].fused["names"])
            for r in self.primary:
                if r["family"] in HOSTS:
                    self.assertEqual(model.predict(r["shape"],r["family"],context_id=model.context_id),
                                     base.predict(r["shape"],r["family"],context_id=base.context_id))

    def test_heldout_labels_cannot_change_that_fold_fit(self):
        target=[4,192,3072,512]
        before,a=grouped_folds(self.primary,"regime_max")
        changed=copy.deepcopy(self.primary)
        for r in changed:
            if r["shape"]==target:r["observed_us"]*=11
        after,b=grouped_folds(changed,"regime_max")
        fa=next(f for f in a if list(f["held_out"])==target)
        fb=next(f for f in b if list(f["held_out"])==target)
        self.assertEqual(fa["model"],fb["model"])
        self.assertEqual([r["predicted_us"] for r in before if r["shape"]==target],
                         [r["predicted_us"] for r in after if r["shape"]==target])

    def test_leave_M_has_no_training_M_or_invented_hosts(self):
        _,folds=grouped_folds(self.primary,"regime_max",axis="M")
        for f in folds:
            self.assertTrue(all(s[1]!=f["held_out"] for s in f["model"]["training_shapes"]))
            self.assertTrue(all(r["family"] not in HOSTS for r in f["predictions"]))

    def test_order_invariance_and_exact_contribution_total(self):
        a=self.model
        b=RegimeModel.fit(self.stable[::-1])
        self.assertEqual(a.to_dict(),b.to_dict())
        predictions=[]
        for r in self.primary:
            p=a.predict(r["shape"],r["family"],context_id=a.context_id)
            self.assertAlmostEqual(p["predicted_us"],math.fsum(p["contributions_us"].values()),places=8)
            self.assertFalse(p["compiler_admission"])
            if r["family"] not in HOSTS:self.assertFalse(p["regime"]["physical_bottleneck_identified"])
            predictions.append({**p,"candidate_id":r["config_id"]})
        self.assertEqual(select_prediction(predictions),select_prediction(predictions[::-1]))

    def test_mismatched_context_policy_quality_are_rejected(self):
        with self.assertRaises(ValueError):self.model.predict((4,64,4096,512),"splitk2",context_id="different")
        with self.assertRaises(ValueError):self.model.predict((4,64,4096,512),"splitk2",context_id=self.model.context_id,dependency_policy="new")
        with self.assertRaises(ValueError):RegimeModel.fit(self.primary)


if __name__=="__main__":unittest.main()
