import copy
import json
from pathlib import Path
import unittest
import numpy as np
import audit
from edmir.compiler.decision_loss_selection import objective
from edmir.compiler.measurement_dataset import sha256

HERE=Path(__file__).resolve().parent


class AuditTest(unittest.TestCase):
    def test_explicit_pair_objective_and_gradients(self):
        rng=np.random.default_rng(28)
        x=rng.normal(size=(4,6,5));y=rng.uniform(.2,2,(4,6));w=np.zeros((4,6));mask=np.zeros_like(w,bool)
        for i,count in enumerate((3,4,5,6)):
            mask[i,:count]=True;w[i,0]=.5;w[i,1:count]=.5/(count-1)
        system=audit.explicit_system(x,y,w,mask)
        norm=np.min(np.where(mask,y,np.inf),axis=1)
        for _ in range(3):
            t=rng.normal(size=5)
            f,g=audit.explicit_objective(t,system)
            ff,gg=objective(t,x,np.zeros_like(y),y,mask,w,norm,1,1,.001)
            self.assertAlmostEqual(f,ff,places=10);np.testing.assert_allclose(g,gg,atol=1e-10)
            numeric=[]
            for j in range(5):
                a,b=t.copy(),t.copy();a[j]+=1e-6;b[j]-=1e-6
                numeric.append((audit.explicit_objective(a,system)[0]-audit.explicit_objective(b,system)[0])/2e-6)
            np.testing.assert_allclose(g,numeric,rtol=1e-6,atol=1e-7)

    def test_zero_information_bound_in_polisher(self):
        # Ridge-only columns must stay at the actual 1e-10 constraint, not
        # the unconstrained zero solution. This caused the initial audit stop.
        system=(np.diag([1.,.03]),np.array([1.,0.]),np.empty((0,2)),np.array([]))
        t,check=audit.polish(np.array([1.,1e-7]),system)
        self.assertTrue(check['valid']);self.assertEqual(t[1],audit.LOWER)
        self.assertGreaterEqual(min(t),audit.LOWER)

    def test_all_outer_errors_have_independent_solution(self):
        s=json.loads((HERE/'verified/summary.json').read_text())
        d=json.loads((HERE/'verified/outer_errors.json').read_text())
        self.assertEqual(s['outer_error_count'],14);self.assertEqual(len(d['independent_checks']),14)
        self.assertTrue(s['all_alternate_choices_equal'])
        self.assertLess(s['max_alternate_prediction_difference_us'],.0002)
        self.assertEqual(sum(not r['observed_best_eligible'] for r in d['inventory']),2)
        for r in d['independent_checks']:
            self.assertLess(r['alternative']['check']['projected_gradient'],1e-9)
            self.assertLess(r['alternative']['check']['global_objective_gap_upper_bound'],1e-16)

    def test_counterfactual_full_eligible_sets_and_coverage(self):
        ds=json.loads((HERE/'verified/deletions.json').read_text())
        p=json.loads((HERE/'verified/perturbations.json').read_text())
        self.assertEqual(len(ds),4);self.assertTrue(all(len(d['records'])==8 for d in ds))
        self.assertEqual(len(p['records']),34);self.assertEqual(len(p['traces']),15)
        for record in [r for d in ds for r in d['records']]+p['records']:
            self.assertEqual(len(record['retuned']['predicted_ms']),6)
            self.assertEqual(set(record['retuned']['predicted_ms']),set(record['frozen_geometry']['predicted_ms']))
            self.assertTrue(record['frozen_fit_check']['valid'])
        self.assertEqual(sum(r['retuned']['choice']=='host' for r in p['records']),31)
        self.assertEqual(sum(r['frozen_geometry']['choice']=='host' for r in p['records']),34)

    def test_identifiability_and_fused_feature_collisions(self):
        i=json.loads((HERE/'verified/identifiability.json').read_text());s=json.loads((HERE/'verified/structure.json').read_text())
        self.assertEqual(i['depth_05_equals_15_times_depth_1_max_abs'],0.)
        self.assertTrue(i['normalized_columns_identical'])
        a,b=i['alternatives_posthoc']
        self.assertAlmostEqual(a['fit']['check']['objective'],b['fit']['check']['objective'],places=13)
        self.assertGreater(abs(a['prediction']['predicted_ms']['host']-b['prediction']['predicted_ms']['host']),.019)
        self.assertEqual(len(s['feature_collisions']),4)
        for collision in s['feature_collisions']:
            self.assertEqual(len({r['best_fused'] for r in collision['shapes']}),2)
        self.assertEqual(len(s['full_training_only_errors']),6)
        for r in s['q_pair_affine_checks']:
            self.assertLess(r['max_feature_affine_residual'],1e-14)
            self.assertEqual(r['depth_tail_output_tiles_difference'],0.)
        self.assertAlmostEqual(s['pair_weights'][0]['each_host_fused_fraction']/s['pair_weights'][0]['each_fused_fused_fraction'],5.)

    def test_previous_delivery_and_audit_sources_unchanged(self):
        old=json.loads((audit.OLD/'delivery.json').read_text())
        for p,digest in old['sha256'].items():self.assertEqual(sha256(audit.ROOT/p),digest)
        for name in ('summary.json','structure.json'):
            result=json.loads((HERE/'verified'/name).read_text())
            for p,digest in result['source_sha256'].items():self.assertEqual(sha256(audit.ROOT/p),digest)


if __name__=='__main__':unittest.main()
