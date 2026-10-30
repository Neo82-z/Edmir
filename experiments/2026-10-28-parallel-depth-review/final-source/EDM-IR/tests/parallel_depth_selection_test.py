from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import parallel_depth_selection as m


class ParallelDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads((ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json').read_text())['p4']
        folds={tuple(s['shape']):s['fold'] for s in data['folds']}
        cls.train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=0]
        cls.test=[{**r,'split':'evaluation'} for r in data['rows'] if r['shape']==[4,64,512,128]]
        cls.all_rows=data['rows']
        cls.prior=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-0.json').read_text())['model']
        cls.models={o:m.fit(cls.train,cls.prior,'depth_parallel_group',objective=o)
                    for o in ('latency','relative','relative_pair')}
        cls.knots=cls.models['latency']['stage_prior']['knots']

    def features(self,r,v):
        n,x=m.features(r,v,self.knots,32);return dict(zip(n,x))

    def test_concurrency_interactions_can_break_old_gap_order(self):
        deltas=[]
        for n in (128,512):
            rr={r['candidate_id']:r for r in self.all_rows if r['shape']==[4,64,512,n]}
            a=self.features(rr['direct'],'depth_parallel');b=self.features(rr['splitk2'],'depth_parallel')
            deltas.append({k:b[k]-a[k] for k in a})
        difference=[deltas[1][k]-deltas[0][k] for k in deltas[0]]
        self.assertLess(min(difference),0.)
        self.assertGreater(max(difference),0.)

    def test_blend_weights_recover_original_depth(self):
        row=next(r for r in self.test if r['candidate_id']=='splitk2')
        n,x=m.stage.features(row,'depth',self.knots,32);base=dict(zip(n,x))
        for v in ('depth_blend','depth_chain'):
            xx=self.features(row,v)
            for n,y in base.items():
                if n.startswith('depth_'):self.assertAlmostEqual(xx['low_'+n]+xx['filled_'+n],y)

    def test_output_chains_are_distinct_from_partition_ctas(self):
        rows={r['candidate_id']:r for r in self.test}
        for q in (2,4):
            c=m.stage.geometry(rows[f'splitk{q}'])
            self.assertEqual(c.estimated_ctas_per_op,q)
            self.assertEqual(c.estimated_ctas_per_op/c.split_k_slices,1)
        a=self.features(rows['splitk2'],'depth_chain');b=self.features(rows['splitk4'],'depth_chain')
        self.assertEqual(a['low_depth_1_2'],b['low_depth_1_2'])
        a=self.features(rows['splitk2'],'depth_blend');b=self.features(rows['splitk4'],'depth_blend')
        self.assertNotEqual(a['low_depth_1_2'],b['low_depth_1_2'])

    def test_width_curves_follow_kernel_geometry(self):
        for name,prefix in [('splitk2','wide_'),('splitk2_groupedn2','narrow_')]:
            row=next(r for r in self.test if r['candidate_id']==name)
            x=self.features(row,'depth_width_chain')
            self.assertTrue(all(v==0 for n,v in x.items() if n.startswith(prefix)))
            self.assertTrue(all(v>=0 for v in x.values()))

    def test_handoff_vanishes_for_unsplit(self):
        row=next(r for r in self.test if r['candidate_id']=='direct')
        self.assertEqual(self.features(row,'depth_handoff')['short_serial_exposure'],0)

    def test_operand_footprints_break_tall_wide_pair_alias(self):
        differences={}
        for variant in ('depth_parallel_group','depth_parallel_group_footprint'):
            differences[variant]=[]
            for shape in ([4,64,4096,2048],[4,256,4096,512]):
                rows={r['candidate_id']:r for r in self.all_rows if r['shape']==shape}
                a=self.features(rows['splitk2'],variant);b=self.features(rows['splitk4'],variant)
                differences[variant].append({n:b[n]-a[n] for n in a})
            first,second=differences[variant]
            if variant=='depth_parallel_group':self.assertEqual(first,second)
            else:
                self.assertNotEqual(first['partition_unique_A_MiB'],second['partition_unique_A_MiB'])
                self.assertNotEqual(first['partition_unique_B_MiB'],second['partition_unique_B_MiB'])

    def test_all_objectives_ignore_query_labels(self):
        rows=deepcopy(self.test)
        for r in rows:r['latency_us']=1e10
        for fitted in self.models.values():self.assertEqual(m.predict(fitted,self.test),m.predict(fitted,rows))

    def test_pair_objective_order_and_serialization(self):
        fitted=self.models['relative_pair'];a=m.predict(fitted,self.test)
        self.assertEqual(a,m.predict(fitted,list(reversed(self.test))))
        self.assertEqual(a,m.predict(json.loads(json.dumps(fitted)),self.test))
        self.assertEqual(fitted,m.fit(list(reversed(self.train)),self.prior,'depth_parallel_group',objective='relative_pair'))
        partial=m.predict(fitted,[r for r in self.test if r['candidate_id'] in ('host','direct','splitk2')])
        self.assertEqual(partial['predicted_log_us'],{n:a['predicted_log_us'][n] for n in partial['predicted_log_us']})

    def test_host_calibration_unchanged(self):
        baseline=m.stage.fit(self.train,self.prior)
        value=m.predict(baseline,self.test)['predicted_log_us']['host']
        for fitted in self.models.values():self.assertEqual(value,m.predict(fitted,self.test)['predicted_log_us']['host'])

    def test_precision_identity_and_whole_shape_separation(self):
        fitted=self.models['relative_pair']
        rows=deepcopy(self.test);rows[0]['precision']['input']='bf16'
        with self.assertRaises(ValueError):m.predict(fitted,rows)
        rows=deepcopy(self.test);rows[0]['implementation_id']='unmeasured'
        with self.assertRaises(ValueError):m.predict(fitted,rows)
        with self.assertRaisesRegex(ValueError,'leakage'):
            m.evaluate(fitted,[{**r,'split':'evaluation'} for r in self.train])

    def test_positive_fit_and_bad_certificate(self):
        for fitted in self.models.values():
            fit=fitted['stage_prior']['fit']
            self.assertLess(fit['kkt_max_violation'],1e-7)
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in fit['coefficients']))
            bad=deepcopy(fitted);bad['stage_prior']['fit']['kkt_max_violation']=1
            with self.assertRaises(ValueError):m.predict(bad,self.test)

    def test_unseen_depth_and_unknown_objective_rejected(self):
        row=next(r for r in self.test if r['candidate_id']=='splitk2')
        with self.assertRaises(ValueError):m.features(row,'depth_parallel',[16,32],32)
        with self.assertRaises(ValueError):m.fit(self.train,self.prior,'depth_parallel',objective='guess')

    def test_new_variant_requires_explicit_selection(self):
        with self.assertRaises(TypeError):m.fit(self.train,self.prior)
        self.assertFalse(self.models['relative_pair']['compiler_admission'])


if __name__=='__main__':unittest.main()
