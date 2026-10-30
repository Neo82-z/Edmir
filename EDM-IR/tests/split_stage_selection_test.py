from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'EDM-IR'))
from edmir.compiler import split_stage_selection as m


class SplitStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=json.loads((ROOT/'experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json').read_text())['p4']
        folds={tuple(s['shape']):s['fold'] for s in data['folds']}
        cls.train=[r for r in data['rows'] if folds[tuple(r['shape'])]!=0]
        cls.test=[{**r,'split':'evaluation'} for r in data['rows'] if r['shape']==[4,64,512,128]]
        cls.prior=json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/support-and-prior/prior-fold-0.json').read_text())['model']
        cls.model=m.fit(cls.train,cls.prior)

    def test_split_activation_is_not_q_minus_one(self):
        data={r['candidate_id']:r for r in self.test}
        for name in ('direct','splitk2','splitk4'):
            names,x=m.features(data[name],'activation',[],32)
            self.assertEqual(x[-1],0. if name=='direct' else 1.)
            self.assertEqual(names[-1],'split_serial_path_enabled')
        self.assertEqual(m.geometry(data['splitk2']).estimated_ctas_per_op,2)

    def test_depth_intervals_preserve_linear_work_with_equal_slopes(self):
        r=next(r for r in self.test if r['candidate_id']=='splitk2')
        knots=[1.,2.,4.,8.,16.,32.,64.,128.,256.]
        names,x=m.features(r,'depth',knots,32)
        self.assertAlmostEqual(sum(x[-8:]),(8-1)/128)
        self.assertNotIn('partition_K',names)
        with self.assertRaises(ValueError):m.features(r,'depth',[16.,32.],32)

    def test_linear_refit_matches_existing_prior(self):
        model=m.fit(self.train,self.prior,'linear')
        a=model['stage_prior']['fit']['coefficients'];b=self.prior['fused']['coefficients']
        for x,y in zip(a,b):self.assertAlmostEqual(x,y,places=10)

    def test_host_predictions_unchanged(self):
        base=m.curves.fit(self.train,self.prior,mode='host_tiles_fused_residual')
        a=m.predict(base,self.test);b=m.predict(self.model,self.test)
        self.assertEqual(a['predicted_log_us']['host'],b['predicted_log_us']['host'])

    def test_label_free_permutation_and_serialization(self):
        a=m.predict(self.model,self.test)
        rows=deepcopy(self.test)
        for r in rows:r['latency_us']=1e12
        self.assertEqual(a,m.predict(self.model,list(reversed(rows))))
        self.assertEqual(a,m.predict(json.loads(json.dumps(self.model)),self.test))
        self.assertEqual(self.model,m.fit(list(reversed(self.train)),self.prior))

    def test_precision_and_whole_shape_contracts(self):
        rows=deepcopy(self.test);rows[0]['precision']['input']='bf16'
        with self.assertRaises(ValueError):m.predict(self.model,rows)
        with self.assertRaisesRegex(ValueError,'leakage'):
            m.evaluate(self.model,[{**r,'split':'evaluation'} for r in self.train])

    def test_candidate_subset_does_not_change_remaining_scores(self):
        full=m.predict(self.model,self.test)['predicted_log_us']
        partial=m.predict(self.model,[r for r in self.test if r['candidate_id'] in ('host','direct','splitk2')])
        self.assertEqual(partial['predicted_log_us'],{n:full[n] for n in partial['predicted_log_us']})

    def test_execution_and_capacity_contracts(self):
        rows=deepcopy(self.test);rows[0]['execution_context']='another_gpu'
        with self.assertRaises(ValueError):m.predict(self.model,rows)
        rows=deepcopy(self.test);rows[0]['implementation_id']='new_kernel'
        with self.assertRaises(ValueError):m.predict(self.model,rows)
        for cap in (0,-1,float('nan'),float('inf')):
            bad=deepcopy(self.model);bad['stage_prior']['capacity']=cap
            with self.assertRaises(ValueError):m.predict(bad,self.test)

    def test_tampered_stage_coefficients_rejected(self):
        bad=deepcopy(self.model)
        bad['stage_prior']['fit']['coefficients']=list(bad['stage_prior']['fit']['coefficients'])
        bad['stage_prior']['fit']['coefficients'][0]=-1
        with self.assertRaises(ValueError):m.predict(bad,self.test)
        bad=deepcopy(self.model);bad['stage_prior']['fit']['names'][0]='invented'
        with self.assertRaises(ValueError):m.predict(bad,self.test)
        bad=deepcopy(self.model);bad['stage_prior']['fit']['kkt_max_violation']=1.
        with self.assertRaises(ValueError):m.predict(bad,self.test)


if __name__=='__main__':unittest.main()
