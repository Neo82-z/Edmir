import copy
import importlib.util
import json
from pathlib import Path
import unittest
import analyze_traces

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p1p2_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

class PlanTests(unittest.TestCase):
    def test_trace_overlap_uses_unions_not_duration_sums(self):
        # Nested communication kernels must not double-count simultaneous time.
        comm=[(0,10),(2,8)];compute=[(5,12),(6,7)]
        self.assertEqual(analyze_traces.duration(comm),10)
        self.assertEqual(analyze_traces.intersect(comm,compute),5)
        self.assertEqual(analyze_traces.intersect(compute,comm),5)
        self.assertEqual(analyze_traces.intersect([(0,5)],[(5,9)]),0)
    def test_hash_accepts_dunder_file_string_and_path(self):
        self.assertEqual(run.sha(str(HERE/'run.py')),run.sha(HERE/'run.py'))
    def setUp(self):
        self.plan=json.loads((HERE/'bundle/plan.json').read_text())
        self.pred=json.loads((HERE/'preregistered-predictions.json').read_text())
    def test_sealed_matrix(self):
        run.validate_plan(self.plan,self.pred)
        checks=[j for j in self.plan['jobs'] if j['phase']=='check']
        self.assertEqual(sum(len(j['order']) for j in checks),246)
        self.assertEqual(sum(len(j['order']) for j in self.plan['jobs'] if j['phase']=='verify'),234)
        self.assertEqual(sum(len(j['order']) for j in self.plan['jobs'] if j['phase']=='pair'),24)
    def test_missing_verification_or_seed_refused(self):
        for phase in ('check','verify','select'):
            p=copy.deepcopy(self.plan);idx=next(i for i,j in enumerate(p['jobs']) if j['phase']==phase);p['jobs'].pop(idx)
            with self.assertRaises(ValueError):run.validate_plan(p,self.pred)
    def test_selection_only_depends_on_frozen_ranks_and_correctness(self):
        tag='b1';ranking=self.pred['predictions'][tag]['frozen_ranking'];failed={(tag,ranking[0])}
        a=run.selection_order(self.plan,self.pred,tag,failed)
        self.assertNotIn(ranking[0],a)
        self.assertTrue(set(ranking[1:4])<=set(a))
        self.assertLessEqual(len(a),6)
        self.assertEqual(a,sorted(set(a)))
    def test_guard_bound_and_recovery(self):
        g=dict(authorization_confirmed=True,platform_verified=True,instance_id='test-only',
          experiment_schema='edmir.p1-p2.closeout.v1',billed_from_epoch=1000,platform_stop_epoch=9000,
          gpu_hourly_usd=10.8,storage_hourly_usd=.14,compute_cap_usd=24,total_budget_usd=25)
        self.assertEqual(run.cutoff(g,2000),8400)
        for k,v in [('compute_cap_usd',25),('total_budget_usd',26),('platform_stop_epoch',10000),('platform_verified',False)]:
            with self.assertRaises(ValueError):run.cutoff(dict(g,**{k:v}),2000)
        with self.assertRaises(TimeoutError):run.cutoff(g,8500)
    def test_frozen_files_unchanged(self):
        protected=json.loads((HERE/'protection-before.json').read_text())
        for name,h in protected.items():self.assertEqual(run.sha(Path(name)),h,name)
    def test_same_work_for_matched_pipeline(self):
        # Work descriptors match apart from synchronization/dependency policy.
        p=[json.loads((HERE/'bundle/m512'/name/'program.json').read_text()) for name in ('pipeline','pipeline_serial')]
        self.assertEqual(p[0]['buffers'],p[1]['buffers'])
        self.assertEqual(p[0]['ops'],p[1]['ops'])
        self.assertNotIn('serialize_pipeline_on_rank',p[0]['metadata'])
        self.assertIs(p[1]['metadata']['serialize_pipeline_on_rank'],True)
    def test_primary_p1_has_matched_seed_reverse_orders(self):
        jobs=json.loads((HERE/'balanced-p1-plan.json').read_text())['jobs']
        self.assertEqual(len(jobs),6)
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual(a['seed'],b['seed'])
            self.assertEqual(a['order'],list(reversed(b['order'])))
        for a,b in [('pipeline_serial','pipeline'),('splitk4','splitk4_groupedn2')]:
            self.assertEqual(sum(j['order'].index(a)<j['order'].index(b) for j in jobs),3)

if __name__=='__main__':unittest.main()
