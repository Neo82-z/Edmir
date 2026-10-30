from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from candidate import Candidate, ROOT, fused_features
from diagnose import ambiguity, certified_lp
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.relative_selection import ScheduleCandidate
from edmir.compiler.host_baseline import PER_SOURCE_CUBLAS
from predict_traffic_schedule import static_candidates


class StructuralAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,*_=load_identified_selector_dataset(ROOT/'experiments/2026-10-20-offline-predictor/data')

    def test_lp_optimality_certificate_and_infeasibility(self):
        x,c=certified_lp([-1.,-1.],[[1.,1.]],[3.],[(0.,None)]*2)
        self.assertAlmostEqual(sum(x),3.)
        self.assertLess(c['duality_gap'],1e-9)
        x,c=certified_lp([1.],[[1.] ],[-1.],[(0.,None)])
        self.assertIsNone(x);self.assertEqual(c['status'],'infeasible')

    def test_old_feature_collision_removed(self):
        a=static_candidates(64,8192,2048,self.data.environment_id)['splitk4']
        b=static_candidates(256,8192,512,self.data.environment_id)['splitk4']
        self.assertNotEqual(fused_features(a,32),fused_features(b,32))

    def test_heldout_labels_cannot_affect_feasible_models(self):
        surface=next(s for s in self.data.validate().values() if s['host'].shape==(4,512,4096,512))
        train=replace(self.data,rows=tuple(r for r in self.data.rows if r.group_id!=surface['host'].group_id))
        # Reverse observed timing order without changing static inference inputs.
        altered={n:replace(r,measured_ms=10./r.measured_ms) for n,r in surface.items()}
        a,b=ambiguity(train,surface),ambiguity(train,altered)
        for obj in (a,b):
            for e in obj['ambiguity']:
                for w in e['possible_winners']:w.pop('heldout_regret_reported_after_solving')
        self.assertEqual(a,b)

    def test_no_admission_and_baseline_or_domain_substitution(self):
        m=Candidate.fit(self.data)
        rows=static_candidates(64,4096,512,self.data.environment_id)
        self.assertFalse(m.select(rows,baseline=self.data.baseline)['eligible_for_compiler_admission'])
        with self.assertRaises(ValueError):m.select(rows,baseline=PER_SOURCE_CUBLAS)
        with self.assertRaises(ValueError):m.select(static_candidates(192,3072,1536,self.data.environment_id),baseline=self.data.baseline)
        with self.assertRaises(TypeError):m.select(next(iter(self.data.validate().values())),baseline=self.data.baseline)
        with self.assertRaises(TimeoutError):Candidate.fit(self.data,timeout_seconds=1e-12)

    def test_fold_isolation_and_frozen_trial_coverage(self):
        result=json.loads(Path(__file__).with_name('results.json').read_text())
        self.assertEqual(len(result['variants']),2)
        for v in result['variants']:
            for protocol in ('primary','stress'):
                for fold in v[protocol]['fits']:
                    held={tuple(r['shape']) for r in v[protocol]['decisions'] if r['heldout']==fold['heldout']}
                    self.assertFalse(held&{tuple(s) for s in fold['model']['training_shapes']})
                    self.assertEqual(len(fold['model']['host_cv']),48)
                    self.assertEqual(len(fold['model']['fused_cv']),12)
            self.assertEqual(len(v['review']['boundary_stress']['records']),34)
            self.assertEqual(sum(len(r['deletions']) for r in v['review']['deletion']),328)
            self.assertFalse(v['assessment']['candidate_accepted'])

    def test_previous_source_and_evidence_preserved(self):
        for folder in ('2026-10-21-additive-selector-repair','2026-10-21-host-boundary-repair'):
            path=ROOT/'experiments'/folder/'verified/decision.json'
            d=json.loads(path.read_text())
            for rel,digest in d['source_sha256'].items():self.assertEqual(sha256(ROOT/rel),digest,rel)
            for name,digest in d['evidence_sha256'].items():self.assertEqual(sha256(path.parent/name),digest,name)

    def test_measurement_design_breaks_axis_alias_without_admitting_new_domain(self):
        from measurement_design import build
        plan=build()
        self.assertFalse(plan['execution_ready'])
        self.assertIsNone(plan['environment_id'])
        self.assertEqual((plan['crossed_design_rank_before'],plan['crossed_design_rank_after']),(3,4))
        self.assertEqual(len(plan['new_shapes']),5)
        runs=plan['run_order']
        self.assertEqual([r['ordinal'] for r in runs],list(range(105)))
        for block in range(3):
            pairs={(tuple(r['shape']),r['family']) for r in runs if r['block']==block}
            self.assertEqual(len(pairs),35)
        self.assertEqual(sum(not r['within_current_shape_domain'] for r in runs),15)


if __name__=='__main__':unittest.main()
