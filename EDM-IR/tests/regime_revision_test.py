import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.regime_prediction import RegimeModel
from edmir.compiler.residual_regime_prediction import ResidualRegimeModel,residual_basis
from edmir.compiler.mechanism_prediction import HOSTS
from audit_regime_revision import folds,algorithm_audit


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((ROOT/'experiments/2026-10-25-final-local-revision/results/configurations.json').read_text())
        cls.primary=[r for r in cls.rows if r['primary_scope']]
        cls.stable=[r for r in cls.primary if r['eligible_for_calibration']]
        cls.model=ResidualRegimeModel.fit(cls.stable)

    def test_eight_parameters_and_nonzero_depth_survives_throughput(self):
        self.assertEqual(len(self.model.fused['names']),8)
        r=self.model.predict((4,512,4096,512),'splitk4',context_id=self.model.context_id)
        self.assertEqual(r['regime']['dominant_proxy'],'throughput')
        self.assertGreater(r['regime']['unhidden_depth_proxy_us'],0)
        self.assertEqual(r['contributions_us']['latency_depth_64'],0)
        self.assertAlmostEqual(math.fsum(r['contributions_us'].values()),r['predicted_us'])
        self.assertFalse(r['compiler_admission'])

    def test_all_hosts_are_numerically_unchanged(self):
        before=RegimeModel.fit(self.stable)
        for r in self.primary:
            if r['family'] not in HOSTS:continue
            a=before.predict(r['shape'],r['family'],context_id=before.context_id)
            b=self.model.predict(r['shape'],r['family'],context_id=self.model.context_id)
            self.assertEqual(a['predicted_us'],b['predicted_us'])
            self.assertEqual(a['contributions_us'],b['contributions_us'])

    def test_permutation_invariant_and_kkt_checked(self):
        self.assertEqual(self.model.to_dict(),ResidualRegimeModel.fit(self.stable[::-1]).to_dict())
        self.assertLess(self.model.fused['kkt_relative_violation'],1e-9)

    def test_whole_shape_label_noninterference(self):
        target=[4,256,8192,512]
        _,before=folds(self.primary,'residual_depth')
        changed=copy.deepcopy(self.primary)
        for r in changed:
            if r['shape']==target:r['observed_us']*=8
        _,after=folds(changed,'residual_depth')
        a=next(f for f in before if list(f['held_out'])==target)
        b=next(f for f in after if list(f['held_out'])==target)
        self.assertEqual(a['model'],b['model'])

    def test_host_quality_inclusion_cannot_change_fused_coefficients(self):
        flagged=[r for r in self.primary if r['quality_flags'] and r['family'] in HOSTS]
        other=ResidualRegimeModel.fit(self.stable+flagged,include_flagged=True)
        self.assertEqual(self.model.fused,other.fused)

    def test_algorithm_ids_are_audited_but_never_fitted(self):
        audit=algorithm_audit(self.primary)
        self.assertEqual(audit['distinct_full_configurations'],5)
        self.assertTrue(all(not r['exact_configuration_seen_in_training'] for r in audit['rows']))
        changed=copy.deepcopy(self.stable)
        for r in changed:r['algorithm_ids']=['synthetic_metadata_only']
        self.assertEqual(self.model.to_dict(),ResidualRegimeModel.fit(changed).to_dict())


if __name__=='__main__':unittest.main()
