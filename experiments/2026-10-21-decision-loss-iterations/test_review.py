import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.decision_loss_selection import DecisionLossSelector,VARIANTS,LOSS_GRID
from edmir.compiler.measurement_dataset import sha256


class IntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=json.loads(Path(os.environ.get('DECISION_REVIEW_RESULTS',str(HERE/'initial/results.json'))).read_text())

    def test_complete_unchanged_coverage(self):
        r=self.result
        self.assertEqual([v['method'] for v in r['variants']],list(VARIANTS))
        for v in r['variants']:
            for protocol in ('primary','stress'):
                self.assertEqual(len(v[protocol]['decisions']),41)
                for fold in v[protocol]['fits']:
                    held={tuple(row['shape']) for row in v[protocol]['decisions'] if row['heldout']==fold['heldout']}
                    model=DecisionLossSelector.from_dict(fold['model'])
                    train=set(model.prior.training_shapes)
                    self.assertFalse(train&held)
                    if v['method']=='traffic_nested':
                        self.assertEqual(len(model.cv),len(LOSS_GRID))
                        for entry in model.cv:
                            for inner in entry['folds']:
                                self.assertTrue(set(map(tuple,inner['training_shapes']))<train)
                                self.assertFalse(set(map(tuple,inner['training_shapes']))&held)
            self.assertEqual(sum(len(row['deletions']) for row in v['review']['deletion']),328)
            self.assertEqual(len(v['review']['boundary_stress']['records']),34)
            self.assertEqual(len(v['assessment']['checks']),5)
            self.assertEqual(v['assessment']['candidate_accepted'],all(v['assessment']['checks'].values()))
            self.assertTrue(v['optimization']['all_success'])
            self.assertLessEqual(v['optimization']['max_projected_gradient'],1e-5)
        self.assertFalse(r['compiler_admission']);self.assertEqual(r['new_independent_labels'],0)
        for path,value in r['source_sha256'].items():self.assertEqual(sha256(ROOT/path),value,path)

    def test_all_previous_evidence_and_gpu_bundle_unchanged(self):
        for name in ('2026-10-21-additive-selector-repair','2026-10-21-host-boundary-repair',
                     '2026-10-21-structural-identifiability','2026-10-21-joint-decision-review','2026-10-21-shared-candidate-review'):
            path=ROOT/'experiments'/name/'verified/decision.json';old=json.loads(path.read_text())
            for relative,value in old['source_sha256'].items():self.assertEqual(sha256(ROOT/relative),value,relative)
            for relative,value in old['evidence_sha256'].items():self.assertEqual(sha256(path.parent/relative),value,relative)
        folder=ROOT/'experiments/2026-10-21-boundary-gpu-package'
        delivery=json.loads((folder/'delivery.json').read_text())
        self.assertEqual(sha256(folder/delivery['archive']),delivery['archive_sha256'])
        from prepare_boundary_gpu import verify_bundle
        verify_bundle(folder/'bundle')


if __name__=='__main__':unittest.main()
