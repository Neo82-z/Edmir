import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.shared_candidate_selection import VARIANTS
from edmir.compiler.measurement_dataset import sha256


class ReplayIntegrityTest(unittest.TestCase):
    def test_every_variant_has_unchanged_outer_and_boundary_coverage(self):
        path=Path(os.environ.get('SHARED_REVIEW_RESULTS',str(HERE/'initial/results.json')))
        result=json.loads(path.read_text())
        self.assertEqual([v['method'] for v in result['variants']],list(VARIANTS))
        for v in result['variants']:
            for protocol in ('primary','stress'):
                self.assertEqual(len(v[protocol]['decisions']),41)
                for fold in v[protocol]['fits']:
                    outer={tuple(r['shape']) for r in v[protocol]['decisions'] if r['heldout']==fold['heldout']}
                    model=fold['model'];train={tuple(s) for s in model['training_shapes']}
                    self.assertFalse(outer&train)
                    self.assertEqual(train,{tuple(s) for s in model['prior']['training_shapes']})
                    self.assertEqual(len(model['prior']['cv_scores']),12)
            self.assertEqual(sum(len(r['deletions']) for r in v['review']['deletion']),328)
            self.assertEqual(len(v['review']['boundary_stress']['records']),34)
            self.assertEqual(len(v['assessment']['checks']),5)
            self.assertEqual(v['assessment']['candidate_accepted'],all(v['assessment']['checks'].values()))
            for audit in v['optimization']['all_starts']:
                best=min(range(len(audit['starts'])),key=lambda i:(audit['starts'][i]['final_loss'],audit['starts'][i]['seed']))
                self.assertEqual(audit['selected_start'],best)
                selected=audit['starts'][best]
                self.assertTrue(selected['success'])
                self.assertLess(selected['gradient_max_abs'],1e-5)
        self.assertEqual(result['new_independent_labels'],0)
        self.assertFalse(result['compiler_admission'])

    def test_old_evidence_and_frozen_gpu_package_unchanged(self):
        for name in ('2026-10-21-additive-selector-repair','2026-10-21-host-boundary-repair',
                     '2026-10-21-structural-identifiability','2026-10-21-joint-decision-review'):
            path=ROOT/'experiments'/name/'verified/decision.json';old=json.loads(path.read_text())
            for relative,value in old['source_sha256'].items():self.assertEqual(sha256(ROOT/relative),value)
            for relative,value in old['evidence_sha256'].items():self.assertEqual(sha256(path.parent/relative),value)
        folder=ROOT/'experiments/2026-10-21-boundary-gpu-package'
        delivery=json.loads((folder/'delivery.json').read_text())
        self.assertEqual(sha256(folder/delivery['archive']),delivery['archive_sha256'])
        from prepare_boundary_gpu import verify_bundle
        verify_bundle(folder/'bundle')


if __name__=='__main__':unittest.main()
