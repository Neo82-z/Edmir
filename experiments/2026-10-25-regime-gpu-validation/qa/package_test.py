"""CPU-only contract/timeout checks for the exact movable GPU package."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(os.environ.get('EDMIR_REGIME_PACKAGE',str(ROOT/'experiments/2026-10-25-regime-gpu-validation/bundle'))).resolve()
sys.path[:0]=[str(PACKAGE/'source/EDM-IR'),str(PACKAGE/'source/EDM-IR/tools'),str(PACKAGE/'source/EDM-IR/tests')]
import prepare_mechanism_gpu as prepare
import run_mechanism_gpu as runner
from mechanism_gpu_package_test import row_for
from edmir.compiler.measurement_dataset import sha256
from independent_selector_validation import digest


class GPUContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=prepare.verify_bundle(PACKAGE)

    def test_all_stages_and_matched_process_blocks(self):
        self.assertEqual({s:v['process_runs'] for s,v in self.p['stages'].items()},
                         {'anchors':93,'crossings':54,'library':24,'warmup':12})
        for stage in prepare.STAGES:
            jobs=[j for j in self.p['jobs'] if j['stage']==stage]
            keys=lambda b:{(tuple(j['shape']),j['candidate_id']) for j in jobs if j['repetition']==b}
            self.assertEqual(keys(0),keys(1));self.assertEqual(keys(1),keys(2))
        self.assertFalse(self.p['independent_validation'])

    def test_execution_sources_match_measured_version(self):
        manifest=json.loads((PACKAGE.parent/'execution-source-identity.json').read_text())
        for rel,value in manifest['unchanged_execution_sources'].items():
            self.assertEqual(sha256(PACKAGE/'source'/rel),value)
        self.assertFalse(manifest['new_current_runtime_included'])

    def test_warmup_diagnostic_is_separate_and_validated(self):
        info={'input_sha256':'0'*64,'reference_sha256':'1'*64}
        for job in self.p['jobs']:
            r=row_for(job,self.p);r['warmup']=job.get('diagnostic_warmup',20)
            valid=runner.check_row(r,job,info,self.p)
            args=runner.measurement_args(PACKAGE,Path('/tmp/test-build'),self.p,job)
            self.assertEqual(args[4],r['warmup'])
            self.assertEqual(valid['warmup'],r['warmup'])
            if job['stage']=='warmup':
                with self.assertRaises(ValueError):runner.check_row({**r,'warmup':20},job,info,self.p)

    def test_every_pair_including_q4g2_vs_q8g2_is_reported(self):
        rows=[]
        for j in self.p['jobs']:
            if j['stage']=='anchors':rows.append({**row_for(j,self.p),'candidate_id':j['candidate_id'],'repetition':j['repetition']})
        report=runner.summarize(self.p,rows,'anchors')
        m256=next(r for r in report['workloads'] if r['shape']==[4,256,8192,512])
        self.assertTrue(any({r['a'],r['b']}=={'splitk4_groupedn2@prepared','splitk8_groupedn2@prepared'} for r in m256['paired_contrasts']))
        self.assertEqual(report,runner.summarize(self.p,rows[::-1],'anchors'))
        with self.assertRaises(ValueError):runner.summarize(self.p,rows[:-1],'anchors')

    def test_frozen_predictions_have_no_new_labels_or_fake_context(self):
        f=json.loads((PACKAGE.parent/'frozen-predictions.json').read_text())
        self.assertEqual(len(f['rows']),61)
        for r in f['rows']:
            self.assertIsNone(r['observed_us'])
            self.assertFalse(r['new_measurement_context_qualified'])
            if r['stage']=='warmup':self.assertIsNone(r['conditional_predictions'])
            else:self.assertEqual(set(r['conditional_predictions']),{'regime_max','residual_depth'})

    def test_time_budget_stops_before_launch_and_resume_rejects_context_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);package=root/'package';package.mkdir();out=root/'out'
            (package/'inputs.json').write_text((PACKAGE/'inputs.json').read_text())
            ctx={'runtime':{'CPU_test_only':True},'environment_id':'test'}
            with patch.object(runner,'preflight',return_value=(self.p,ctx,{})),patch.object(runner,'telemetry'),patch.object(runner.base,'execute_job') as launch:
                self.assertFalse(runner.collect(package,root/'build',out,stage='anchors',budget_seconds=1))
                launch.assert_not_called()
                with patch.object(runner,'preflight',return_value=(self.p,{**ctx,'environment_id':'changed'},{})):
                    with self.assertRaises(ValueError):runner.collect(package,root/'build',out,stage='anchors',resume=True)

    def test_fresh_review_preserves_diagnostics_and_checks_stdout_and_seal(self):
        spec=importlib.util.spec_from_file_location('fresh_review',PACKAGE.parent/'review.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        infos=json.loads((PACKAGE/'inputs.json').read_text())['workloads']
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);folder=work/'warmup-01';folder.mkdir()
            ctx={'stage':'warmup','runtime':{'test':'CPU bookkeeping only'},'package_sha256':digest(self.p)}
            (folder/'context.json').write_text(json.dumps(ctx));rows=[]
            for i,j in enumerate(j for j in self.p['jobs'] if j['stage']=='warmup'):
                raw=row_for(j,self.p);raw['warmup']=200
                rows.append(runner.check_row(raw,j,infos[j['workload_id']],self.p))
                (folder/f'stdout_{i:05d}.txt').write_text(json.dumps(raw))
            path=folder/'measurements.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (folder/'run.json').write_text(json.dumps({'context':ctx,'rows':rows,'raw_sha256':sha256(path)}))
            report=module.review(PACKAGE,work)
            self.assertFalse(report['fit_performed']);self.assertTrue(report['stages'][0]['complete'])
            self.assertTrue(all(not c['primary_timing_label'] and not c['forecast_errors'] for c in report['stages'][0]['configurations']))
            raw=json.loads((folder/'stdout_00000.txt').read_text());raw['warmup']=20
            (folder/'stdout_00000.txt').write_text(json.dumps(raw))
            with self.assertRaises(ValueError):module.review(PACKAGE,work)


if __name__=='__main__':unittest.main()
