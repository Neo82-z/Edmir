import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
import prepare_boundary_gpu as prepare
import run_boundary_gpu as runner
from independent_selector_validation import digest

PLAN=ROOT/'experiments/2026-10-21-structural-identifiability/verified/next_measurements.json'


def row_for(job,p,t=1.):
    q=1 if job['family'] in ('host','direct') else int(job['family'].removeprefix('splitk'))
    return {'family':job['family'],'shape':job['shape'],'metadata_mode':'legacy',
            'implementation_id':job['implementation_id'],'gemm_algorithm_id':'sm80_64x128x32'+(f'_splitk{q}' if q>1 else ''),
            'correctness_passed':True,'nonfinite_outputs':0,'checked_elements':16*job['shape'][1]*job['shape'][3],
            'max_abs_error':0.,'max_rel_error':0.,'atol':.02,'rtol':.02,
            'correctness_reference':'CPU_FP32_on_identical_FP16_inputs','input_seed':20261020,
            'warmup':20,'iterations':100,'samples_ms':[t]*100,'median_ms':t,
            'mean_ms':t,'total_ms':100*t,'min_ms':t,'max_ms':t,'timing_scope':p['timing_scope'],
            'measurement_policy':p['measurement_policy'],'runtime':{'test':'CPU-only control flow'}}


class BoundaryPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'bundle'
        cls.p=prepare.prepare(cls.root,PLAN)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_complete_frozen_plan_programs_sources_and_forecasts(self):
        p=prepare.verify_bundle(self.root,require_inputs=False)
        self.assertEqual(len(p['programs']),35);self.assertEqual(len(p['jobs']),105)
        self.assertEqual([j['ordinal'] for j in p['jobs']],list(range(105)))
        self.assertTrue(all(j['metadata_mode']=='legacy' for j in p['jobs']))
        forecasts=json.loads((self.root/'frozen_predictions.json').read_text())
        self.assertEqual(forecasts['fit_operations'],0);self.assertEqual(len(forecasts['models']),4)
        for m in forecasts['models']:
            unsupported=[f for f in m['forecasts'] if f['status']=='outside_model_domain']
            self.assertEqual([f['shape'] for f in unsupported],[[4,192,3072,1536]])
            self.assertIsNone(unsupported[0]['choice'])
        with self.assertRaises(FileNotFoundError):prepare.verify_bundle(self.root)
        with self.assertRaises(ValueError):runner.require_frozen_runner(self.root)

    def test_rehashed_contract_and_order_mutations_rejected(self):
        for modify in (
            lambda p:p.update(warmup=10),lambda p:p.update(timing_scope='device events'),
            lambda p:p['jobs'][0].update(metadata_mode='prepared'),
            lambda p:p['jobs'].reverse(),lambda p:p['programs'].pop(next(iter(p['programs'])))):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'bundle';shutil.copytree(self.root,root)
                obj=copy.deepcopy(self.p);modify(obj)
                (root/'package.json').write_text(json.dumps({'package_sha256':digest(obj),'package':obj}))
                with self.assertRaises(ValueError):prepare.verify_bundle(root,require_inputs=False)

    def test_changed_snapshot_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'bundle';shutil.copytree(self.root,root)
            path=root/'source'/next(iter(self.p['sources']))
            path.write_text(path.read_text()+'\n# altered\n')
            with self.assertRaises(ValueError):prepare.verify_bundle(root,require_inputs=False)

    def test_label_identity_and_all_samples_are_enforced(self):
        job=self.p['jobs'][0];r=row_for(job,self.p);info={'input_sha256':'0'*64,'reference_sha256':'1'*64}
        valid=runner.check_row(r,job,info,self.p)
        self.assertEqual(runner.check_row(valid,job,info,self.p),valid)
        for changes in ({'metadata_mode':'prepared'},{'gemm_algorithm_id':'cublas'},
                        {'samples_ms':[1.]*99},{'median_ms':2.},{'correctness_passed':False},
                        {'measurement_policy':{'profiler':True,'nvtx':False,'discarded_samples':0}}):
            with self.assertRaises(ValueError):runner.check_row({**r,**changes},job,info,self.p)

    def test_summary_pairs_blocks_and_has_no_full_oracle_claim(self):
        rows=[]
        for j in self.p['jobs']:
            _,m,k,n=j['shape'];t=1.+m*n/1e6+j['repetition']*.1
            rows.append({**row_for(j,self.p,t),'repetition':j['repetition']})
        summary=runner.summarize(self.p,list(reversed(rows)))
        self.assertFalse(summary['full_candidate_oracle'])
        self.assertFalse(summary['independent_validation'])
        expected=(192-64)*(1536-512)/1e6
        for contrast in summary['paired_contrasts']:
            for value in contrast['crossed_interaction_ms']:self.assertAlmostEqual(value,expected)
        with self.assertRaises(ValueError):runner.summarize(self.p,rows[:-1])
        with self.assertRaises(ValueError):runner.summarize(self.p,rows[:-1]+rows[:1])

    def test_real_subprocess_timeout_resume_and_profile_separation(self):
        # Only GPU hardware inspection is replaced. Collector validation and
        # fresh-process timeout/resume bookkeeping execute normally.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package=root/'p';build=root/'b';out=root/'out'
            package.mkdir();build.mkdir();p=copy.deepcopy(self.p);p['process_timeout_seconds']=1
            infos={w['id']:{'input_sha256':'0'*64,'reference_sha256':'1'*64} for w in p['workloads']}
            (package/'inputs.json').write_text(json.dumps({'workloads':infos}))
            context={'runtime':{'test':'CPU-only control flow'},'topology':'test','inventory':'test','environment_id':'boundary-test'}
            def exe(job,hang=False):
                path=build/job['executable']
                body='import time;print("partial",flush=True);time.sleep(30)' if hang else 'print('+repr(json.dumps(row_for(job,p)))+')'
                path.write_text(f'#!{sys.executable}\n{body}\n');path.chmod(0o755)
            for j in p['jobs'][:35]:exe(j)
            failed=p['jobs'][2];exe(failed,True)
            with patch.object(runner,'preflight',return_value=(p,context,{})),patch.object(runner.base,'environment',return_value={}),patch.object(runner,'telemetry'):
                with self.assertRaises(subprocess.TimeoutExpired):runner.collect(package,build,out)
                self.assertEqual(len((out/'measurements.jsonl').read_text().splitlines()),2)
                self.assertEqual(json.loads((out/'failure.json').read_text())['job_index'],2)
                self.assertIn('partial',(out/'stdout_00002.txt').read_text())
                exe(failed)
                with patch.object(runner.base,'execute_job',wraps=runner.base.execute_job) as launch:
                    runner.collect(package,build,out,resume=True)
                    self.assertEqual(launch.call_count,103)
                self.assertEqual(len((out/'measurements.jsonl').read_text().splitlines()),105)
                self.assertEqual(json.loads((out/'progress.json').read_text())['state'],'complete')
                with patch.object(runner,'preflight',return_value=(p,{**context,'environment_id':'different'},{})):
                    with self.assertRaises(ValueError):runner.collect(package,build,out,resume=True)
                nsys=root/'fake_nsys';nsys.write_text(f'#!{sys.executable}\nimport sys,pathlib\n'
                    'if "--version" in sys.argv: print("test profiler")\n'
                    'else: pathlib.Path(sys.argv[sys.argv.index("--output")+1]+".nsys-rep").write_text("not a real GPU trace")\n')
                nsys.chmod(0o755);profile=root/'profile'
                runner.trace(package,build,profile,job_index=0,nsys=str(nsys))
                receipt=json.loads((profile/'diagnostic.json').read_text())
                self.assertTrue(receipt['profiler']);self.assertEqual(receipt['state'],'complete')
                self.assertFalse((profile/'measurements.jsonl').exists());self.assertFalse((profile/'run.json').exists())


if __name__=='__main__':unittest.main()
