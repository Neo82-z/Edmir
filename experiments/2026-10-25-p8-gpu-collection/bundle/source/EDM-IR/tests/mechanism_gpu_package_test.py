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
import prepare_mechanism_gpu as prepare
import run_mechanism_gpu as runner
from independent_selector_validation import digest


def row_for(job,p,t=1.):
    family=job['family']
    if family=='host_merged_cublaslt': algo=''.join(f'rank{r}:0=1;workspace=0;budget=33554432|' for r in range(4))
    elif family=='host_compiler_cublas': algo='library_default'
    else:
        q=1 if family.startswith('host') or family=='direct' else int(family.split('_')[0].removeprefix('splitk'))
        algo='sm80_64x128x32'+(f'_splitk{q}' if q>1 else '')
    return {'family':family,'shape':job['shape'],'metadata_mode':job['metadata_mode'],
        'implementation_id':job['implementation_id'],'gemm_algorithm_id':algo,
        'correctness_passed':True,'nonfinite_outputs':0,'checked_elements':16*job['shape'][1]*job['shape'][3],
        'max_abs_error':0.,'max_rel_error':0.,'atol':.02,'rtol':.02,
        'correctness_reference':'CPU_FP32_on_identical_FP16_inputs','input_seed':20261020,
        'warmup':20,'iterations':100,'samples_ms':[t]*100,'median_ms':t,
        'mean_ms':t,'total_ms':100*t,'min_ms':t,'max_ms':t,'timing_scope':p['timing_scope'],
        'measurement_policy':p['measurement_policy'],'runtime':{'test':'CPU bookkeeping only'}}


class MechanismPackageTest(unittest.TestCase):
    def test_driver_terminal_styles_do_not_change_topology_contract(self):
        plain='\tGPU0 GPU1\nGPU0 X NV12\nGPU1 NV12 X\n'
        styled='\t\x1b[4mGPU0 GPU1\x1b[0m\nGPU0 X NV12\nGPU1 NV12 X\n'
        self.assertEqual(runner.plain_topology(styled),plain)
        self.assertEqual(runner.plain_topology(plain),plain)
        self.assertIn('PHB',runner.plain_topology(styled.replace('NV12','PHB')))

    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)/'bundle'
        cls.p=prepare.prepare(cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_plan_pairing_and_no_training(self):
        p=prepare.verify_bundle(self.root,require_inputs=False)
        self.assertEqual({s:v['process_runs'] for s,v in p['stages'].items()},
                         {'anchors':30,'mechanisms':177,'geometry':75})
        self.assertEqual(len(p['jobs']),282)
        self.assertFalse(p['new_training']);self.assertFalse(p['independent_validation'])
        for stage in prepare.STAGES:
            jobs=[j for j in p['jobs'] if j['stage']==stage]
            keys=lambda b:{(j['workload_id'],j['candidate_id']) for j in jobs if j['repetition']==b}
            self.assertEqual(keys(0),keys(1));self.assertEqual(keys(0),keys(2))
        accounts=json.loads((self.root/'execution_accounts.json').read_text())
        merged = accounts['w000_host_merged_cutlass@merged']
        self.assertEqual(merged['schema'],'edmir.execution-ledger.v2')
        self.assertEqual(merged['counting_identities']['gemm_launches']['value'],4)
        self.assertIsNone(accounts['w000_host_merged_cublaslt@merged']['identity']['algorithm_id'])
        with self.assertRaises(FileNotFoundError):prepare.verify_bundle(self.root)
        with self.assertRaises(ValueError):runner.require_frozen_runner(self.root)

    def test_changed_plan_even_if_rehashed_is_rejected(self):
        for change in (lambda p:p.update(iterations=5),lambda p:p['jobs'].reverse(),
                       lambda p:p['jobs'][0].update(metadata_mode='prepared')):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t)/'bundle';shutil.copytree(self.root,root);p=copy.deepcopy(self.p);change(p)
                (root/'package.json').write_text(json.dumps({'package_sha256':digest(p),'package':p}))
                with self.assertRaises(ValueError):prepare.verify_bundle(root,require_inputs=False)

    def test_changed_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'bundle';shutil.copytree(self.root,root)
            path=root/'source'/next(iter(self.p['sources']));path.write_text('changed')
            with self.assertRaises(ValueError):prepare.verify_bundle(root,require_inputs=False)

    def test_every_family_identity_and_numerical_policy(self):
        info={'input_sha256':'0'*64,'reference_sha256':'1'*64}
        for j in self.p['jobs']:
            r=row_for(j,self.p)
            valid=runner.check_row(r,j,info,self.p)
            self.assertEqual(runner.check_row(valid,j,info,self.p),valid)
        j=self.p['jobs'][0];r=row_for(j,self.p)
        for change in ({'gemm_algorithm_id':'other'},{'metadata_mode':'merged'},
            {'correctness_passed':False},{'samples_ms':[1.]*99},{'median_ms':2.},
            {'measurement_policy':{'profiler':True,'nvtx':False,'discarded_samples':0}}):
            with self.assertRaises(ValueError):runner.check_row({**r,**change},j,info,self.p)

    def test_summary_uses_matched_blocks_and_geometry(self):
        for stage in prepare.STAGES:
            rows=[]
            for j in self.p['jobs']:
                if j['stage']!=stage:continue
                _,m,k,n=j['shape'];t=1.+m*n/1e6+j['repetition']*.1
                rows.append({**row_for(j,self.p,t),'candidate_id':j['candidate_id'],'repetition':j['repetition']})
            a=runner.summarize(self.p,rows,stage)
            self.assertEqual(a,runner.summarize(self.p,list(reversed(rows)),stage))
            self.assertFalse(a['compiler_admission']);self.assertFalse(a['full_candidate_oracle'])
            with self.assertRaises(ValueError):runner.summarize(self.p,rows[:-1],stage)
            if stage=='geometry':
                for c in a['geometry_contrasts']:
                    for v in c['crossed_interaction_us']:self.assertAlmostEqual(v,1000*(192-64)*(1536-512)/1e6)

    def test_budget_timeout_resume_and_profile_isolation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);package=root/'p';build=root/'b';out=root/'out';package.mkdir();build.mkdir()
            p=copy.deepcopy(self.p);p['process_timeout_seconds']=.3
            jobs=[j for j in p['jobs'] if j['stage']=='anchors']
            infos={w['id']:{'input_sha256':'0'*64,'reference_sha256':'1'*64} for w in p['workloads']}
            (package/'inputs.json').write_text(json.dumps({'workloads':infos}))
            ctx={'runtime':{'test':'CPU bookkeeping only'},'environment_id':'local-test'}
            def args(package,build,p,j):
                if j['ordinal']==jobs[2]['ordinal'] and fail[0]:
                    code='import time;print("partial",flush=True);time.sleep(30)'
                else: code='print('+repr(json.dumps(row_for(j,p)))+')'
                return [sys.executable,'-c',code]
            fail=[True]
            with patch.object(runner,'preflight',return_value=(p,ctx,{})),patch.object(runner,'telemetry'),patch.object(runner,'measurement_args',side_effect=args):
                with patch.object(runner.base,'execute_job',wraps=runner.base.execute_job) as launch:
                    self.assertFalse(runner.collect(package,build,out,stage='anchors',budget_seconds=1))
                    self.assertEqual(launch.call_count,0)
                with self.assertRaises(subprocess.TimeoutExpired):runner.collect(package,build,out,stage='anchors',resume=True)
                self.assertEqual(len((out/'measurements.jsonl').read_text().splitlines()),2)
                fail[0]=False
                with patch.object(runner.base,'execute_job',wraps=runner.base.execute_job) as launch:
                    self.assertTrue(runner.collect(package,build,out,stage='anchors',resume=True))
                    self.assertEqual(launch.call_count,28)
                with patch.object(runner,'preflight',return_value=(p,{**ctx,'environment_id':'changed'},{})):
                    with self.assertRaises(ValueError):runner.collect(package,build,out,stage='anchors',resume=True)
                with self.assertRaises(ValueError):runner.collect(package,build,out,stage='mechanisms',resume=True)
                nsys=root/'fake_nsys';nsys.write_text(f'#!{sys.executable}\nimport sys,pathlib\n'
                    'if "--version" in sys.argv: print("test profiler")\n'
                    'else: pathlib.Path(sys.argv[sys.argv.index("--output")+1]+".nsys-rep").write_text("CPU test")\n')
                nsys.chmod(0o755);prof=root/'prof'
                runner.trace(package,build,prof,job_index=0,nsys=str(nsys))
                self.assertTrue(json.loads((prof/'diagnostic.json').read_text())['profiler'])
                self.assertFalse((prof/'measurements.jsonl').exists());self.assertFalse((prof/'run.json').exists())


if __name__=='__main__':unittest.main()
