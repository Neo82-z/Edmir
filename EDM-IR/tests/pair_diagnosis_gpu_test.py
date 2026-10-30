from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_pair_diagnosis_gpu as prep
import run_pair_diagnosis_gpu as runner


def fixture_row(job,p):
    samples=[1.+i/10000 for i in range(p['iterations'])]
    q=1 if job['family']=='host' else runner.parse_schedule_family(job['family'])[0]
    P,M,_,N=job['shape']
    return {'family':job['family'],'shape':job['shape'],'implementation_id':job['implementation_id'],
      'gemm_algorithm_id':'sm80_64x128x32'+(f'_splitk{q}' if q>1 else ''),
      'correctness_passed':True,'nonfinite_outputs':0,'checked_elements':P*P*M*N,
      'max_abs_error':.001,'max_rel_error':.001,'atol':.02,'rtol':.02,
      'correctness_reference':'CPU_FP32_on_identical_FP16_inputs','input_seed':20261020,
      'warmup':p['warmup'],'iterations':p['iterations'],'samples_ms':samples,
      'median_ms':statistics.median(samples),'mean_ms':statistics.mean(samples),'total_ms':sum(samples),
      'min_ms':min(samples),'max_ms':max(samples),'metadata_mode':job['metadata_mode'],
      'dependency_policy':job['dependency_policy'],'input_readiness':'completed_upload_v1',
      'timing_scope':p['timing_scope'],'measurement_policy':p['measurement_policy'],
      'runtime':{'test':'CPU bookkeeping only; not GPU evidence'}}


class PairDiagnosisTests(unittest.TestCase):
    def test_exact_candidate_coverage_and_position_balance(self):
        for profile,n,runs in [('p4',20,70),('p8-grouped',5,25)]:
            p=prep.plan(profile)
            self.assertEqual(len(p['programs']),n);self.assertEqual(len(p['jobs']),runs)
            for w in p['workloads']:
                orders=[[j['family'] for j in p['jobs'] if j['workload_id']==w['id'] and j['repetition']==b] for b in range(w['blocks'])]
                for order in orders:self.assertEqual(set(order),set(w['families']))
                for pos in range(len(w['families'])):self.assertEqual({o[pos] for o in orders},set(w['families']))
            self.assertFalse(p['new_training']);self.assertFalse(p['frozen_six_used'])

    def test_row_identity_correctness_samples_and_algorithm(self):
        p=prep.plan('p4');job=p['jobs'][0];row=fixture_row(job,p);info={'input_sha256':'a','reference_sha256':'b'}
        runner.check_row(row,job,info,p)
        corruptions=[('shape',[8,*job['shape'][1:]]),('max_abs_error',1.),('samples_ms',[1.]),
                     ('gemm_algorithm_id','library_default'),('implementation_id','other'),
                     ('metadata_mode','legacy'),('dependency_policy','uniform_immutable_peer_v1')]
        for key,value in corruptions:
            with self.subTest(key=key):
                bad=deepcopy(row);bad[key]=value
                if key=='max_abs_error':bad['max_rel_error']=1.
                with self.assertRaises(ValueError):runner.check_row(bad,job,info,p)

    def test_pairing_and_complete_matrix_do_not_hide_missing_rows(self):
        p=prep.plan('p8-grouped');info={'input_sha256':'a','reference_sha256':'b'}
        rows=[runner.check_row(fixture_row(j,p),j,info,p) for j in p['jobs']]
        full=runner.summarize(p,rows);partial=runner.summarize(p,rows[:-1])
        self.assertTrue(full['complete']);self.assertFalse(partial['complete'])
        self.assertEqual(len(full['workloads'][0]['grouped_interactions']),5)
        self.assertEqual(len(partial['workloads'][0]['grouped_interactions']),4)
        with self.assertRaises(ValueError):runner.summarize(p,rows+[rows[0]])

    def test_bundle_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';prep.prepare(root,'p8-grouped');prep.verify_bundle(root,require_inputs=False)
            path=root/'generated_targets.cmake';path.write_text(path.read_text()+'# modified\n')
            with self.assertRaisesRegex(ValueError,'changed'):prep.verify_bundle(root,require_inputs=False)

    def test_unsafe_source_path_is_rejected_even_with_rehashed_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';prep.prepare(root,'p8-grouped')
            envelope=json.loads((root/'package.json').read_text());p=envelope['package']
            p['sources']['../outside']='not-a-hash';p['collector_sha256']=prep.digest(p['sources'])
            envelope['package_sha256']=prep.digest(p);prep.write(root/'package.json',envelope)
            with self.assertRaisesRegex(ValueError,'unsafe'):prep.verify_bundle(root,require_inputs=False)

    def test_partial_failure_and_explicit_resume_keep_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'package';build=root/'build';out=root/'out';package.mkdir();build.mkdir()
            p=prep.plan('p4');p['jobs']=p['jobs'][:2];p['process_timeout_seconds']=1
            info={'input_sha256':'a','reference_sha256':'b'}
            prep.write(package/'inputs.json',{'workloads':{'w000':info}})
            runtime={'test':'CPU bookkeeping only; not GPU evidence'};context={'runtime':runtime,'test':'fixed'}
            state={'fail':True};launched=[]
            original=runner.base.execute_job
            def local_job(args,job,path,index,**kw):
                launched.append(index)
                source="import sys; print('partial',flush=True); sys.exit(7)" if index==1 and state['fail'] else 'print('+repr(json.dumps(fixture_row(job,p)))+')'
                return original([sys.executable,'-c',source],job,path,index,**kw)
            with patch.object(runner,'preflight',return_value=(p,context,{})),patch.object(runner,'telemetry'),patch.object(runner.base,'execute_job',side_effect=local_job):
                with self.assertRaises(Exception):runner.collect(package,build,out,budget_seconds=300)
                self.assertEqual(len((out/'measurements.jsonl').read_text().splitlines()),1)
                self.assertEqual(json.loads((out/'progress.json').read_text())['state'],'failed')
                self.assertIn('partial',(out/'stdout_00001.txt').read_text())
                deadline=json.loads((out/'budget.json').read_text())
                state['fail']=False;launched.clear()
                self.assertTrue(runner.collect(package,build,out,resume=True,budget_seconds=300))
                self.assertEqual(launched,[1]);self.assertEqual(deadline,json.loads((out/'budget.json').read_text()))
                self.assertIn('partial',(out/'stdout_00001.txt').read_text())
                self.assertEqual(len(list((out/'attempts').glob('00001-retry-*'))),1)

    def test_expired_budget_never_launches_or_extends_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();package=root/'package';package.mkdir()
            p=prep.plan('p4');context={'runtime':{'test':'cpu'}}
            prep.write(package/'inputs.json',{'workloads':{}});prep.write(out/'context.json',context)
            started=time.time()-1000;deadline={'started_unix':started,'deadline_unix':started+100,'budget_seconds':100}
            prep.write(out/'budget.json',deadline)
            with patch.object(runner,'preflight',return_value=(p,context,{})),patch.object(runner.base,'execute_job') as launch:
                self.assertFalse(runner.collect(package,root/'build',out,resume=True,budget_seconds=100))
                launch.assert_not_called()
            self.assertEqual(deadline,json.loads((out/'budget.json').read_text()))
            self.assertFalse(json.loads((out/'summary.json').read_text())['complete'])

    def test_recovery_archive_retains_incomplete_results_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();(out/'stderr.txt').write_text('failure evidence')
            prep.write(out/'summary.json',{'complete':False});archive=root/'result.tar.gz'
            runner.pack_results(out,archive)
            manifest=json.loads((out/'recovery-manifest.json').read_text())
            self.assertFalse(manifest['measurement_complete'])
            with tarfile.open(archive) as tar:
                for rel,h in manifest['files'].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(tar.extractfile('out/'+rel).read()).hexdigest(),h)
            with self.assertRaises(ValueError):runner.pack_results(out,archive)

    def test_active_collection_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();prep.write(out/'progress.json',{'state':'collecting'})
            with self.assertRaisesRegex(ValueError,'stop collection'):runner.pack_results(out,root/'result.tar.gz')


if __name__=='__main__':unittest.main()
