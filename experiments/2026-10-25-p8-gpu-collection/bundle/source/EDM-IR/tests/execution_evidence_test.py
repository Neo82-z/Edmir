import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'EDM-IR'))
from edmir.compiler.execution_ledger import execution_ledger, execution_ledgers, source_provenance, FAMILIES
from edmir.compiler.evidence_selection import TimingEvidence, assess_candidates
from edmir.compiler.host_baseline import MERGED_CUTLASS, PER_SOURCE_CUBLAS


def ledger(shape=(4,64,4096,512), family='splitk2', **kwargs):
    return execution_ledger(shape, family, environment_id='test-context', **kwargs)


def value(account, name):
    return account['counting_identities'][name]['value']


class ExecutionLedgerTest(unittest.TestCase):
    def test_counts_match_independent_valid_tile_enumeration(self):
        # Enumerate actual valid tile rectangles and K intervals. Do not use
        # the implementation's closed-form byte or CTA helpers as the oracle.
        for m,k,n in ((32,512,128),(65,1024,264),(192,3072,1536),(512,8192,512)):
            for name in FAMILIES:
                x = ledger((4,m,k,n), name)
                tm,tn,_ = x['source_facts']['tile_mnk']
                intervals = x['source_facts']['k_intervals']
                a=b=c=0
                for r in range(0,m,tm):
                    for s in range(0,n,tn):
                        for lo,hi in intervals:
                            a += 2*min(tm,m-r)*(hi-lo)
                            b += 2*min(tn,n-s)*(hi-lo)
                            c += 1
                self.assertEqual((value(x,'logical_a_load_bytes'),value(x,'logical_b_load_bytes'),value(x,'ctas')),(a,b,c))
                self.assertEqual(sum(hi-lo for lo,hi in intervals),k)

    def test_same_grid_preserves_M_N_working_sets(self):
        a=ledger((4,64,4096,2048)); b=ledger((4,256,4096,512))
        self.assertEqual(value(a,'ctas'),value(b,'ctas'))
        self.assertEqual(value(a,'logical_a_load_bytes'),value(b,'logical_a_load_bytes'))
        self.assertEqual(value(a,'logical_b_load_bytes'),value(b,'logical_b_load_bytes'))
        self.assertEqual(value(a,'partition_a_bytes')*4,value(b,'partition_a_bytes'))
        self.assertEqual(value(a,'partition_b_bytes'),value(b,'partition_b_bytes')*4)

    def test_grouping_both_replay_and_resource_effects(self):
        a=ledger(); b=ledger(family='splitk2_groupedn2')
        self.assertEqual(value(a,'logical_a_load_bytes'),2*value(b,'logical_a_load_bytes'))
        self.assertEqual(value(a,'logical_b_load_bytes'),value(b,'logical_b_load_bytes'))
        self.assertEqual(a['source_facts']['threads_per_cta']*2,b['source_facts']['threads_per_cta'])
        self.assertEqual(value(a,'ctas'),2*value(b,'ctas'))
        self.assertIn('residency',{r['id'] for r in b['unknowns']})

    def test_split_k_costs_and_preparation_scope(self):
        for q in (1,2,4,8,16):
            x=ledger(family='direct' if q==1 else f'splitk{q}')
            self.assertEqual(value(x,'output_write_bytes'),2*64*512*q)
            self.assertEqual(value(x,'output_read_bytes'),2*64*512*(q-1))
            self.assertEqual(value(x,'extra_output_bytes'),4*64*512*(q-1))
            self.assertEqual(value(x,'workspace_bytes'),0 if q==1 else 4*4*16)
            self.assertEqual(x['source_facts']['split_k_serial'],q>1)
            self.assertIsNone(x['latency_us'])
            self.assertTrue(any('outside steady timing' in s for s in x['source_facts']['runtime_contract']))

    def test_host_buffers_transfers_and_dependencies_come_from_program(self):
        x=ledger(family='host'); program=x['source_facts']['program']
        self.assertEqual(x['source_facts']['operation_counts'],{'barrier':4,'gemm':16,'transfer':16})
        self.assertEqual(value(x,'staging_allocation_bytes'),sum(b['size_bytes'] for b in program['buffers'] if b['role']=='staging'))
        self.assertEqual(value(x,'explicit_transfer_payload_bytes'),12*2*64*4096)
        self.assertEqual(value(x,'local_copy_payload_bytes'),4*2*64*4096)
        self.assertEqual(value(x,'final_stream_synchronizations'),12)
        by_id={op['id']:op for op in program['ops']}
        edges=x['source_facts']['program_dependencies']
        for op in program['ops']:
            if op['kind']=='gemm':
                self.assertTrue(any(e['dst']==op['id'] and by_id[e['src']]['kind']=='barrier' for e in edges))
        y=ledger(family='direct')
        self.assertEqual(value(y,'staging_allocation_bytes'),0)
        self.assertEqual(value(y,'explicit_transfer_payload_bytes'),0)
        self.assertGreater(value(y,'remote_logical_a_load_bytes'),0)

    def test_identity_distinguishes_mode_and_rejects_mislabeled_hosts(self):
        a=ledger(); b=ledger(metadata_mode='prepared')
        self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])
        for identity in (MERGED_CUTLASS.baseline_id, PER_SOURCE_CUBLAS.baseline_id):
            with self.assertRaises(ValueError):ledger(baseline_id=identity)

    def test_bounded_inputs_and_nondivisible_partitions(self):
        for shape in ((True,64,4096,512),(10000,64,4096,512),(4,0,4096,512),(4,64,4097,512),(4,64,1056,512)):
            with self.assertRaises(ValueError):ledger(shape, 'splitk16')
        with self.assertRaises(ValueError):execution_ledgers((4,64,4096,512),['host','host'],environment_id='x')

    def test_changed_source_does_not_silently_reuse_old_formulas(self):
        for source in ('lowering','kernel_type','runtime','timer'):
            provenance=source_provenance()
            provenance['project'][source]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'re-audit'):
                ledger(provenance=provenance)


class EvidenceSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rows=execution_ledgers((4,512,4096,512),('host','splitk2','splitk4'),environment_id='test-context')

    def evidence(self, times):
        return {n:TimingEvidence(self.rows[n]['identity_sha256'],'test-only synthetic blocks',
                    tuple((str(i),v) for i,v in enumerate(values)),('synthetic test fixture, not GPU data',),True)
                for n,values in times.items()}

    def test_point_predictions_never_manufacture_choice(self):
        out=assess_candidates(self.rows,model_predictions_us={'host':700,'splitk2':699.52,'splitk4':705})
        self.assertIsNone(out['choice'])
        self.assertEqual(out['retained_candidates'],sorted(self.rows))
        self.assertFalse(out['model_reference']['used_for_selection'])

    def test_host_and_fused_both_can_have_separated_evidence(self):
        for best in self.rows:
            times={n:[100,101,102] if n==best else [110,111,112] for n in self.rows}
            out=assess_candidates(self.rows,evidence=self.evidence(times))
            self.assertEqual(out['choice'],best)
            self.assertEqual(out['status'],'separated_under_supplied_evidence')
            self.assertFalse(out['eligible_for_compiler_admission'])

    def test_overlap_touching_and_equal_times_remain_unresolved(self):
        for v in ([100,101,102],[102,103,104],[101,102,103]):
            e=self.evidence({'host':[100,101,102],'splitk2':v,'splitk4':[200,201,202]})
            out=assess_candidates(self.rows,evidence=e)
            self.assertIsNone(out['choice'])
            self.assertEqual(out['retained_candidates'],['host','splitk2'])

    def test_missing_evidence_cannot_be_eliminated(self):
        e=self.evidence({'host':[100,101,102],'splitk2':[110,111,112]})
        out=assess_candidates(self.rows,evidence=e)
        self.assertIsNone(out['choice'])
        self.assertEqual(out['missing_evidence'],['splitk4'])
        self.assertEqual(out['retained_candidates'],['host','splitk4'])

    def test_candidate_order_invariance(self):
        e=self.evidence({'host':[100,101,102],'splitk2':[110,111,112],'splitk4':[200,201,202]})
        expected=assess_candidates(self.rows,evidence=e)
        for order in itertools.permutations(self.rows):
            self.assertEqual(expected,assess_candidates({n:self.rows[n] for n in order},evidence={n:e[n] for n in order}))

    def test_identity_profiler_context_and_block_mismatch(self):
        e=self.evidence({n:[100,101,102] for n in self.rows})
        for invalid in (replace(e['host'],identity_sha256='0'*64),replace(e['host'],profiler=True),
                        replace(e['host'],correctness_passed=False),replace(e['host'],measurement_context='other'),
                        replace(e['host'],process_medians_us=(('a',100),('b',101),('c',102))),
                        replace(e['host'],process_medians_us=(('a',100),('a',101),('b',102))),
                        replace(e['host'],process_medians_us=(('0',100),('1',float('nan')),('2',102)))):
            with self.assertRaises(ValueError):assess_candidates(self.rows,evidence={**e,'host':invalid})
        mixed=copy.deepcopy(self.rows)
        mixed['host']=ledger((4,512,4096,512),'host',metadata_mode='prepared')
        # Metadata is candidate identity, not a reason to prohibit a measured
        # implementation comparison. Old timing still cannot be relabeled.
        self.assertIsNone(assess_candidates(mixed)['choice'])
        with self.assertRaises(ValueError):assess_candidates(mixed,evidence=e)

    def test_margin_applies_symmetrically(self):
        e=self.evidence({'host':[100,101,102],'splitk2':[103,104,105],'splitk4':[200,201,202]})
        self.assertEqual(assess_candidates(self.rows,evidence=e)['choice'],'host')
        self.assertIsNone(assess_candidates(self.rows,evidence=e,minimum_margin_us=1)['choice'])

    def test_single_candidate_is_not_a_comparison(self):
        e=self.evidence({'host':[100,101,102]})
        out=assess_candidates({'host':self.rows['host']},evidence=e)
        self.assertIsNone(out['choice'])
        self.assertEqual(out['status'],'insufficient_comparison')


class LocalEntryPointTest(unittest.TestCase):
    def test_builtin_cli_matches_standalone_contract(self):
        import contextlib
        import io
        from edmir.cli import main
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['explain-execution','--m','64','--k','4096','--n','2048',
                                   '--environment','test','--families','host','splitk2']),0)
        report=json.loads(output.getvalue())
        self.assertIsNone(report['decision']['choice'])
        self.assertEqual(report['decision']['retained_candidates'],['host','splitk2'])

    def test_cli_works_without_site_packages_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json'
            command=[sys.executable,'-S',str(ROOT/'EDM-IR/tools/explain_execution.py'),
                     '--m','512','--k','4096','--n','512','--environment','planned','--out',str(path)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            original=path.read_bytes(); report=json.loads(original)
            self.assertEqual(len(report['ledgers']),11)
            self.assertIsNone(report['decision']['choice'])
            second=subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':
    unittest.main()
