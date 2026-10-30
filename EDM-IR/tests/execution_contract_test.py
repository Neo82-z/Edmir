"""Offline contract tests: no measured labels, fitting or GPU claims."""
import copy
from dataclasses import replace
import itertools
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler import A2AGemmProblem, CppManifestBackend, Target, TopologySnapshot
from edmir.compiler.schedule import build_fused_program, build_pipelined_program, build_sequential_program
from edmir.compiler.dependence import derive_dependencies
from edmir.compiler.model import Program, Op
from edmir.compiler.execution_policy import LEGACY_POLICY, UNIFORM_PEER_POLICY, with_dependency_policy
from edmir.compiler.execution_ledger import execution_ledger, execution_ledgers, source_provenance, SOURCE_PATHS, digest
from edmir.compiler.evidence_selection import TimingEvidence, assess_candidates

ROOT = Path(__file__).resolve().parents[2]
SHAPE = (4, 512, 4096, 512)


def ledger(family, **kwargs):
    return execution_ledger(SHAPE, family, environment_id="synthetic-contract-test", **kwargs)


def chain_length(result):
    predecessors = result.predecessors()
    depths = {}
    for name in result.topological_order:
        depths[name] = 1 + max((depths[p] for p in predecessors[name]), default=0)
    return max(depths.values())


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.problem = A2AGemmProblem(*SHAPE)
        self.program = build_fused_program(self.problem, chunk_rows=512)

    def test_explicit_policy_and_serialization_roundtrip(self):
        before = derive_dependencies(self.program)
        program = with_dependency_policy(self.program, UNIFORM_PEER_POLICY)
        after = derive_dependencies(Program.from_dict(program.to_dict()))
        after.require_valid()
        self.assertEqual((len(before.dependencies), chain_length(before)), (31, 11))
        self.assertEqual((len(after.dependencies), chain_length(after)), (12, 4))
        self.assertEqual(self.program.ops, program.ops)
        self.assertEqual(self.program.buffers, program.buffers)
        self.assertEqual(derive_dependencies(self.program), before)
        self.assertNotEqual(ledger('splitk2')['identity_sha256'],
                            ledger('splitk2', dependency_policy=UNIFORM_PEER_POLICY)['identity_sha256'])

    def test_policy_rejects_chunking_mutation_aliasing_and_producers(self):
        variants = [build_fused_program(self.problem, chunk_rows=256),
                    build_sequential_program(self.problem),
                    replace(self.program, ops=self.program.ops[:-1]),
                    replace(self.program, metadata={}),
                    replace(self.program, ops=(replace(self.program.ops[0], writes=self.program.ops[1].writes),)
                            + self.program.ops[1:]),
                    replace(self.program, ops=(Op('producer', 'gemm', 0, 'compute'),)+self.program.ops)]
        for program in variants:
            with self.subTest(program=program.ops[0].id):
                with self.assertRaises(ValueError):
                    with_dependency_policy(program, UNIFORM_PEER_POLICY)
        with self.assertRaises(ValueError):with_dependency_policy(self.program, "typo")

    def test_existing_hazards_are_preserved(self):
        program = build_pipelined_program(self.problem, chunk_rows=256, buffer_count=1)
        result = derive_dependencies(program)
        result.require_valid()
        kinds = {edge.kind for edge in result.dependencies}
        self.assertTrue({'read_after_write', 'buffer_reuse', 'write_after_write'} <= kinds)
        with self.assertRaises(ValueError):with_dependency_policy(program, UNIFORM_PEER_POLICY)

    def test_codegen_carries_policy_and_rejects_other_kernel_providers(self):
        program = with_dependency_policy(self.program, UNIFORM_PEER_POLICY)
        topology = TopologySnapshot(4, ('test',)*4, ((True,)*4,)*4)
        target = Target('test', topology, arch='sm_80', gemm_provider='cutlass',
                        gemm_kernel='sm80_64x128x32_splitk2', communication_api='nccl_device')
        header = CppManifestBackend().emit(program, target).files['edmir_program.hpp']
        self.assertIn('kUniformImmutablePeerV1', header)
        self.assertIn('std::array<DependencyDesc, 12>', header)
        with self.assertRaises(ValueError):
            CppManifestBackend().emit(program, replace(target, gemm_provider='cuda_reference', gemm_kernel='lsa_naive_f16'))


class IdentityAndHostTest(unittest.TestCase):
    def test_resource_definition_mutation_rejected_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, relative in SOURCE_PATHS.items():
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT/relative).read_bytes())
            model = root/SOURCE_PATHS['model']
            model.write_text(model.read_text().replace('return (*resources, f"rank{self.rank}.compute")',
                                                       'return (f"rank{self.rank}.compute",)'))
            with self.assertRaisesRegex(ValueError, 'model.*re-audit'):
                ledger('splitk2', provenance=source_provenance(root))

    def test_changed_graph_or_source_cannot_reuse_identity(self):
        for field in ('execution_graph', 'provenance'):
            row = ledger('splitk2')
            row[field]['mutated'] = True
            with self.assertRaisesRegex(ValueError, 'graph or source'):
                assess_candidates({'splitk2': row})

    def test_old_schema_and_old_evidence_rejected(self):
        old = ledger('splitk2')
        old['schema'] = 'edmir.execution-ledger.v1'
        with self.assertRaises(ValueError):assess_candidates({'splitk2': old})
        before = ledger('splitk2')
        after = ledger('splitk2', dependency_policy=UNIFORM_PEER_POLICY)
        evidence = TimingEvidence(before['identity_sha256'], 'synthetic', (('0',1),('1',2),('2',3)), ('fixture',), True)
        with self.assertRaises(ValueError):assess_candidates({'splitk2':after}, evidence={'splitk2':evidence})

    def test_merged_counts_total_work_and_library_unknowns(self):
        a, b = ledger('host'), ledger('host_merged_cutlass')
        counts = lambda x, n: x['counting_identities'][n]['value']
        self.assertEqual(counts(b,'gemm_launches'), 4)
        self.assertEqual(counts(b,'final_stream_synchronizations'), 8)
        self.assertEqual(counts(a,'logical_flops')*counts(a,'gemm_launches'),
                         counts(b,'logical_flops')*counts(b,'gemm_launches'))
        for field in ('remote_unique_input_bytes','explicit_transfer_payload_bytes','local_copy_payload_bytes', 'staging_allocation_bytes'):
            self.assertEqual(counts(a,field),counts(b,field))
        self.assertEqual(b['source_facts']['gemm_mnk'], [2048,512,4096])
        for family in ('host_compiler_cublas', 'host_merged_cublaslt'):
            lib = ledger(family)
            self.assertIsNone(lib['source_facts']['tile_mnk'])
            self.assertNotIn('logical_a_load_bytes',lib['counting_identities'])
            self.assertNotIn('ctas',lib['counting_identities'])

    def test_strong_host_and_fused_share_context_without_identity_aliasing(self):
        names = ('host','host_compiler_cublas','host_merged_cutlass','host_merged_cublaslt','splitk2')
        rows = execution_ledgers(SHAPE, names, environment_id='synthetic-contract-test',
                                 metadata_mode='prepared', dependency_policy=UNIFORM_PEER_POLICY,
                                 algorithm_ids={'host_merged_cublaslt':'synthetic-recorded-algorithm'})
        self.assertEqual(len({v['identity_sha256'] for v in rows.values()}),len(names))
        for best in names:
            evidence = {n:TimingEvidence(row['identity_sha256'],'synthetic matched block',
                        tuple((str(i),100+i if n==best else 200+i) for i in range(3)),('synthetic',),True)
                        for n,row in rows.items()}
            for order in (names, tuple(reversed(names))):
                self.assertEqual(assess_candidates({n:rows[n] for n in order},evidence=evidence)['choice'],best)
        missing = ledger('host_merged_cublaslt')
        item = TimingEvidence(missing['identity_sha256'],'synthetic', (('0',1),('1',2),('2',3)), ('fixture',), True)
        with self.assertRaisesRegex(ValueError, 'algorithm'):
            assess_candidates({'host_merged_cublaslt':missing}, evidence={'host_merged_cublaslt':item})

    def test_real_comparison_context_mismatches_still_rejected(self):
        a, b = ledger('host'), ledger('splitk2')
        for key, value in (('environment_id','different-hardware'), ('shape',[4,64,4096,512]),
                           ('timing_scope','kernel-only'), ('input_readiness','unknown')):
            changed = copy.deepcopy(b)
            changed['identity']['comparison_context'][key] = value
            changed['identity_sha256'] = digest(changed['identity'])
            with self.assertRaises(ValueError):assess_candidates({'host':a,'splitk2':changed})


if __name__ == '__main__':
    unittest.main()
