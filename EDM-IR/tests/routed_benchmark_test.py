from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from edmir.benchmark.routed_reference import (
    bf16, canonicalize, compare_outputs, execute_reference, make_fixture, toy_trace,
)
from edmir.benchmark.measurement import (
    MEASURED, block_durations, compare_batches, digest, require_paper_evidence,
    request_split, summarize, synthetic_batch, validate_batch, validate_isolation,
)
from edmir.compiler.routed_contract import RoutedTrace, build_routed_traffic_ledger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from export_routed_measurements import export_payload
from run_tlc import run_case


class RoutedReferenceTest(unittest.TestCase):
    def test_independent_scalar_reference_all_routes(self):
        # Independent expert/activation indexing and math.fsum reference; not
        # just two calls through the same schedule implementation.
        f = make_fixture(toy_trace(4, 'balanced'), 5)
        result = execute_reference(f, split_k=4, group_n=2)
        for out_index, route_index in enumerate(result.output_to_input):
            r = f.trace.routes[route_index]
            for n in range(f.trace.n):
                ref = math.fsum(f.activations[r.global_token_id][k]*f.weights[r.expert_id][k][n]
                                for k in range(f.trace.k))
                self.assertAlmostEqual(result.values[out_index][n], ref, places=5)

    def test_schedule_layout_matrix(self):
        # 3 worlds x 3 patterns x 3 seeds x 4 split counts x 2 N groups.
        for p in (2, 4, 8):
            for pattern in ('balanced', 'hotset', 'empty'):
                trace = toy_trace(p, pattern)
                for seed in (0, 1, 2):
                    fixture = make_fixture(trace, seed)
                    expected = execute_reference(fixture)
                    for q in (1, 2, 4, 7):
                        for g in (1, 2):
                            with self.subTest(p=p, pattern=pattern, seed=seed, q=q, g=g):
                                actual = execute_reference(fixture, split_k=q, group_n=g,
                                                           private_order=reversed(range(len(trace.routes))))
                                self.assertTrue(compare_outputs(expected, actual)['passed'])
                                for i, out in enumerate(actual.input_to_output):
                                    self.assertEqual(actual.output_to_input[out], i)

    def test_experts_have_distinct_weights_and_gates_are_metadata(self):
        f = make_fixture(toy_trace(2, 'hotset'), 1)
        out = execute_reference(f)
        token = f.trace.routes[0].global_token_id
        matching = [i for i, r in enumerate(f.trace.routes) if r.global_token_id == token]
        values = [out.values[out.input_to_output[i]] for i in matching]
        self.assertNotEqual(values[0], values[1])
        first = f.trace.routes[0]
        expected = math.fsum(a*w[0] for a, w in zip(f.activations[token], f.weights[first.expert_id]))
        self.assertAlmostEqual(out.values[out.input_to_output[0]][0], expected, places=5)
        self.assertTrue(any(start == end for start, end in out.expert_spans))

    def test_missing_duplicate_and_nonfinite_outputs_rejected(self):
        t = toy_trace(2, 'balanced')
        out = execute_reference(make_fixture(t, 0))
        for ids, values in [(out.output_to_input[:-1], out.values[:-1]),
                            ((out.output_to_input[0],)*len(t.routes), out.values),
                            (out.output_to_input, ((float('nan'),)*t.n,)*len(t.routes))]:
            with self.assertRaises(ValueError):
                canonicalize(t, ids, values)
        corrupt = replace(out, gate_weights=(1.,)*len(t.routes))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            compare_outputs(out, corrupt)
        wrong = replace(out, values=tuple(tuple(v+1 for v in row) for row in out.values))
        self.assertFalse(compare_outputs(out, wrong)['passed'])

    def test_zero_reference_and_nonfinite_inputs(self):
        f = make_fixture(toy_trace(2, 'balanced'), 0)
        f = replace(f, activations={k:(0.,)*f.trace.k for k in f.activations})
        out = execute_reference(f)
        self.assertTrue(compare_outputs(out, out)['passed'])
        with self.assertRaises(ValueError):
            bf16(float('inf'))

    def test_bf16_ties_even(self):
        self.assertEqual(bf16(1 + 1/256), 1.)
        self.assertEqual(bf16(1 + 3/256), 1 + 2/128)

    def test_route_identity_mutations(self):
        original = toy_trace(2, 'balanced').to_dict()
        for key, value in [('world_size', 2.5), ('k', True), ('n', '9')]:
            d = deepcopy(original); d[key] = value
            with self.assertRaises(ValueError):
                RoutedTrace.from_dict(d)
        for key, value in [('global_token_id', False), ('route_slot', 99)]:
            d = deepcopy(original); d['routes'][0][key] = value
            with self.assertRaises(ValueError):
                RoutedTrace.from_dict(d)
        d = deepcopy(original)
        d['routes'][1]['source_rank'] = 1
        d['routes'].sort(key=lambda r:(r['source_rank'],r['global_token_id'],r['route_slot']))
        with self.assertRaisesRegex(ValueError, 'exactly one source'):
            RoutedTrace.from_dict(d)

    def test_noncontiguous_expert_placement(self):
        t = toy_trace(2, 'balanced')
        placement = (1, 0, 1, 0)
        t = replace(t, expert_placement=placement,
                    routes=tuple(replace(r, destination_rank=placement[r.expert_id]) for r in t.routes))
        out = execute_reference(make_fixture(t, 0))
        for e, (start, end) in enumerate(out.expert_spans):
            self.assertTrue(all(t.routes[i].expert_id == e for i in out.output_to_input[start:end]))

    def test_split_does_not_reduce_ledger_bytes(self):
        t = toy_trace(4, 'balanced')
        a = build_routed_traffic_ledger(t, split_k_slices=1)
        b = build_routed_traffic_ledger(t, split_k_slices=4)
        self.assertEqual(a.replayed_remote_rx_bytes, b.replayed_remote_rx_bytes)


class RoutedMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.host = synthetic_batch(4, 'host', trace_hash=digest('toy'))
        self.candidate = synthetic_batch(4, 'candidate', trace_hash=digest('toy'), scale=.8)

    def test_slowest_rank_not_average(self):
        self.host['blocks'][0]['rank_ms'][0] = [1., 2., 3., 9.]
        self.assertEqual(block_durations(self.host)[0][0], 9.)
        self.assertEqual(summarize(self.host)['samples'], 200)

    def test_paired_block_statistics_and_repeat(self):
        result = compare_batches(self.host, self.candidate, resamples=100)
        self.assertAlmostEqual(result['speedup_over_comparator'], 1.25)
        self.assertTrue(all(abs(x+.2) < 1e-10 for x in result['relative_latency_ci']))
        self.assertFalse(result['needs_repeat'])
        self.assertIsNone(result['deployment_decision'])
        tie = deepcopy(self.host); tie['family'] = 'tie'
        self.assertTrue(compare_batches(self.host, tie, resamples=100)['needs_repeat'])

    def test_trimming_invalid_block_and_nan_rejected(self):
        variants = [deepcopy(self.host) for _ in range(5)]
        variants[0]['blocks'][0]['rank_ms'].pop()
        variants[1]['blocks'][0]['status'] = 'invalid'
        variants[2]['blocks'][0]['rank_ms'][0][0] = float('nan')
        variants[3]['blocks'][0]['rank_ms'][0].pop()
        variants[4]['blocks'][0]['block_id'] = False
        for variant in variants:
            with self.assertRaises(ValueError):
                validate_batch(variant)

    def test_pair_binding_required(self):
        for key in ('environment_hash', 'source_hash', 'trace_hash', 'protocol_hash', 'pairing_id'):
            other = deepcopy(self.candidate); other[key] = digest('different')
            with self.assertRaisesRegex(ValueError, 'binding'):
                compare_batches(self.host, other, resamples=100)

    def test_paper_guard_and_missing_values(self):
        campaign = {'status':'sealed', 'unresolved':[],
                    'protocol_hash':self.host['protocol_hash'],
                    'trace_hashes':[self.host['trace_hash']],
                    'environment_hashes':[self.host['environment_hash']],
                    'source_hashes':[self.host['source_hash']]}
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            require_paper_evidence([self.host], campaign=campaign)
        with self.assertRaisesRegex(ValueError, 'missing measurements'):
            require_paper_evidence([], campaign=campaign)
        # Schema-only mock of measured provenance; no real GPU measurement here.
        measured = dict(self.host, evidence_kind=MEASURED, timer='cuda_event', phase='decode')
        for block in measured['blocks']:
            block['candidate_order'] = ['host', 'candidate']
        require_paper_evidence([measured], campaign=campaign)
        campaign['source_hashes'] = []
        with self.assertRaisesRegex(ValueError, 'backend build'):
            require_paper_evidence([measured], campaign=campaign)

    def test_parent_request_split_isolation(self):
        salt = 'toy-only'
        records = [{'record_id':str(p), 'parent_request_id':'parent-1',
                    'split':request_split('parent-1',salt)} for p in (2,4,8)]
        validate_isolation(records, salt)
        records[2]['split'] = 'evaluation' if records[0]['split'] != 'evaluation' else 'development'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_isolation(records, salt)

    def test_supported_regression_and_uncertain_interval(self):
        slow = synthetic_batch(4, 'slow', trace_hash=self.host['trace_hash'], scale=1.1)
        result = compare_batches(self.host, slow, resamples=500)
        self.assertTrue(result['statistically_supported_host_regression_over_1pct'])
        variable = deepcopy(self.candidate)
        for i, block in enumerate(variable['blocks']):
            scale = .7 if i % 2 else 1.3
            block['rank_ms'] = [[t*scale for t in row] for row in self.host['blocks'][i]['rank_ms']]
        result = compare_batches(self.host, variable, resamples=500)
        self.assertLess(result['relative_latency_ci'][0], 0)
        self.assertGreater(result['relative_latency_ci'][1], 0)
        self.assertTrue(result['needs_repeat'])

    def test_export_fails_before_creating_output(self):
        import json
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'raw.json').write_text(json.dumps([self.host]))
            (root/'campaign.json').write_text(json.dumps({'status':'draft_needs_assets'}))
            result = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/
                                     'tools/export_routed_measurements.py'),
                                     '--input',str(root/'raw.json'),'--campaign',str(root/'campaign.json'),
                                     '--out-dir',str(root/'export')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'export').exists())

    def test_export_duplicate_identity_rejected(self):
        b = deepcopy(self.host)
        b.update(evidence_kind=MEASURED,timer='cuda_event',phase='prefill')
        for block in b['blocks']:
            block['candidate_order']=['host','candidate']
        campaign = {'status':'sealed','unresolved':[],'protocol_hash':b['protocol_hash'],
                    'trace_hashes':[b['trace_hash']],'source_hashes':[b['source_hash']],
                    'environment_hashes':[b['environment_hash']]}
        self.assertEqual(len(export_payload([b],campaign)['rows']),1)
        with self.assertRaisesRegex(ValueError, 'duplicate timing identity'):
            export_payload([b,b],campaign)


class TLCRunnerGuardTest(unittest.TestCase):
    def test_wrong_invariant_or_plain_failure_is_not_expected_counterexample(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = root/'toy.cfg'; config.write_text('SPECIFICATION Spec\n')
            for output, expected in [
                ('ExpectedInvariant mentioned, but Invariant Other is violated.\n'
                 'The behavior up to this point is:',False),
                ('Invariant ExpectedInvariant is violated.',False),
                ('Invariant ExpectedInvariant is violated.\nThe behavior up to this point is:',True)]:
                with patch('run_tlc.subprocess.run',return_value=subprocess.CompletedProcess([],12,output,'')):
                    result = run_case(java='unused',jar=root/'unused.jar',config=config,
                                      expected_violation='ExpectedInvariant',raw_output=root/'raw.txt',workers=1)
                self.assertEqual(result['expectation_met'],expected)
                self.assertEqual(len(result['raw_output_sha256']),64)


if __name__ == '__main__':
    unittest.main()
