import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_window as w


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.bundle = w.ROOT.parent / '2026-11-13-w1-local-convergence/gpu-bundle-v2'
        self.plan = json.loads((w.ROOT / 'queue.json').read_text())
        self.source = json.loads((self.bundle / 'plan.json').read_text())
        self.guard = dict(authorization_confirmed=True, platform_verified=True, instance_id='TEST_ONLY',
                          billed_from_epoch=1000, hourly_total_usd=12, budget_usd=12,
                          platform_stop_epoch=4600)

    def test_fixed_complete_job_coverage(self):
        w.validate_plan(self.plan, self.source, self.bundle)
        self.assertEqual(sum(len(s['jobs']) for s in self.plan['stages']), 24)
        self.assertEqual(sum(len(s['traces']) for s in self.plan['stages']), 7)

    def test_changed_jobs_or_trace_order_rejected(self):
        for field in ('jobs', 'traces'):
            p = copy.deepcopy(self.plan)
            p['stages'][0][field] = []
            with self.assertRaises(ValueError): w.validate_plan(p, self.source, self.bundle)

    def test_no_current_authorization_no_execution(self):
        for field in ('authorization_confirmed', 'platform_verified'):
            g = dict(self.guard, **{field: False})
            with self.assertRaises(ValueError): w.cutoff_from_guard(g, 1100)

    def test_cost_and_runtime_limits(self):
        self.assertEqual(w.cutoff_from_guard(self.guard, 1100), 4000)
        for changes in ({'budget_usd': 11.99}, {'platform_stop_epoch': 4601},
                        {'budget_usd': 15.01},
                        {'hourly_total_usd': float('nan')}, {'billed_from_epoch': 1200}):
            with self.assertRaises(ValueError): w.cutoff_from_guard(dict(self.guard, **changes), 1100)
        with self.assertRaises(TimeoutError): w.cutoff_from_guard(self.guard, 4000)

    def test_timing_precedes_trace_failure_preserved(self):
        calls = []
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), inputs=Path(tmp), build_dir=Path(tmp))
            context = {'runtime': {}, 'window_identity': {'test_only': True}}
            def collect(a, p, r, c, x):
                a.out.mkdir(parents=True, exist_ok=True)
                calls.append(a.action)
            def trace(a, p, c, x):
                calls.append(a.trace_case)
                dest = a.out / 'diagnostic' / a.trace_case
                dest.mkdir(parents=True)
                (dest / 'stdout').write_text('TEST_ONLY incomplete profiler output')
                raise RuntimeError('TEST_ONLY unavailable profiler')
            def save(p, value): p.write_text(json.dumps(value))
            stub = SimpleNamespace(collect=collect, trace=trace, save=save,
                                   preflight=lambda *a: {'runtime': {}})
            stage = self.plan['stages'][0]
            with patch('run_window.time.time', return_value=1100):
                status = w.run_stage(stub, args, self.source, {}, stage,
                                     SimpleNamespace(end=4000), context)
                again = w.run_stage(stub, args, self.source, {}, stage,
                                    SimpleNamespace(end=4000), context)
            self.assertEqual(calls[:3], ['pilot', 'collect', 'm512-fp32-q8-g2'])
            self.assertEqual(calls.count('m512-fp32-q8-g2'), 1)
            self.assertEqual(status['m512-fp32-q8-g2'], 'failed_or_candidate_not_admitted')
            self.assertEqual(again['m512-fp32-q8-g2'], 'failed_preserved_no_overwrite')
            self.assertTrue((Path(tmp)/'m512/m512-fp32-q8-g2-failure.json').is_file())

    def test_pre_capture_rejection_is_not_retried_or_overwritten(self):
        calls = []
        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), inputs=Path(tmp), build_dir=Path(tmp))
            context = {'runtime': {}, 'window_identity': {'test_only': True}}
            def collect(a, *rest): a.out.mkdir(parents=True, exist_ok=True)
            def reject(a, *rest):
                calls.append(a.trace_case)
                raise ValueError('TEST_ONLY candidate not admitted')
            stub = SimpleNamespace(collect=collect, trace=reject,
                                   save=lambda p, v: p.write_text(json.dumps(v)),
                                   preflight=lambda *a: {'runtime': {}})
            stage = self.plan['stages'][0]
            with patch('run_window.time.time', return_value=1100):
                w.run_stage(stub, args, self.source, {}, stage, SimpleNamespace(end=4000), context)
                failure = Path(tmp)/'m512/m512-fp32-q8-g2-failure.json'
                original = failure.read_bytes()
                again = w.run_stage(stub, args, self.source, {}, stage, SimpleNamespace(end=4000), context)
            self.assertEqual(len(calls), 1)
            self.assertEqual(failure.read_bytes(), original)
            self.assertEqual(again['m512-fp32-q8-g2'], 'failed_preserved_no_overwrite')


if __name__ == '__main__': unittest.main()
