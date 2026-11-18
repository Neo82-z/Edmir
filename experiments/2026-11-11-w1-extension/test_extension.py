import copy
import json
from pathlib import Path
import unittest
import tempfile
import time
from types import SimpleNamespace

from run_extension import deadline_from_guard, validate_plan, execute_stage

ROOT = Path(__file__).resolve().parent
BUNDLE = Path('/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01/bundle')


class ExtensionTest(unittest.TestCase):
    def test_plan_covers_only_original_jobs(self):
        plan = json.loads((ROOT / 'plan.json').read_text())
        source = json.loads((BUNDLE / 'plan.json').read_text())
        validate_plan(plan, source, BUNDLE)
        changed = copy.deepcopy(plan)
        changed['stages'][0]['jobs'][0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'job differs'):
            validate_plan(changed, source, BUNDLE)
        changed = copy.deepcopy(plan)
        changed['stages'][0]['jobs'].pop()
        changed['stages'][0]['original_job_indices'].pop()
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            validate_plan(changed, source, BUNDLE)

    def test_fee_expiry_and_unverified_guards(self):
        guard = dict(platform_verified=True, instance_id='test-only-no-instance',
                     active_epoch=1000, hourly_usd=10.8, auto_delete_epoch=4300,
                     spend_cap_usd=10.8)
        self.assertEqual(deadline_from_guard(guard, 1100), 4000)
        for change in ({'platform_verified': False}, {'hourly_usd': 10.81},
                       {'auto_delete_epoch': 4601}, {'spend_cap_usd': 11},
                       {'spend_cap_usd': 1}, {'hourly_usd': float('nan')}):
            with self.assertRaises(ValueError):
                deadline_from_guard({**guard, **change}, 1100)
        with self.assertRaises(TimeoutError):
            deadline_from_guard(guard, 4001)
        with self.assertRaises(ValueError):
            deadline_from_guard(guard, 4301)

    def test_failed_trace_is_not_counted_as_capture_or_retried(self):
        calls = []
        base_context = {'test_fixture': True}
        context = {**base_context, 'extension_identity': {'test_only': True}}
        class Faults:
            def collect(self, args, plan, records, clock, ctx):
                args.out.mkdir(parents=True, exist_ok=True)
                calls.append(args.action)
            def trace(self, args, plan, clock, ctx):
                (args.out / 'diagnostic' / args.trace_case).mkdir(parents=True)
                calls.append(args.trace_case)
                raise RuntimeError('injected profiler failure; no synthetic timing data')
            def preflight(self, args, plan, clock):
                return base_context
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), inputs=Path(tmp), build_dir=Path(tmp))
            stage = {'name': 'grouped', 'jobs': [], 'traces': ['grouped-g1']}
            clock = SimpleNamespace(end=time.time()+600)
            first = execute_stage(Faults(), args, {}, {}, stage, clock, context)
            second = execute_stage(Faults(), args, {}, {}, stage, clock, context)
            self.assertEqual(first, {'grouped-g1': 'failed'})
            self.assertEqual(second, {'grouped-g1': 'failed_preserved_no_overwrite'})
            self.assertEqual(calls.count('grouped-g1'), 1)
            self.assertTrue((Path(tmp) / 'grouped/grouped-g1-failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
