"""Contracts that prevent false timing labels and confounded next-run designs."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"EDM-IR/tools"))
import analyze_precision_window as analysis
import run_precision_followup as followup
import run_native_moe_smoke as native


class AnalysisTests(unittest.TestCase):
    def test_union_is_not_sum_of_overlapping_kernels(self):
        self.assertEqual(analysis.union_length([(0,10),(3,8),(8,15),(20,25)]),20)
        self.assertEqual(analysis.union_length([]),0)
        with self.assertRaises(ValueError):analysis.union_length([(2,1)])

    def test_mixed_directions_are_retained_as_unstable(self):
        self.assertFalse(analysis.pair_directions([1,3,1],[2,2,2])["consistent"])
        self.assertTrue(analysis.pair_directions([1,1,1],[2,2,2])["consistent"])

    def test_modified_file_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/"raw";p.write_text("original")
            manifest={"raw":dict(bytes=p.stat().st_size,sha256=analysis.sha(p))}
            self.assertEqual(analysis.checked_file(root,manifest,"raw"),p)
            p.write_text("modified")
            with self.assertRaisesRegex(ValueError,"mismatch"):analysis.checked_file(root,manifest,"raw")
            with self.assertRaisesRegex(ValueError,"escapes"):analysis.checked_file(root,manifest,"../other")

    def test_prediction_query_excludes_all_observed_costs(self):
        row={k:1 for k in (*analysis.model.CONTRACT_FIELDS,"checks","features","sm_count")}
        row.update(latency_us=999,repeat_medians_us=[999],measurement_records=["label"])
        self.assertNotIn("latency_us",analysis.query(row))
        self.assertNotIn("repeat_medians_us",analysis.query(row))
        self.assertNotIn("measurement_records",analysis.query(row))


class FollowupTests(unittest.TestCase):
    def test_full_cross_and_adjacent_balance(self):
        p=followup.plan();followup.validate_plan(p)
        self.assertEqual(len(p["jobs"]),21)
        self.assertEqual(Counter(j['seed'] for j in p['jobs'] if j['phase']=='measure'),
                         Counter({seed:6 for seed in followup.SEEDS}))

    def test_changed_order_or_added_shape_refused(self):
        p=followup.plan();p['jobs'][3]['order'].reverse()
        with self.assertRaises(ValueError):followup.validate_plan(p)
        p=followup.plan();p['jobs'][3]['shape']='unprescribed'
        with self.assertRaises(ValueError):followup.validate_plan(p)

    def test_guard_requires_fresh_scope_budget_and_recovery_time(self):
        g=dict(authorization_confirmed=True,platform_verified=True,instance_id='test-only',
               experiment_schema=followup.SCHEMA,billed_from_epoch=1000,hourly_total_usd=10.8,
               budget_usd=15,platform_stop_epoch=5000)
        self.assertEqual(followup.cutoff_from_guard(g,2000),4400)
        # Removing the one-hour task limit must not remove the spending bound.
        long_window=dict(g,platform_stop_epoch=6000)
        self.assertEqual(followup.cutoff_from_guard(long_window,1000),5400)
        from types import SimpleNamespace
        class LegacyDeadline:
            def __init__(self,end):
                raise AssertionError('obsolete one-hour constructor invoked')
            def remaining(self,now):return self.end-now
        clock=followup.collection_clock(SimpleNamespace(Deadline=LegacyDeadline),long_window,1000)
        self.assertEqual(clock.remaining(1000),4400)
        with self.assertRaises(TimeoutError):
            followup.collection_clock(SimpleNamespace(Deadline=LegacyDeadline),long_window,5500)
        for key,value in [('authorization_confirmed',False),('experiment_schema','old'),
                          ('budget_usd',16),('hourly_total_usd',100),('instance_id','')]:
            bad=dict(g);bad[key]=value
            with self.assertRaises(ValueError):followup.cutoff_from_guard(bad,2000)
        with self.assertRaises(TimeoutError):followup.cutoff_from_guard(g,4500)


class NativePreparationTests(unittest.TestCase):
    def test_model_heads_do_not_allow_tp8(self):
        config=dict(num_attention_heads=20,n_routed_experts=64)
        self.assertEqual(native.validate_parallelism(config)['tensor_parallel_size'],4)
        with self.assertRaises(ValueError):native.validate_parallelism(config,tp=8,dp=1)

    def test_missing_assets_never_admit_gpu(self):
        manifest=dict(id=native.MODEL_ID,revision=native.REVISION,weight_file_bytes=5,
            weight_files=[dict(rfilename='model-00001.safetensors',size=5,lfs={'sha256':'x'})],
            auxiliary_hashes={'tokenizer.json':'x','tokenizer_config.json':'x'},
            **{'config.json_sha256':'x','model.safetensors.index.json_sha256':'x'})
        with tempfile.TemporaryDirectory() as d:
            result=native.inspect_assets(manifest,Path(d))
            self.assertFalse(result['ASSETS_VERIFIED'])
            self.assertFalse(result['GPU_runtime_verified'])
            self.assertFalse(result['weights_present'])

    def test_routed_token_length_ids_and_duplicates(self):
        import numpy as np
        a=np.tile(np.arange(4),(5,47,1));a[:,0,:]=0
        self.assertEqual(native.check_routes(a,4,2)['routed_assignments'],5*46*4)
        with self.assertRaises(ValueError):native.check_routes(a,4,3)
        with self.assertRaises(ValueError):native.check_routes(a[:,1:,:],4,2)
        b=a.copy();b[0,0,0]=64
        with self.assertRaises(ValueError):native.check_routes(b,4,2)
        b=a.copy();b[0,1,0]=1
        with self.assertRaises(ValueError):native.check_routes(b,4,2)
        b=a.copy();b[0,1,0]=-1
        with self.assertRaises(ValueError):native.check_routes(b,4,2)


if __name__=='__main__':unittest.main()
