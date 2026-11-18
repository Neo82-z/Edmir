import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from run_w1_gpu import Deadline, collect, require_successful_pilot
from unittest.mock import patch


class TestClock:
    """Collector fault injection only. Rows never leave the test directory."""
    def __init__(self):
        self.end=time.time()+30
        self.calls=[]
        self.interrupt_measure=True

    def run(self,args,limit,stdout,stderr):
        mode=args[1]
        self.calls.append((mode,args[5:]))
        for family in args[5:]:
            row=dict(family=family,input_seed=int(args[4]),checked_elements=64)
            if mode=="--check":
                row.update(correctness_passed=family=="host",check_only=True)
                if family!="host":row.update(failure="injected numerical failure",median_ms=None)
            else:
                row.update(correctness_passed=True,shape=[8,1,1,1],runtime={},
                    implementation_id="host_nccl_cutlass_sm80_per_source_v1_prepared_metadata_v2",
                    metadata_mode="prepared",dependency_policy="resource_queues_v1",
                    measurement_policy=dict(profiler=False,nvtx=False,discarded_samples=0),
                    nonfinite_outputs=0,warmup=20,iterations=100,samples_ms=[1.0]*100,median_ms=1.0)
            print(json.dumps(row),file=stdout)
            if mode=="--measure" and self.interrupt_measure:
                self.interrupt_measure=False
                raise TimeoutError("injected interruption")


class CollectionTest(unittest.TestCase):
    @patch("run_w1_gpu.require_successful_pilot")
    def test_failure_exclusion_interruption_and_resume(self, pilot_gate):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            args=SimpleNamespace(action="collect",out=root,inputs=root,build_dir=root)
            clock=TestClock()
            job=dict(shape="dev",order=["host","splitk16"],seed=31,repetition=0)
            plan=dict(shapes={"dev":[8,1,1,1]},jobs=[dict(job,phase="check"),dict(job,phase="measure")])
            records={"dev":{"31":{"test_only":True}}}
            context={"runtime":{},"synthetic_fixture":True}
            with self.assertRaises(TimeoutError):collect(args,plan,records,clock,context)
            self.assertFalse((root/"collect-001.json").exists())
            collect(args,plan,records,clock,context)
            self.assertEqual(clock.calls,[("--check",job["order"]),("--measure",["host"]),("--measure",["host"])])
            self.assertEqual(json.loads((root/"collect-001.json").read_text())["excluded_by_numerical_check"],["splitk16"])
            self.assertEqual(len(list(root.glob("collect-001-*.stdout"))),2)
            with self.assertRaisesRegex(ValueError,"identical binary"):
                collect(args,plan,records,clock,{"other_gpu":True})
            receipt=json.loads((root/"collect-000.json").read_text())
            (root/receipt["stdout"]).write_text("tampered")
            with self.assertRaisesRegex(ValueError,"raw output"):
                collect(args,plan,records,clock,context)

    def test_missing_pilot_blocks_collection(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError,"two-candidate pilot"):
            require_successful_pilot(Path(tmp),"unused",{},{})

    def test_expired_and_unbounded_deadlines(self):
        for end in (time.time()-1,time.time()+7200,float("nan")):
            with self.assertRaises(ValueError):Deadline(end)

if __name__=="__main__":unittest.main()
