from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import run_selector_gpu as runner


@unittest.skipUnless(os.name == "posix", "GPU collector targets POSIX servers")
class SelectorRunnerSafetyTest(unittest.TestCase):
    def test_timeout_preserves_output_and_kills_descendant(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp)/"child_survived"
            # The child inherits output handles, ignores SIGTERM and outlives
            # the parent timeout unless the entire process group is killed.
            child = ("import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                     "print('child-ready',flush=True); time.sleep(1.5); "
                     f"pathlib.Path({str(marker)!r}).write_text('alive')")
            parent = ("import subprocess,sys,time; "
                      f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                      "print('parent-ready',flush=True); print('diagnostic',file=sys.stderr,flush=True); "
                      "time.sleep(30)")
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired) as ctx:
                runner.command([sys.executable, "-c", parent], capture_output=True, timeout=.6)
            self.assertLess(time.monotonic()-start, 5)
            self.assertIn("child-ready", ctx.exception.output)
            self.assertIn("parent-ready", ctx.exception.output)
            self.assertIn("diagnostic", ctx.exception.stderr)
            time.sleep(1.1)
            self.assertFalse(marker.exists(), "descendant survived collector timeout")

    def test_job_failure_keeps_logs_and_does_not_create_measurement(self):
        for source, exception in (
            ("import sys; print('partial',flush=True); print('error',file=sys.stderr); sys.exit(7)",
             subprocess.CalledProcessError),
            ("import time; print('partial',flush=True); time.sleep(30)", subprocess.TimeoutExpired),
        ):
            with self.subTest(exception=exception), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                with self.assertRaises(exception):
                    runner.execute_job([sys.executable, "-c", source], {"family": "test"}, out, 3, timeout=.3)
                self.assertIn("partial", (out/"stdout_00003.txt").read_text())
                self.assertEqual(json.loads((out/"failure.json").read_text())["job_index"], 3)
                self.assertEqual(json.loads((out/"progress.json").read_text())["state"], "failed")
                self.assertFalse((out/"measurements.jsonl").exists())

    def test_collect_stops_at_failure_and_resume_only_launches_unfinished_job(self):
        # Exercise actual collector bookkeeping and real local subprocesses.
        # GPU inventory, input checks and numerical validation have their own
        # tests; these doubles do not stand in for GPU correctness evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, build, out = root/"package", root/"build", root/"out"
            package.mkdir()
            build.mkdir()
            data = package/"data/w000"
            data.mkdir(parents=True)
            for name in ("inputs.bin", "reference.bin"):
                (data/name).write_bytes(b"local-test")
            runtime = {"test": "CPU bookkeeping only"}
            jobs = [{"family": family, "executable": family, "shape": [4, 64, 1024, 512],
                     "workload_id": "w000", "repetition": 0, "trial_order": i}
                    for i, family in enumerate(("host", "direct"))]
            p = {"mode": "pilot_development", "programs": {}, "jobs": jobs, "warmup": 20,
                 "iterations": 100, "plan_sha256": None, "timing_scope": "test",
                 "baseline_contracts": {}, "collector_sha256": "test",
                 "workloads": [{"id": "w000", "shape": jobs[0]["shape"], "trial_orders": [["host", "direct"]]}]}
            runner.write_json(build/"selector_build.json", {
                "package_sha256": runner.digest(p), "dependencies": {}, "binaries": {}, "environment": {}})
            hashes = {"input_sha256": runner.sha256(data/"inputs.bin"),
                      "reference_sha256": runner.sha256(data/"reference.bin")}
            runner.write_json(package/"inputs.json", {"package_sha256": runner.digest(p),
                                                      "workloads": {"w000": hashes}})

            def executable(family, hang=False):
                row = {**jobs[0 if family == "host" else 1], "runtime": runtime, "median_ms": 1.}
                source = ("import time; print('partial',flush=True); time.sleep(30)" if hang
                          else f"print({json.dumps(row)!r})")
                path = build/family
                path.write_text(f"#!{sys.executable}\n{source}\n")
                path.chmod(0o755)

            executable("host")
            executable("direct", hang=True)
            original_command, original_execute = runner.command, runner.execute_job

            def local_command(args, **kwargs):
                if str(args[0]) == "nvidia-smi":
                    return subprocess.CompletedProcess(args, 0, "local-test", "")
                if "--inspect" in args:
                    return subprocess.CompletedProcess(args, 0, json.dumps(runtime), "")
                return original_command(args, **kwargs)

            def local_job(args, job, path, index):
                return original_execute(args, job, path, index, timeout=2)

            with patch.dict(runner.BASELINES, {"host": runner.BASELINES["host"]}, clear=True), (
                patch.object(runner, "verify_package", return_value=p)
            ), (
                patch.object(runner, "environment", return_value={})
            ), patch.object(runner, "validate_measurement", side_effect=lambda row, *args: row), (
                patch.object(runner, "command", side_effect=local_command)
            ), patch.object(runner, "execute_job", side_effect=local_job) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.collect(package, build, out)
                self.assertEqual(launch.call_count, 2)
                self.assertEqual(len((out/"measurements.jsonl").read_text().splitlines()), 1)
                executable("direct")
                launch.reset_mock()
                runner.collect(package, build, out, resume=True)
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[1]["family"], "direct")
                self.assertEqual(len((out/"measurements.jsonl").read_text().splitlines()), 2)
                self.assertEqual(json.loads((out/"progress.json").read_text())["state"], "complete")


if __name__ == "__main__":
    unittest.main()
