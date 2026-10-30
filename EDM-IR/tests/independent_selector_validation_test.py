from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from edmir.compiler.measurement_dataset import sha256
from edmir.compiler.two_stage_selection import parse_schedule_family
from independent_selector_validation import (
    ALL_FAMILIES, BASELINES, digest, freeze, read_run, score, verify_plan,
)

REPO = Path(__file__).resolve().parents[2]


class IndependentValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = freeze(REPO/"experiments/2026-10-20-offline-predictor/data", 4)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # A small synthetic fixture for validator/scorer tests, never evidence.
        self.envelope = copy.deepcopy(self.frozen)
        self.envelope["plan"]["workloads"] = self.envelope["plan"]["workloads"][:2]
        self.envelope["plan"]["predeclared_analysis"]["bootstrap_resamples"] = 20
        self.envelope["plan_sha256"] = digest(self.envelope["plan"])
        plan = self.envelope["plan"]
        env = {**plan["hardware"], "world_size": 4, **{
            f: "synthetic-test" for f in ("driver_version", "cuda_version", "nccl_version",
                "nccl_commit", "cutlass_version", "cutlass_commit", "cublas_version")},
            "collector_sha256": "a"*64}
        self.run = {k: plan[k] for k in (
            "environment_id", "timing_scope", "latency_unit", "dtype", "baseline_contracts")}
        self.run.update(schema="edmir.independent-selector-run.v1",
                        plan_sha256=self.envelope["plan_sha256"], environment=env,
                        fresh_process_per_repetition=True, rows=[])
        for work in plan["workloads"]:
            for rep, order in enumerate(work["trial_orders"]):
                for position, family in enumerate(order):
                    p, m, _, n = work["shape"]
                    if family in BASELINES:
                        implementation = BASELINES[family].baseline_id
                    else:
                        q, g = parse_schedule_family(family)
                        implementation = f"direct_peer_cutlass_sm80_q{q}_g{g}"
                    self.run["rows"].append({
                        "shape": work["shape"], "family": family, "repetition": rep,
                        "trial_order": position, "median_ms": 1.0 if family == "host_merged_cublaslt" else 2.0,
                        "correctness_passed": True, "max_abs_error": 0., "max_rel_error": 0.,
                        "atol": 0.02, "rtol": 0.02, "warmup": 20, "iterations": 100,
                        "checked_elements": p*p*m*n, "nonfinite_outputs": 0,
                        "correctness_reference": plan["correctness"]["reference"],
                        "input_seed": 20261020, "input_sha256": "b"*64,
                        "implementation_id": implementation, "gemm_algorithm_id": "synthetic-test",
                    })
        self.path = self.directory/"run.json"
        self.write_run()

    def write_run(self):
        raw_path = self.directory/"raw.jsonl"
        raw_path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "source"})+"\n"
                                    for r in self.run["rows"]))
        for i, row in enumerate(self.run["rows"]):
            row["source"] = {"path": "raw.jsonl", "sha256": sha256(raw_path), "line": i+1}
        self.path.write_text(json.dumps(self.run))

    def test_freeze_is_deterministic_and_excludes_every_examined_MKN(self):
        repeat = freeze(REPO/"experiments/2026-10-20-offline-predictor/data", 4)
        self.assertEqual(self.frozen, repeat)
        plan = repeat["plan"]
        self.assertEqual(len(plan["workloads"]), 45)
        self.assertEqual(len(ALL_FAMILIES), 14)
        for work in plan["workloads"]:
            self.assertNotIn(work["shape"][1:], plan["examined_MKN"])
            self.assertEqual(set(work["choices"]), {"host", "train_best_fixed", "relative",
                                                  "relative_supported", "factorized_supported"})
            for order in work["trial_orders"]:
                self.assertEqual(set(order), set(ALL_FAMILIES))
            if work["shape"][1] != 64:
                self.assertFalse(work["factorized_details"]["grouped_eligible"])
        p8 = freeze(REPO/"experiments/2026-10-20-offline-predictor/data", 8)["plan"]
        for method in ("factorized", "relative"):
            self.assertIsNone(p8["models"][method]["calibration_environment_id"])
            self.assertEqual(p8["models"][method]["calibration_offset"], 0)

    def test_complete_synthetic_run_scores_strong_baseline_without_refitting(self):
        # Frozen choices have latency 2; the new strong host has latency 1.
        # Any scorer silently dropping new hosts or recomputing choices fails.
        with patch("independent_selector_validation.fit_models", side_effect=AssertionError("must not refit")):
            result = score(self.envelope, self.path)
        self.assertEqual(result["timing_rows"], 2*5*14)
        self.assertFalse(result["eligible_for_compiler_admission"])
        for entry in result["metrics"].values():
            self.assertEqual(entry["mean_regret_all_candidates"], 1.0)
            self.assertEqual(entry["mean_ratio_to_strong_host"], 2.0)
            self.assertEqual(entry["strong_host_regressions"], 2)
            self.assertEqual(entry["legacy_host_regressions"], 0)
            self.assertEqual(entry["bootstrap_95pct"]["mean_regret_all_candidates"], [1., 1.])

    def test_missing_duplicate_and_wrong_order_are_rejected(self):
        original = copy.deepcopy(self.run)
        for case in ("missing", "duplicate", "order"):
            with self.subTest(case=case):
                self.run = copy.deepcopy(original)
                if case == "missing":
                    self.run["rows"].pop()
                elif case == "duplicate":
                    self.run["rows"].append(copy.deepcopy(self.run["rows"][0]))
                else:
                    self.run["rows"][0]["trial_order"] += 1
                self.write_run()
                with self.assertRaises(ValueError):
                    read_run(self.envelope, self.path)

    def test_baseline_and_timer_mismatch_cannot_be_silently_pooled(self):
        original = copy.deepcopy(self.run)
        for field, value in (("dtype", "bfloat16"), ("latency_unit", "us"),
                             ("timing_scope", "CUDA events"), ("baseline_contracts", {}),
                             ("environment_id", "changed")):
            self.run = copy.deepcopy(original)
            self.run[field] = value
            self.write_run()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract mismatch"):
                read_run(self.envelope, self.path)

    def test_invalid_measurement_and_accuracy_provenance_are_rejected(self):
        original = copy.deepcopy(self.run)
        cases = (("median_ms", float("nan")), ("median_ms", -1),
                 ("correctness_passed", False), ("atol", 100.),
                 ("warmup", 1), ("iterations", 2), ("nonfinite_outputs", 1),
                 ("checked_elements", 1), ("input_seed", 0),
                 ("implementation_id", "cublas-or-cutlass"),
                 ("input_sha256", "c"*64))
        for field, value in cases:
            self.run = copy.deepcopy(original)
            self.run["rows"][0][field] = value
            self.write_run()
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_run(self.envelope, self.path)

    def test_source_frozen_predictions_and_raw_data_tampering_fail(self):
        envelope = copy.deepcopy(self.envelope)
        envelope["plan"]["workloads"][0]["choices"]["relative"] = "host"
        with self.assertRaisesRegex(ValueError, "hash"):
            verify_plan(envelope)
        envelope = copy.deepcopy(self.envelope)
        key = next(iter(envelope["plan"]["source_sha256"]))
        envelope["plan"]["source_sha256"][key] = "0"*64
        envelope["plan_sha256"] = digest(envelope["plan"])
        with self.assertRaisesRegex(ValueError, "source changed"):
            verify_plan(envelope)
        raw = self.directory/"raw.jsonl"
        raw.write_text(raw.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "raw measurement hash"):
            read_run(self.envelope, self.path)


if __name__ == "__main__":
    unittest.main()
