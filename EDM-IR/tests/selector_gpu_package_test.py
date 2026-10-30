from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from prepare_selector_gpu import (prepare, verify_package, write_inputs, matrix,
                                  implementation_id, LEGACY_CONTROLS, PILOT_MKN)
from run_selector_gpu import (environment, validate_measurement, measurement_environment,
                              summarize_metadata)
from independent_selector_validation import ALL_FAMILIES, BASELINES

try:
    import numpy as np
except ImportError:
    np = None


class SelectorGpuPackageTest(unittest.TestCase):
    def test_metadata_ablation_keeps_four_shapes_and_materializes_every_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"package"
            p = prepare(root, metadata_ablation=True)
            self.assertEqual(verify_package(root), p)
            self.assertEqual([tuple(w["shape"][1:]) for w in p["workloads"]], list(PILOT_MKN))
            self.assertEqual(len(p["programs"]), 48)
            self.assertEqual(len(p["jobs"]), 216)
            contracts = p["implementation_contracts"]
            self.assertEqual(sum(v["primary"] for v in contracts.values()), 14)
            self.assertEqual({v["family"] for v in contracts.values() if not v["primary"]},
                             set(LEGACY_CONTROLS))
            self.assertEqual(len({v["implementation_id"] for v in contracts.values()}), 18)
            for work in p["workloads"]:
                for order in work["trial_orders"]:
                    self.assertEqual(set(order), set(contracts))
            for job in p["jobs"]:
                spec = contracts[job["candidate_id"]]
                self.assertEqual(job["implementation_id"], spec["implementation_id"])
                self.assertEqual(job["metadata_mode"], spec["metadata_mode"])
                self.assertEqual(job["family"], spec["family"])
                self.assertEqual(job["implementation_id"],
                                 implementation_id(job["family"], job["metadata_mode"]))
            with self.assertRaises(ValueError):
                prepare(Path(temporary)/"invalid", plan_path=Path("unused"), metadata_ablation=True)
            for family, mode in (("host", "merged"), ("host_merged_cutlass", "prepared"), ("direct", "unknown")):
                with self.assertRaises(ValueError):
                    implementation_id(family, mode)

    def test_metadata_measurements_keep_tail_and_reject_old_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = prepare(Path(temporary)/"package", metadata_ablation=True)
            job = next(j for j in p["jobs"] if j["candidate_id"] == "host@prepared")
            n = job["shape"]
            inputs = {"input_sha256": "0"*64, "reference_sha256": "1"*64}
            row = {"family": job["family"], "shape": n, "metadata_mode": "prepared",
                   "implementation_id": job["implementation_id"], "gemm_algorithm_id": "kernel",
                   "correctness_passed": True, "nonfinite_outputs": 0, "checked_elements": n[0]**2*n[1]*n[3],
                   "max_abs_error": 0., "max_rel_error": 0., "atol": .02, "rtol": .02,
                   "correctness_reference": "CPU_FP32_on_identical_FP16_inputs", "input_seed": 20261020,
                   "warmup": 20, "iterations": 100, "samples_ms": [1.]*99+[101.],
                   "median_ms": 1., "mean_ms": 2., "total_ms": 200., "min_ms": 1., "max_ms": 101.,
                   "timing_scope": p["timing_scope"], "measurement_policy": p["measurement_policy"]}
            checked = validate_measurement(row, job, inputs, p)
            self.assertEqual(checked["samples_ms"][-1], 101.)
            self.assertEqual(checked["candidate_id"], "host@prepared")
            self.assertEqual(validate_measurement(checked, job, inputs, p), checked)
            for changes in (
                {"implementation_id": BASELINES["host"].baseline_id},
                {"metadata_mode": "legacy"}, {"mean_ms": 1.}, {"total_ms": math.inf},
                {"max_ms": 1.}, {"samples_ms": [1.]*100}, {"timing_scope": "cuda event"},
                {"measurement_policy": {**p["measurement_policy"], "discarded_samples": 1}},
                {"measurement_policy": {**p["measurement_policy"], "profiler": True}},
            ):
                with self.assertRaises(ValueError):
                    validate_measurement({**row, **changes}, job, inputs, p)

    def test_metadata_summary_pairs_processes_and_excludes_controls_from_primary_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = prepare(Path(temporary)/"package", metadata_ablation=True)
            rows = [{**j, "median_ms": (.1 if j["metadata_mode"] == "legacy" else 2.)+j["repetition"],
                     "mean_ms": (1. if j["metadata_mode"] == "legacy" else 4.)+j["repetition"]}
                    for j in p["jobs"]]
            result = summarize_metadata(p, list(reversed(rows)))
            for work in result["workloads"]:
                self.assertFalse(work["observed_best_primary"].endswith("@legacy"))
                self.assertEqual(work["candidates"]["host@prepared"]["process_medians_ms"], [2., 3., 4.])
                ratios = work["paired_legacy_controls"]["host"]["legacy_over_prepared_process_median_ratios"]
                self.assertEqual(ratios, [.1/2., 1.1/3., 2.1/4.])
            for invalid in (rows[:-1], rows[:-1]+[rows[0]]):
                with self.assertRaises(ValueError):
                    summarize_metadata(p, invalid)

    def test_timing_environment_rejects_known_observer_modes(self):
        self.assertEqual(measurement_environment({"CUDA_VISIBLE_DEVICES": "0,1,2,3",
                                                 "NCCL_DEBUG": "WARN", "UNRELATED_SECRET": "hidden"}),
                         {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "NCCL_DEBUG": "WARN"})
        for env in ({"CUDA_LAUNCH_BLOCKING": "1"}, {"CUDA_PROFILE": "1"},
                    {"CUDA_INJECTION64_PATH": "/example/profiler.so"},
                    {"LD_PRELOAD": "/example/libcupti.so"}, {"NCCL_DEBUG": "TRACE"}):
            with self.assertRaises(ValueError):
                measurement_environment(env)

    def test_complete_pilot_surface_and_distinct_host_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"package"
            p = prepare(root)
            self.assertEqual(verify_package(root), p)
            self.assertEqual(len(p["workloads"]), 4)
            self.assertEqual(len(p["jobs"]), 4*14*3)
            self.assertEqual(len(p["programs"]), 4*12)
            for work in p["workloads"]:
                for order in work["trial_orders"]:
                    self.assertEqual(set(order), set(ALL_FAMILIES))
            ids = [b["baseline_id"] for b in p["baseline_contracts"].values()]
            self.assertEqual(len(set(ids)), 4)
            for job in p["jobs"]:
                compiled = p["programs"][job["executable"]]["compiled_family"]
                if job["family"].startswith("host_merged"):
                    self.assertEqual(compiled, "host")
                else:
                    self.assertEqual(compiled, job["family"])
            (root/"generated_targets.cmake").write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_package(root)

    def test_runtime_topology_uses_physical_pci_identity(self):
        devices = [{"name": "NVIDIA A100-SXM4-80GB", "major": 8, "minor": 0,
                    "sm_count": 108, "pci_bus_id": f"0000:{i:02x}:00.0", "peers": [True]*4}
                   for i in (4, 2, 7, 5)]
        runtime = {"devices": devices, **{k: "test" for k in (
            "driver_version", "cuda_version", "nccl_version", "cublas_version")}}
        topo = "\t"+" ".join(f"GPU{i}" for i in range(8))+" CPU Affinity\n"
        topo += "\n".join(f"GPU{i} "+" ".join("X" if i == j else "NV12" for j in range(8))
                          for i in range(8))
        inventory = "\n".join(f"{i}, 00000000:{i:02x}:00.0" for i in range(8))
        self.assertEqual(environment(runtime, topo, inventory, 4)["world_size"], 4)
        with self.assertRaises(ValueError):
            environment(runtime, topo.replace("NV12", "PHB"), inventory, 4)
        bad = deepcopy(runtime)
        bad["devices"][0]["peers"][2] = False
        with self.assertRaises(ValueError):
            environment(bad, topo, inventory, 4)
        bad = deepcopy(runtime)
        bad["devices"][0]["sm_count"] = 80
        with self.assertRaises(ValueError):
            environment(bad, topo, inventory, 4)

    def test_collector_rejects_partial_checks_relabeling_and_false_medians(self):
        job = {"family": "host", "shape": [4, 64, 1024, 512], "repetition": 0, "trial_order": 0}
        config = {"warmup": 20, "iterations": 100}
        inputs = {"input_sha256": "0"*64, "reference_sha256": "1"*64}
        row = {"family": "host", "shape": job["shape"], "correctness_passed": True,
               "checked_elements": 4*4*64*512, "nonfinite_outputs": 0,
               "max_abs_error": 0., "max_rel_error": 0., "median_ms": 1.,
               "atol": 0.02, "rtol": 0.02, "input_seed": 20261020,
               "correctness_reference": "CPU_FP32_on_identical_FP16_inputs",
               "warmup": 20, "iterations": 100, "samples_ms": [1.]*100,
               "implementation_id": BASELINES["host"].baseline_id, "gemm_algorithm_id": "kernel"}
        self.assertEqual(validate_measurement(row, job, inputs, config)["input_sha256"], "0"*64)
        for change in (
            {"median_ms": 1.2}, {"max_rel_error": math.nan}, {"checked_elements": 1},
            {"implementation_id": BASELINES["host_compiler_cublas"].baseline_id},
            {"nonfinite_outputs": 1}, {"iterations": 20}, {"samples_ms": [1.]*99},
            {"max_rel_error": 0.5, "max_abs_error": 0.5},
        ):
            with self.assertRaises(ValueError):
                validate_measurement({**row, **change}, job, inputs, config)

    @unittest.skipIf(np is None, "optional NumPy is needed only to prepare CPU reference files")
    def test_canonical_files_and_merged_gemm_equal_per_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"inputs"
            shape = (4, 3, 32, 16)
            metadata = write_inputs(shape, root)
            p, m, k, n = shape
            inputs = np.fromfile(root/"inputs.bin", dtype="<f2")
            a = inputs[:p*p*m*k].reshape(p, p, m, k)
            weights = inputs[p*p*m*k:].reshape(p, k, n)
            reference = np.fromfile(root/"reference.bin", dtype="<f4").reshape(p, p, m, n)
            self.assertEqual(inputs.size, p*p*m*k+p*k*n)
            for source in range(p):
                for destination in range(p):
                    np.testing.assert_array_equal(a[source, destination], matrix(m, k, source*p+destination))
                    # Scalar accumulation is independent of the preparation GEMM call.
                    for row in range(m):
                        for col in range(n):
                            exact = sum(float(a[source, destination, row, i])*float(weights[destination, i, col])
                                        for i in range(k))
                            self.assertEqual(reference[source, destination, row, col], exact)
            for destination in range(p):
                merged = a[:, destination].reshape(p*m, k).astype(np.float32) @ weights[destination].astype(np.float32)
                np.testing.assert_array_equal(merged.reshape(p, m, n), reference[:, destination])
            self.assertEqual(metadata["input_bytes"], inputs.nbytes)
            with self.assertRaises(FileExistsError):
                write_inputs(shape, root)


if __name__ == "__main__":
    unittest.main()
