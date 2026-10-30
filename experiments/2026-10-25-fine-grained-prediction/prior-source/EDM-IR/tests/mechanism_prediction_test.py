"""Regression checks for evidence separation, identifiability and bounded fitting."""
import copy
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
import clean_mechanism_data as cleaning
from evaluate_mechanism_prediction import DEFAULT_DATA, folds, structural_aliases
from edmir.compiler.mechanism_prediction import (
    HOSTS, MAX_ROWS, MechanismModel, basis, conditional_policy_latency,
    critical_path_us, geometry, nonnegative_fit, select_prediction,
)


class NumericalTests(unittest.TestCase):
    def test_known_positive_solution_and_kkt(self):
        x = [[1., 0.], [1., 1.], [1., 2.], [1., 4.]]
        fit = nonnegative_fit(x, [3., 5., 7., 11.], [1.]*4)
        for a, b in zip(fit["coefficients_us"], (3., 2.)):
            self.assertAlmostEqual(a, b, places=4)
        self.assertLess(fit["kkt_relative_violation"], 1e-8)

    def test_active_constraint_cannot_produce_negative_cost(self):
        fit = nonnegative_fit([[1., 0.], [1., 1.], [1., 2.]], [6., 4., 2.], [1.]*3)
        self.assertAlmostEqual(fit["coefficients_us"][1], 0.)
        self.assertAlmostEqual(fit["coefficients_us"][0], 4., places=4)

    def test_rank_deficiency_is_not_hidden_by_regularization(self):
        fit = nonnegative_fit([[1., 2., 4.], [1., 3., 6.], [1., 4., 8.]], [5., 7., 9.], [1.]*3)
        self.assertEqual(fit["unregularized_rank"], 2)
        self.assertEqual(fit["feature_count"], 3)

    def test_bounded_rows_dimensions_nonfinite_and_deadline(self):
        bad = [([[1.]]*(MAX_ROWS+1), [1.]*(MAX_ROWS+1), [1.]*(MAX_ROWS+1)),
               ([[1.]*8], [1.], [1.]), ([[float("nan")]], [1.], [1.]),
               ([[1.]], [float("inf")], [1.]), ([[1.]], [1.], [0.])]
        for args in bad:
            with self.assertRaises(ValueError):
                nonnegative_fit(*args)
        with self.assertRaises(TimeoutError):
            nonnegative_fit([[1.]*7]*20, [1.]*20, [1.]*20, timeout_seconds=1e-12)

    def test_equal_traffic_does_not_imply_equal_cutlass_depth(self):
        a, b = (4, 512, 4096, 512), (4, 256, 8192, 512)
        self.assertEqual(basis(a, "host", "coupled_counts"), basis(b, "host", "coupled_counts"))
        self.assertNotEqual(basis(a, "host", "cutlass_depth"), basis(b, "host", "cutlass_depth"))
        self.assertEqual(geometry(a, "host_merged_cutlass")["actual_gemm_M"], 2048)
        self.assertEqual(geometry(a, "host")["actual_gemm_M"], 512)

    def test_split_and_group_counts_are_distinct(self):
        a = geometry((4,512,4096,512), "splitk4")
        b = geometry((4,512,4096,512), "splitk4_groupedn2")
        self.assertEqual(a["ctas"], 2*b["ctas"])
        self.assertEqual(a["logical_A_replay_bytes"], 2*b["logical_A_replay_bytes"])
        self.assertEqual(a["extra_serial_output_bytes"], b["extra_serial_output_bytes"])
        self.assertEqual(a["extra_serial_output_bytes"], 4*512*512*3)
        with self.assertRaises(ValueError):
            geometry((4,64,1000,512), "splitk4")
        with self.assertRaises(ValueError):
            geometry((8,64,4096,512), "splitk4")

    def test_critical_path_recomputes_tied_paths(self):
        graph = {0: [], 1: [], 2: [0,1]}
        self.assertEqual(critical_path_us(graph, {0: 10, 1: 10, 2: 2}), 12)
        # Removing one tied branch does not expose the other branch's time.
        self.assertEqual(critical_path_us(graph, {0: 0, 1: 10, 2: 2}), 12)
        with self.assertRaises(ValueError):
            critical_path_us({0:[1],1:[0]}, {0:1.,1:1.})

    def test_scenario_can_lose_when_contention_increases(self):
        self.assertAlmostEqual(conditional_policy_latency(1100, 0, 1), 400)
        self.assertEqual(conditional_policy_latency(1100, 1, 3), 1100)
        self.assertGreater(conditional_policy_latency(1100, .25, 4), 1100)
        with self.assertRaises(ValueError):
            conditional_policy_latency(1100, -.1, 1)

    def test_candidate_permutation_and_explicit_tie_rule(self):
        rows = [{"candidate_id": k, "predicted_us": t} for k, t in (("q4",10.),("q2",10.),("host",20.))]
        self.assertEqual({select_prediction(list(p)) for p in permutations(rows)}, {"q2"})
        with self.assertRaises(ValueError):
            select_prediction([rows[0], rows[0]])


class RealCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.configs, cls.manifest = cleaning.clean_collection(DEFAULT_DATA)
        cls.primary = [r for r in cls.configs if r["primary_scope"]]
        cls.stable = [r for r in cls.primary if r["eligible_for_calibration"]]
        cls.package = cleaning.read(DEFAULT_DATA/"revision-01/package.json")["package"]
        cls.infos = cleaning.read(DEFAULT_DATA/"revision-01/inputs.json")["workloads"]
        directory = DEFAULT_DATA/"results/mechanisms-01"
        cls.main_rows = [json.loads(x) for x in (directory/"measurements.jsonl").read_text().splitlines()]
        cls.jobs = [j for j in cls.package["jobs"] if j["stage"] == "mechanisms"]
        cls.stdout = [cleaning.read(directory/f"stdout_{i:05d}.txt") for i in range(len(cls.jobs))]
        cls.context = cleaning.read(directory/"context.json")

    def validate(self, rows=None, stdout=None):
        cleaning.validate_block_rows(rows if rows is not None else self.main_rows,
            self.jobs, self.infos, self.package, self.context, stdout if stdout is not None else self.stdout)

    def test_all_raw_samples_and_source_identities_are_retained(self):
        self.assertEqual(len(self.rows), 306)
        self.assertEqual(sum(len(r["samples_ms"]) for r in self.rows), 30600)
        self.assertEqual(len(self.primary), 69)
        self.assertEqual(len(self.stable), 63)
        self.assertEqual(self.manifest["verified_sources"], 136)
        self.assertEqual(len(self.manifest["input_binary_hashes"]), 20)
        for row in self.rows:
            raw = {k:v for k,v in row.items() if k != "cleaning"}
            self.assertEqual(cleaning.digest(raw), row["cleaning"]["raw_row_sha256"])

    def test_configuration_quality_keeps_all_three_processes(self):
        for c in self.configs:
            self.assertEqual(len(c["process_ids"]), 3)
            self.assertEqual(c["iteration_samples_retained"], 300)
        self.assertEqual(sum(bool(c["quality_flags"]) for c in self.primary), 6)
        self.assertTrue(any(c["warmup"] == 200 for c in self.configs))
        self.assertTrue(all(c["warmup"] == 20 for c in self.stable))

    def test_missing_and_duplicate_blocks_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(rows=self.main_rows[:-1])
        rows = copy.deepcopy(self.main_rows)
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            self.validate(rows=rows)
        with self.assertRaises(ValueError):
            cleaning.aggregate(self.rows[:-1])

    def test_mutated_hash_input_protocol_sample_and_statistics_rejected(self):
        for field, value in (("input_sha256", "0"*64), ("mean_ms", 0.001),
                             ("warmup",200), ("metadata_mode","merged"),
                             ("gemm_algorithm_id","unknown"), ("max_rel_error",float("nan")),
                             ("measurement_policy",{"profiler":True,"nvtx":False,"discarded_samples":0})):
            rows = copy.deepcopy(self.main_rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(rows=rows)
        rows = copy.deepcopy(self.main_rows)
        rows[0]["samples_ms"][0] = float("inf")
        with self.assertRaises(ValueError):
            self.validate(rows=rows)
        stdout = copy.deepcopy(self.stdout)
        stdout[0]["max_abs_error"] = .001
        with self.assertRaises(ValueError):
            self.validate(stdout=stdout)

    def test_sealed_file_change_rejected_before_calibration(self):
        original = cleaning.sha
        def changed(path):
            return "0"*64 if path.name == "execution_accounts.json" else original(path)
        with patch.object(cleaning, "sha", side_effect=changed), self.assertRaisesRegex(ValueError, "sealed file"):
            cleaning.clean_collection(DEFAULT_DATA)

    def test_context_and_quality_mix_cannot_silently_enter_fit(self):
        with self.assertRaises(ValueError):
            MechanismModel.fit(self.primary)
        mixed = copy.deepcopy(self.stable)
        mixed[0]["context_id"] = "other-source"
        with self.assertRaises(ValueError):
            MechanismModel.fit(mixed)
        mixed = copy.deepcopy(self.stable)
        mixed[0]["dependency_policy"] = "uniform_immutable_peer_v1"
        with self.assertRaises(ValueError):
            MechanismModel.fit(mixed)

    def test_new_source_or_policy_cannot_reuse_old_identity(self):
        model = MechanismModel.fit(self.stable)
        row = self.stable[0]
        with self.assertRaises(ValueError):
            model.predict(row["shape"], row["family"], context_id="new-source")
        with self.assertRaises(ValueError):
            model.predict(row["shape"], row["family"], context_id=model.context_id,
                          dependency_policy="uniform_immutable_peer_v1")

    def test_shuffled_calibration_rows_are_identical(self):
        a = MechanismModel.fit(self.stable, variant="cutlass_depth")
        b = MechanismModel.fit(list(reversed(self.stable)), variant="cutlass_depth")
        self.assertEqual(a.to_dict(), b.to_dict())
        for row in self.stable:
            p = a.predict(row["shape"], row["family"], context_id=a.context_id)
            self.assertAlmostEqual(sum(p["contributions_us"].values()), p["predicted_us"])

    def test_complete_shape_holdout_and_label_noninterference(self):
        predictions, audits = folds(self.primary, "cutlass_depth")
        target = [4,512,4096,512]
        changed = copy.deepcopy(self.primary)
        for row in changed:
            if row["shape"] == target:
                row["observed_us"] *= 100
        altered_predictions, altered_audits = folds(changed, "cutlass_depth")
        for audit in audits:
            self.assertNotIn(tuple(audit["held_out_shape"]), audit["model"]["training_shapes"])
        before = next(a for a in audits if list(a["held_out_shape"]) == target)
        after = next(a for a in altered_audits if list(a["held_out_shape"]) == target)
        self.assertEqual(before, after)
        x = {r["candidate_id"]:r["predicted_us"] for r in predictions if r["shape"] == target}
        y = {r["candidate_id"]:r["predicted_us"] for r in altered_predictions if r["shape"] == target}
        self.assertEqual(x, y)

    def test_alias_lower_bound_and_repair(self):
        aliases = structural_aliases(self.stable, ("coupled_counts", "cutlass_depth"))
        old = [r for r in aliases if r["family"] == "host" and r["variant"] == "coupled_counts"]
        self.assertEqual(len(old), 1)
        self.assertGreater(old[0]["unavoidable_max_absolute_error_us"], 100)
        self.assertFalse(any(r["family"] == "host" and r["variant"] == "cutlass_depth" for r in aliases))


if __name__ == "__main__":
    unittest.main()
