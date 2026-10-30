from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / "tools")]
from edmir.compiler.fused_evaluation import fit_and_evaluate_fused_boundary
from edmir.compiler.fused_selection import select_fused_a2a_gemm
from edmir.compiler.measurement_dataset import P4_ENV, load_selector_dataset, sha256
from edmir.compiler.relative_selection import RelativeCostSelector, ScheduleCandidate
from edmir.compiler.two_stage_selection import (
    AnalyticalScheduleFeatures, SelectionGates, TwoStageSelector,
    _summarize_decisions, evaluate_group_heldout, parse_schedule_family,
)
from evaluate_relative_selector import score
from fused_evaluation_test import shape
from fused_selection_test import environment, problem, profile, topology
from two_stage_selection_test import measurement, synthetic_surface

REPO = Path(__file__).resolve().parents[2]


class PredictorReliabilityTest(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_surface()
        self.relative = RelativeCostSelector.fit(self.rows, environment_id="source")
        self.old = TwoStageSelector.fit(self.rows)
        self.measured = {r.family: r for r in self.rows[:6]}
        self.candidates = {n: ScheduleCandidate.from_measurement(r, "source")
                           for n, r in self.measured.items()}

    def test_nonfinite_evidence_cannot_admit_a_compiler_path(self):
        for value in (math.nan, math.inf, -math.inf):
            invalid = replace(profile(), evidence=replace(
                profile().evidence, heldout_max_selection_regret=value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_fused_a2a_gemm(problem(), topology(), invalid, environment(),
                                     allow_experimental_fused=True)
        with self.assertRaises(ValueError):
            replace(profile().evidence, heldout_host_wins=41).validate()

    def test_old_selector_rejects_mixed_workloads_and_mapping_aliases(self):
        mixed = {**self.measured, "host": measurement("g", "host", 32, 8192, 512, 0.5)}
        with self.assertRaises(ValueError):
            self.old.select(mixed)
        alias = dict(self.measured)
        alias["direct_alias"] = alias.pop("direct")
        with self.assertRaises(ValueError):
            self.old.select(alias)

    def test_group_holdout_rejects_the_same_workload_in_two_groups(self):
        duplicates = tuple(replace(r, group_id="duplicate") for r in self.rows[:6])
        with self.assertRaises(ValueError):
            evaluate_group_heldout(self.rows + duplicates)

    def test_static_geometry_matches_fixed_fp16_kernel_contract(self):
        direct = self.candidates["direct"]
        for field, value in (("rows_per_peer", math.nan), ("k", 0.5), ("n", True),
                             ("remote_input_bytes", direct.remote_input_bytes // 4),
                             ("remote_a_replay", direct.remote_a_replay + 1),
                             ("estimated_ctas_per_op", direct.estimated_ctas_per_op + 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(direct, **{field: value}).validate()
        for alias in ("host_groupedn1", "splitk1", "splitk02", "direct_groupedn1"):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                parse_schedule_family(alias)
        with self.assertRaisesRegex(ValueError, "alignment"):
            measurement("g", "splitk16", 32, 768, 512, 1.0).validate()
        # Odd base-N tile counts require ceiling division under grouping.
        tail = measurement("g", "splitk8_groupedn2", 65, 1024, 384, 1.0)
        tail.validate()
        self.assertEqual((tail.remote_a_replay, tail.estimated_ctas_per_op), (2, 32))

    def test_model_state_rejects_truncation_nonfinite_and_invalid_scales(self):
        for change in ({"coefficients": (0.0,)},
                       {"coefficients": (math.nan,) + self.relative.coefficients[1:]},
                       {"scales": (0.0,) + self.relative.scales[1:]},
                       {"calibration_offset": math.nan}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.relative, **change).select(self.candidates)
        for change in ({"coefficients": (0.0,)},
                       {"means": self.old.fused_model.means[:-1]},
                       {"ridge": math.nan}, {"sm_count": math.inf}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.old.fused_model, **change).predict_ms(self.measured["direct"])

    def test_world_size_cannot_change_under_an_existing_environment_identity(self):
        target = {n: replace(c, world_size=8,
                             remote_input_bytes=2 * 8 * 7 * c.rows_per_peer * c.k)
                  for n, c in self.candidates.items()}
        with self.assertRaises(ValueError):
            self.relative.select(target)
        with self.assertRaises(ValueError):
            self.relative.select(target, allow_environment_transfer=True)
        renamed = {n: replace(c, environment_id="p8") for n, c in target.items()}
        self.relative.select(renamed, allow_environment_transfer=True)

    def test_evaluation_rejects_training_workloads(self):
        with self.assertRaisesRegex(ValueError, "training"):
            score(self.relative, self.rows, "source")

    def test_dataset_requires_hash_coverage_dtype_and_truthful_counts(self):
        original = REPO / "experiments/2026-10-20-offline-predictor/data"
        for mode in ("missing_hashes", "dtype", "environment", "counts", "kernel", "timer"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                dest = Path(d)
                for name in ("manifest.json", "selector_measurements.jsonl", "component_measurements.jsonl"):
                    shutil.copyfile(original / name, dest / name)
                manifest = json.loads((dest / "manifest.json").read_text())
                path = dest / "selector_measurements.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                if mode == "missing_hashes":
                    manifest["files"] = {}
                elif mode == "dtype":
                    records[0]["dtype"] = "float32"
                elif mode == "environment":
                    manifest["environments"][P4_ENV]["world_size"] = 8
                elif mode == "kernel":
                    records[0]["kernel"] = "different-kernel"
                elif mode == "timer":
                    records[0]["timing_scope"] = "CUDA events"
                else:
                    manifest["counts"]["selector_rows"] = 1
                if mode in ("dtype", "kernel", "timer"):
                    path.write_text("".join(json.dumps(r) + "\n" for r in records))
                    manifest["files"][path.name] = sha256(path)
                (dest / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_selector_dataset(dest)

    def test_invalid_gates_or_metrics_cannot_report_admission(self):
        base = dict(oracle_direction="fused", selected_direction="fused",
                    oracle_family="direct", selected_family="direct",
                    oracle_ms=1.0, selected_ms=1.0, host_ms=2.0, regret=0.0)
        for change in ({"maximum_mean_regret": math.nan},
                       {"maximum_worst_regret": math.inf},
                       {"minimum_problems": 0.5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _summarize_decisions([base], replace(SelectionGates(), **change))
        with self.assertRaises(ValueError):
            _summarize_decisions([{**base, "regret": math.nan}], SelectionGates())
        with self.assertRaises(ValueError):
            _summarize_decisions([{**base, "regret": 0.9}], SelectionGates())

    def test_legacy_boundary_checks_labels_and_train_test_overlap(self):
        train = (*shape(256, "direct"), *shape(2048, "splitk8"))
        with self.assertRaises(ValueError):
            fit_and_evaluate_fused_boundary(train, train)
        invalid = (replace(train[0], measured_ms=math.nan), *train[1:])
        with self.assertRaises(ValueError):
            fit_and_evaluate_fused_boundary(invalid, shape(512, "direct", m=32))

    def test_host_gamma_is_per_source_gemm_not_merged_pm_grid(self):
        row = measurement("g", "host", 33, 512, 512, 1.0)
        self.assertEqual(AnalyticalScheduleFeatures.from_measurement(row).gamma, 4)

    def test_centered_residuals_independently_verify_pairwise_objective_stationarity(self):
        # Identity: mean_{a<b}(r_a-r_b)^2 = 2/(n-1) * sum_a(r_a-mean(r))^2.
        # Check the gradient using centered residuals rather than rebuilding
        # the normal equations or iterating over training pairs.
        groups = {}
        for row in self.rows:
            groups.setdefault(row.shape, []).append(row)
        model = self.relative
        gradient = [0.0] * len(model.coefficients)
        for rows in groups.values():
            designs, residuals = [], []
            for row in rows:
                explanation = model.explain(ScheduleCandidate.from_measurement(row, "source"))
                x = ((0.0,) * len(gradient) if row.is_host else
                     (1.0, *(term["standardized"] for term in explanation["terms"])))
                designs.append(x)
                residuals.append(explanation["predicted_log_ratio"] - math.log(row.measured_ms))
            mean = sum(residuals) / len(rows)
            for x, residual in zip(designs, residuals):
                for i in range(len(gradient)):
                    gradient[i] += 4 / (len(rows) - 1) * (residual - mean) * x[i]
        for i in range(1, len(gradient)):
            gradient[i] += 2 * model.ridge * model.coefficients[i]
        self.assertLess(max(map(abs, gradient)), 1e-9)

    def test_explanation_sums_to_prediction_and_evaluation_labels_cannot_change_choice(self):
        train = tuple(r for r in self.rows if r.group_id != "m64-n1024")
        test = tuple(r for r in self.rows if r.group_id == "m64-n1024")
        model = RelativeCostSelector.fit(train, environment_id="source")
        for row in test:
            explanation = model.explain(ScheduleCandidate.from_measurement(row, "source"))
            self.assertAlmostEqual(
                explanation["intercept"] + explanation["calibration_offset"] +
                sum(term["contribution"] for term in explanation["terms"]),
                explanation["predicted_log_ratio"],
            )
        baseline = score(model, test, "source")
        perturbed = tuple(replace(r, measured_ms=(1000.0 if r.is_host else 1.0 / (i + 1)))
                          for i, r in enumerate(test))
        changed = score(model, perturbed, "source")
        self.assertEqual([r["selected_family"] for r in baseline],
                         [r["selected_family"] for r in changed])
        self.assertNotEqual([r["regret"] for r in baseline], [r["regret"] for r in changed])


if __name__ == "__main__":
    unittest.main()
