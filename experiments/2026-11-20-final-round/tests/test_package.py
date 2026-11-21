#!/usr/bin/env python3
"""Offline tests for the final-round package. Run: python3 -I tests/test_package.py"""

import csv
import gzip
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DESIGN = ROOT / "design"
SCENARIO = ROOT / "params" / "scenario.json"
GLM_ROUTES = ROOT.parent / "2026-11-17-real-moe-gpu" / "recovered" / "routes-attempt01" / "workers"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402
import hw_gate  # noqa: E402
import predict  # noqa: E402


def run(script, *args, check=True):
    r = subprocess.run([sys.executable, "-I", str(SCRIPTS / script), *map(str, args)],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{script} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r


def rows(path):
    return common.read_csv(path)


class Design(unittest.TestCase):
    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as t:
            run("make_design.py", "--out", t)
            for name in ("cells.csv", "snapshots.csv", "arms.csv", "comparisons.csv", "blocks.csv",
                         "queue.csv", "policies.json"):
                self.assertEqual((Path(t) / name).read_bytes(), (DESIGN / name).read_bytes(), name)

    def test_invariants(self):
        cells = rows(DESIGN / "cells.csv")
        self.assertEqual(len(cells), 33)
        self.assertEqual({c["cohort"] for c in cells if c["width_id"] == "QWEN36" and c["m_per_source_destination"] in ("32", "64")},
                         {"historical_overlap", "historical_calibration"})
        queue = rows(DESIGN / "queue.csv")
        self.assertEqual(queue[0]["job_type"], "hw_gate")
        first_timing = next(i for i, j in enumerate(queue) if j["job_type"] == "timing_block")
        seal = next(i for i, j in enumerate(queue) if j["job_type"] == "calibrate_predict_seal")
        self.assertLess(seal, first_timing, "predictions must be sealed before any timing block")
        snap_seal = next(i for i, j in enumerate(queue) if j["job_type"] == "route_stats_predict_seal")
        first_snap = next(i for i, j in enumerate(queue) if j["job_type"] == "timing_block" and j["process_id"].startswith("S"))
        self.assertLess(snap_seal, first_snap)


class Predict(unittest.TestCase):
    def test_scenario_runs_and_is_sane(self):
        with tempfile.TemporaryDirectory() as t:
            run("predict.py", "--params", SCENARIO, "--design", DESIGN, "--out", t)
            pred = {(r["unit_id"], r["arm_id"]): float(r["predicted_ms"]) for r in rows(Path(t) / "predictions.csv")}
            for width in ("GLM", "QWEN36", "DSV2L"):
                times = [pred[(f"{width}_M{m:04d}", "D_Q1_G2")] for m in (2 ** i for i in range(11))]
                self.assertEqual(times, sorted(times), f"direct time should not fall as M grows ({width})")
            # Replay 12 (N3072) costs more than replay 4 (N1024) at the same M.
            self.assertGreater(pred[("GLM_M1024", "D_Q1_G2")], pred[("QWEN36_M1024", "D_Q1_G2")])
            seal = json.loads((Path(t) / "PREDICTION_SEAL.json").read_text())
            self.assertIn("scenario", seal["params_source"])

    def test_bandwidth_table_interpolation(self):
        p = json.loads(SCENARIO.read_text())
        p["b_in"] = {"table": [[7, 8, 50e9], [7, 64, 200e9], [7, 108, 230e9]]}
        mc = predict.Machine(p)
        self.assertAlmostEqual(mc.b_in(7, 8), 50e9)
        self.assertAlmostEqual(mc.b_in(7, 4), 25e9)
        self.assertAlmostEqual(mc.b_in(7, 500), 230e9)
        self.assertTrue(50e9 < mc.b_in(7, 20) < 200e9)


class Calibrate(unittest.TestCase):
    def test_recovers_known_parameters(self):
        recs = [dict(kind="F_direct", seconds=1e-4), dict(kind="F_direct", seconds=1.2e-4), dict(kind="F_direct", seconds=1.1e-4),
                dict(kind="G_launch", seconds=2e-5), dict(kind="F_staged", seconds=3e-5)]
        recs += [dict(kind="b_in", peers=7, ctas=c, bytes=1e8, seconds=1e8 / bw)
                 for c, bw in ((8, 40e9), (32, 150e9), (108, 240e9))]
        recs += [dict(kind="a2a", bytes_in_per_rank=b, seconds=25e-6 + b / 180e9) for b in (1e4, 1e6, 1e8)]
        recs += [dict(kind="hbm", bytes=b, seconds=b / 1.6e12) for b in (1e6, 1e8, 1e9, 4e9)]
        recs += [dict(kind="mma", ctas=108, g=2, flops=1e12, seconds=1e12 / 2.2e14)]
        with tempfile.TemporaryDirectory() as t:
            mb = Path(t) / "mb.jsonl"
            mb.write_text("".join(json.dumps(r) + "\n" for r in recs))
            run("calibrate.py", "--base", SCENARIO, "--microbench", mb, "--out", Path(t) / "p.json")
            p = json.loads((Path(t) / "p.json").read_text())
        self.assertEqual(p["source"], "microbench")
        self.assertAlmostEqual(p["F_direct"], 1.1e-4)
        self.assertAlmostEqual(p["a2a"]["latency"], 25e-6, delta=1e-9)
        self.assertAlmostEqual(p["a2a"]["bw"], 180e9, delta=1e6)
        self.assertAlmostEqual(p["b_hbm"], 1.6e12, delta=1e3)
        self.assertAlmostEqual(p["mma_peak"], 2.2e14, delta=1e6)
        self.assertEqual(p["b_in"]["table"][0], [7, 8, 40e9])

    def test_missing_kind_fails(self):
        with tempfile.TemporaryDirectory() as t:
            mb = Path(t) / "mb.jsonl"
            mb.write_text(json.dumps(dict(kind="F_direct", seconds=1e-4)) + "\n")
            r = run("calibrate.py", "--base", SCENARIO, "--microbench", mb, "--out", Path(t) / "p.json", check=False)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing microbenchmark kinds", r.stderr)


class RouteStats(unittest.TestCase):
    def write_capture(self, d):
        # 2 ranks, 4 experts (2 per rank), top-2; one prefill step and one decode step, layer 5.
        routes = {0: [[0, 2], [1, 3], [0, 1]], 1: [[2, 3], [0, 2]]}
        for r in (0, 1):
            with gzip.open(d / f"blk-rank{r}.jsonl.gz", "wt") as f:
                for step, cat in ((10, "pure_prefill"), (11, "pure_decode")):
                    f.write(json.dumps(dict(kind="step", step=step, rank=r, category=cat)) + "\n")
                    f.write(json.dumps(dict(kind="route", step=step, rank=r, layer=5, topk_ids=routes[r])) + "\n")

    def test_synthetic_counts(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.write_capture(d)
            run("route_stats.py", "--dir", d, "--block", "blk", "--experts", 4, "--ranks", 2,
                "--snapshot", "X_PREFILL:pure_prefill:1:5", "--aggregate", "pure_prefill", "--out", d / "o.json")
            o = json.loads((d / "o.json").read_text())
        s = o["snapshots"]["X_PREFILL"]
        # destination 0 holds experts 0,1; destination 1 holds experts 2,3
        self.assertEqual(s["rows"][0][0], {"0": 2, "1": 2})
        self.assertEqual(s["rows"][0][1], {"0": 1})
        self.assertEqual(s["rows"][1][0], {"2": 1, "3": 1})
        self.assertEqual(s["rows"][1][1], {"2": 2, "3": 1})
        self.assertEqual(s["dest_load"], [5, 5])
        self.assertEqual(s["tokens_per_source"], [3, 2])
        # remote destinations per token: rank0 tokens -> {1},{1},{} ; rank1 tokens -> {},{0}
        self.assertAlmostEqual(s["mean_distinct_remote_ranks"], 3 / 5)

    @unittest.skipUnless((GLM_ROUTES / "diag-S32-rank0.jsonl.gz").exists(), "GLM 2026-11-17 routes not present")
    def test_reproduces_glm_numbers(self):
        with tempfile.TemporaryDirectory() as t:
            run("route_stats.py", "--dir", GLM_ROUTES, "--block", "diag-S32", "--experts", 64, "--ranks", 8,
                "--aggregate", "pure_prefill", "--out", Path(t) / "o.json")
            a = json.loads((Path(t) / "o.json").read_text())["aggregate"]["pure_prefill"]
        self.assertEqual(a["step_layers"], 138)
        self.assertAlmostEqual(a["busiest_over_mean_median"], 1.453, places=3)
        self.assertAlmostEqual(a["mean_distinct_remote_ranks"], 2.957, places=3)


def synthetic_metrics(path, fail=None):
    """Ground truth: on GLM the q1 direct arm beats SLT up to M16 and loses from M32; elsewhere it ties."""
    cells = {c["cell_id"]: c for c in rows(DESIGN / "cells.csv")}
    out = []
    for b in rows(DESIGN / "blocks.csv"):
        unit, arm = b["unit_id"], b["arm_id"]
        jitter = 1.001 if b["orientation"] == "forward" else 0.999
        if unit in cells:
            m, width = int(cells[unit]["m_per_source_destination"]), cells[unit]["width_id"]
            base = {"SLT": 1.0, "D_Q4_G2": 0.9}.get(arm, 1.0)
            if arm == "D_Q1_G2":
                base = (0.8 if m <= 16 else 1.3) if width == "GLM" else 1.0
        else:
            base = {"NATIVE": 1.0, "D_Q1_G2_R": 1.2}.get(arm, 1.1)
        status = "ok"
        if fail and (b["process_id"], arm) == fail:
            status = "numerical_failure"
        row = dict(process_id=b["process_id"], arm_id=arm, status=status, harness="h1")
        if status == "ok":
            row["median_wall_ms"] = base * jitter
        out.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))


def synthetic_admission(path):
    cells = rows(DESIGN / "cells.csv")
    arms = rows(DESIGN / "arms.csv")
    seeds = sorted({b["rep"] for b in rows(DESIGN / "blocks.csv") if b["stage"] == "balanced"})
    out = [dict(unit_id=c["cell_id"], arm_id=a["arm_id"], input_id=s, status="ok")
           for c in cells for a in arms if a["stage"] == "balanced" for s in seeds]
    out += [dict(unit_id=s["snapshot_id"], arm_id=a["arm_id"], input_id="real", status="ok")
            for s in rows(DESIGN / "snapshots.csv") for a in arms if a["stage"] == "snapshot"]
    path.write_text("".join(json.dumps(r) + "\n" for r in out))


class Analyze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        synthetic_metrics(t / "m.jsonl")
        synthetic_admission(t / "a.jsonl")
        run("predict.py", "--params", SCENARIO, "--design", DESIGN, "--out", t / "pred")
        run("analyze.py", "--design", DESIGN, "--metrics", t / "m.jsonl", "--admission", t / "a.jsonl",
            "--pred-cells", t / "pred", "--out", t / "out")
        cls.out = t / "out"
        cls.t = t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_crossover_detected_at_the_true_bin(self):
        cross = rows(self.out / "crossover_measured.csv")
        glm = [r for r in cross if r["comparison_id"] == "Bb_q1" and r["group"] == "GLM"]
        self.assertEqual([(r["kind"], r["m_low"], r["m_high"]) for r in glm], [("strict", "16", "32")])
        self.assertEqual(glm[0]["direction"], "candidate_wins->baseline_wins")
        qwen = [r for r in cross if r["comparison_id"] == "Bb_q1" and r["group"] == "QWEN36"]
        self.assertEqual(qwen[0]["kind"], "none")

    def test_r1_scores_scenario_prediction(self):
        r1 = {(r["comparison_id"], r["group"]): r["verdict"] for r in rows(self.out / "r1_crossover.csv")}
        self.assertEqual(r1[("Bb_q1", "GLM")], "miss")      # scenario predicts no crossover
        self.assertEqual(r1[("Bb_q1", "QWEN36")], "hit")    # both none

    def test_regret(self):
        reg = rows(self.out / "r2_regret.csv")
        get = {(r["unit_id"], r["policy"]): r for r in reg if r["stage"] == "balanced"}
        self.assertEqual(get[("GLM_M0004", "FIXED_DIRECT_Q4")]["oracle_arm"], "D_Q1_G2")
        self.assertAlmostEqual(float(get[("GLM_M0004", "FIXED_DIRECT_Q4")]["regret"]), 0.9 / 0.8 - 1, places=6)
        self.assertAlmostEqual(float(get[("GLM_M0064", "FIXED_DIRECT_Q4")]["regret"]), 0.0, places=9)
        self.assertEqual(get[("GLM_M0004", "ALWAYS_STAGED")]["chosen_arm"], "SLT")
        summary = rows(self.out / "r2_summary.csv")
        p0 = [r for r in summary if r["stage"] == "balanced" and r["scope"] == "all" and r["policy"] == "P0"][0]
        self.assertTrue(p0["verdict"].startswith("R2a="))

    def test_numerical_failure_excludes_arm_on_unit(self):
        t = self.t
        b = next(r for r in rows(DESIGN / "blocks.csv") if r["unit_id"] == "GLM_M0004" and r["arm_id"] == "D_Q1_G2"
                 and r["block_type"] == "core")
        synthetic_metrics(t / "m2.jsonl", fail=(b["process_id"], "D_Q1_G2"))
        run("analyze.py", "--design", DESIGN, "--metrics", t / "m2.jsonl", "--admission", t / "a.jsonl",
            "--out", t / "out2")
        units = {(r["comparison_id"], r["unit_id"]): r for r in rows(t / "out2" / "unit_effects.csv")}
        self.assertEqual(units[("Bb_q1", "GLM_M0004")]["unit_class"], "incomplete")
        self.assertEqual(units[("Bb_q1", "GLM_M0004")]["pairs_complete"], "0")

    def test_preview_without_admission(self):
        t = self.t
        run("analyze.py", "--design", DESIGN, "--metrics", t / "m.jsonl", "--out", t / "out3")
        self.assertIn("PREVIEW", (t / "out3" / "SUMMARY.txt").read_text())
        units = rows(t / "out3" / "unit_effects.csv")
        self.assertTrue(units and all(u["admission_verified"] == "0" for u in units))

    def test_tampered_prediction_is_rejected(self):
        t = self.t
        pred = t / "pred_tampered"
        run("predict.py", "--params", SCENARIO, "--design", DESIGN, "--out", pred)
        with (pred / "policy_choice.csv").open("a") as f:
            f.write("GLM_M0001,P0,D_Q1_G2\n")
        r = run("analyze.py", "--design", DESIGN, "--metrics", t / "m.jsonl", "--pred-cells", pred,
                "--out", t / "out4", check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("changed after it was sealed", r.stderr)

    def test_unplanned_metric_is_rejected(self):
        t = self.t
        bad = t / "bad.jsonl"
        bad.write_text(json.dumps(dict(process_id="B9999", arm_id="SLT", status="ok", median_wall_ms=1, harness="h")) + "\n")
        r = run("analyze.py", "--design", DESIGN, "--metrics", bad, "--out", t / "out5", check=False)
        self.assertNotEqual(r.returncode, 0)


class Amdahl(unittest.TestCase):
    def test_bound(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            attr = [dict(width_id="GLM", phase="DECODE", run=i, instrumented=True, step_wall_ms=10.0, replaceable_ms=1.0)
                    for i in range(3)]
            attr += [dict(width_id="GLM", phase="DECODE", run=i, instrumented=False, step_wall_ms=9.9) for i in range(3)]
            (t / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in attr))
            with (t / "u.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["comparison_id", "stage", "unit_id", "candidate_arm", "median_ratio", "unit_class"])
                w.writeheader()
                w.writerow(dict(comparison_id="R_Bb_q1", stage="snapshot", unit_id="GLM_DECODE",
                                candidate_arm="D_Q1_G2_R", median_ratio="2.0", unit_class="candidate_wins"))
            run("amdahl.py", "--attr", t / "a.jsonl", "--units", t / "u.csv", "--out", t / "o.csv")
            r = rows(t / "o.csv")[0]
        self.assertAlmostEqual(float(r["f"]), 0.1)
        self.assertAlmostEqual(float(r["whole_model_bound"]), 1 / (0.9 + 0.05), places=6)


class Manifest(unittest.TestCase):
    def test_create_verify_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "res"
            (root / "sub").mkdir(parents=True)
            (root / "a.txt").write_text("a")
            (root / "sub" / "b.bin").write_bytes(b"\x00\x01")
            run("manifest.py", "create", "--root", root, "--out", root / "MANIFEST.json")
            self.assertEqual(run("manifest.py", "verify", "--root", root, "--manifest", root / "MANIFEST.json").returncode, 0)
            (root / "a.txt").write_text("changed")
            r = run("manifest.py", "verify", "--root", root, "--manifest", root / "MANIFEST.json", check=False)
            self.assertEqual(r.returncode, 1)
            self.assertIn("changed a.txt", r.stdout)


class HardwareGateParsing(unittest.TestCase):
    def test_topology(self):
        good = "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\x1b[0m\nGPU0\t X \tNV12\t0-63\nGPU1\tNV12\t X \t0-63\n\nLegend:\n  X = Self\n"
        self.assertEqual(hw_gate.topology_ok(good, 2), (True, []))
        self.assertFalse(hw_gate.topology_ok("\tGPU0\tGPU1\nGPU0\tX\tPIX\nGPU1\tPIX\tX\n", 2)[0])
        self.assertTrue(hw_gate.p2p_ok("\tGPU0\tGPU1\nGPU0\tX\tOK\nGPU1\tOK\tX\n", 2))
        self.assertFalse(hw_gate.p2p_ok("\tGPU0\tGPU1\nGPU0\tX\tNS\nGPU1\tNS\tX\n", 2))


@unittest.skipUnless(sys.platform.startswith("linux"), "lease_guard needs Linux /proc")
class LeaseGuard(unittest.TestCase):
    def test_launch_and_refuse(self):
        far = "2099-01-01T00:00:00+00:00"
        self.assertEqual(run("lease_guard.py", "--lease-end", far, "--", "true").returncode, 0)
        near = "2000-01-01T00:00:00+00:00"
        self.assertEqual(run("lease_guard.py", "--lease-end", near, "--", "true", check=False).returncode, 75)


class Intervals(unittest.TestCase):
    def test_strict_and_loose(self):
        seq = [(1, "candidate_wins"), (2, "tie"), (4, "baseline_wins"), (8, "baseline_wins"), (16, "candidate_wins")]
        self.assertEqual(common.intervals(seq), [("strict", 8, 16, "baseline_wins->candidate_wins"),
                                                 ("loose", 1, 4, "candidate_wins->baseline_wins")])
        self.assertEqual(common.intervals([(1, "candidate_wins"), (2, "incomplete"), (4, "baseline_wins")]), [])
        self.assertEqual(common.classify(1.031, 1.03), "candidate_wins")
        self.assertEqual(common.classify(1 / 1.031, 1.03), "baseline_wins")
        self.assertEqual(common.classify(1.0, 1.03), "tie")
        self.assertTrue(math.isinf(predict.Machine(json.loads(SCENARIO.read_text())).b_in(0, 10)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
