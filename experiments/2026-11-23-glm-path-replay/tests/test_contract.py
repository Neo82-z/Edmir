import copy
import json
from pathlib import Path
import tempfile
import unittest

from analyze import check_rows, crossing_rows, effect_rows, score_crossing, load_prediction
from make_design import ROOT, BRIDGE, generate, read, sha
from snapshots import correctness, layout, load_receipts, reference, validate_tensors, validate_alignment, tiles

DESIGN = ROOT/"design"
DUMMY = "a"*64
BALANCED_MAIN = [arm for arm in __import__("make_design").BALANCED if arm not in BRIDGE]


def setup_run(out):
    blocks = read(DESIGN/"blocks.csv")
    run = dict(origin="synthetic_test",run_id="cpu-contract",harness="fixture",
               design_sha256=sha(DESIGN/"DESIGN.json"),hardware_sha256=DUMMY,inputs={},builds={})
    admission, metrics = {}, []
    for b in blocks:
        p,u,arm = b["process_id"],b["unit_id"],b["arm_id"]
        seed = b["rep"] if b["stage"]=="balanced" else "real"
        run["inputs"][f"{u}:{seed}"] = DUMMY
        run["builds"][arm] = DUMMY
        common = dict(unit_id=u,arm_id=arm,status="ok",run_id=run["run_id"],harness=run["harness"],
                      input_sha256=DUMMY,build_sha256=DUMMY,hardware_sha256=DUMMY)
        admission[u,seed,arm] = dict(**common,input_id=seed,scaled_max_err=.001,nonfinite=0,fingerprint_ok=True)
        size = int(u.removeprefix("GLM_M")) if b["stage"]=="balanced" else 1
        ms = (0.8 if size<64 else 1.2) if arm.startswith("D_") else 1.
        rank_ms = [ms*.95]*7+[ms]
        filename = f"samples-{ms}.json"
        path = out/filename
        if not path.exists():
            path.write_text(json.dumps([rank_ms]*100))
        metrics.append(dict(**common,process_id=p,rep=b["rep"],orientation=b["orientation"],
                            arm_position=b["arm_position"],warmup=20,timed=100,median_wall_ms=ms,
                            samples_file=filename,samples_sha256=sha(path)))
    return run,list(admission.values()),metrics


class TestDesign(unittest.TestCase):
    def test_generation_is_stable_and_forward_reverse_exact(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            generate(out)
            self.assertEqual({f.name:sha(f) for f in DESIGN.iterdir()},
                             {f.name:sha(f) for f in out.iterdir()})
        planned = read(DESIGN/"blocks.csv")
        by_pair = {}
        for row in planned:
            arms = by_pair.setdefault(row["pair_id"], {}).setdefault(row["orientation"], [])
            arms.append(row["arm_id"])
        self.assertEqual(len({r["process_id"] for r in planned}),78)
        self.assertEqual(sum(r["arm_id"] not in BRIDGE for r in planned),390)
        self.assertEqual(len(planned),522)
        jobs = [r["job"] for r in read(DESIGN/"queue.csv")]
        # The smoke needs the CUTLASS/cuBLASLt library, and both precede model capture.
        self.assertEqual(jobs[:3],["hw_gate","build_balanced","device_smoke"])
        self.assertLess(jobs.index("tune_native_glm"),jobs.index("capture_glm"))
        self.assertLess(jobs.index("microbench"),jobs.index("capture_glm"))
        status = {r["arm_id"]:r["gpu_status"] for r in read(DESIGN/"arms.csv")}
        self.assertEqual({a for a,s in status.items() if s=="gpu_run"},{"BOLD_Q1_G1","NATIVE"})
        self.assertEqual(len(by_pair),39)
        self.assertEqual(sum(r["stage"]=="snapshot" for r in planned),60)
        for orientations in by_pair.values():
            self.assertEqual(orientations["forward"],orientations["reverse"][::-1])


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.run,self.admission,self.metrics = setup_run(self.path)

    def results(self):
        blocks,policy,timings = check_rows(DESIGN,self.run,self.path,self.admission,self.metrics)
        return effect_rows(DESIGN,blocks,policy,timings)

    def test_complete_and_directional(self):
        effects = self.results()
        self.assertEqual(len(effects),63)
        self.assertTrue(all(r["pairs_complete"]==3 for r in effects))
        cross = crossing_rows(DESIGN,effects)
        self.assertEqual(len(cross),4)
        bins = [2**i for i in range(11)]
        for entry in cross:
            row, = entry["intervals"]
            self.assertEqual((row["m_low"],row["m_high"]),(32,64))
            self.assertEqual(row["direction"],"candidate_wins->baseline_wins")
            self.assertEqual(score_crossing(entry,[row],bins),"hit")
            self.assertEqual(score_crossing(entry,[dict(row,direction="baseline_wins->candidate_wins")],bins),"miss")
            self.assertEqual(score_crossing(entry,[dict(row,m_low=64,m_high=128)],bins),"hit")
            self.assertEqual(score_crossing(entry,[dict(row,m_low=128,m_high=256)],bins),"miss")
            self.assertEqual(score_crossing(entry,[dict(row,m_low=8,m_high=64)],bins),"unsupported")
            wide = dict(row,kind="loose",m_low=16,m_high=128)
            self.assertEqual(score_crossing(entry,[wide],bins),"hit_loose")
            self.assertEqual(score_crossing(entry,[dict(wide,direction="baseline_wins->candidate_wins")],bins),"miss")
            self.assertEqual(score_crossing(entry,[dict(wide,m_high=32)],bins),"miss")

    def test_half_pair_remains_incomplete(self):
        target = self.metrics[-1]
        self.metrics = [m for m in self.metrics if m["process_id"]!=target["process_id"]]
        rows = [r for r in self.results() if r["unit_id"]==target["unit_id"]]
        self.assertEqual(len(rows),5)
        self.assertTrue(all(r["pairs_complete"]==2 and r["ratio"] is None for r in rows))

    def test_failed_seed_excludes_arm_other_seeds_still_in_raw(self):
        index = next(i for i,r in enumerate(self.admission) if r["unit_id"].startswith("GLM_M") and
                     r["arm_id"]=="D_Q1_G2")
        good = self.admission[index]
        for bad in (dict(scaled_max_err=.0201),dict(nonfinite=1),dict(fingerprint_ok=False),dict(status="fallback_detected")):
            with self.subTest(bad=bad):
                self.admission[index] = dict(good,**bad)
                affected = [r for r in self.results() if r["unit_id"]==good["unit_id"] and
                            r["candidate_arm"]==good["arm_id"]]
                self.assertTrue(affected)
                self.assertTrue(all(r["pairs_complete"]==0 and r["outcome"]=="incomplete" for r in affected))

    def test_identity_and_order_and_sample_denominator(self):
        good = self.metrics[0]
        for field,value in (("arm_position","4"),("hardware_sha256","b"*64),("build_sha256","b"*64),
                            ("input_sha256","b"*64),("samples_sha256","b"*64),("median_wall_ms",.76)):
            with self.subTest(field=field):
                self.metrics[0] = dict(good,**{field:value})
                with self.assertRaises(ValueError):
                    self.results()

    def test_release_sensitivity_changes_only_direct_arms(self):
        from analyze import release_sensitivity
        blocks,policy,timings = check_rows(DESIGN,self.run,self.path,self.admission,self.metrics)
        phase = {key:dict(transport=.05,consumer=.5,release=.25) for key in timings}
        result = release_sensitivity(DESIGN,blocks,policy,timings,phase)
        ids = {r["comparison_id"] for r in result["effects"]}
        self.assertEqual(ids,{"Ba_q1","Ba_q4","R_Ba_q1","R_Ba_q4"})
        # Direct is 1.2 ms from M64 up and staged 1.0 ms: without the 0.25 ms release it wins everywhere.
        self.assertTrue(all(r["outcome"]=="candidate_wins" for r in result["effects"]))
        self.assertTrue(all(not c["intervals"] for c in result["crossings"]))

    def test_tie_separates_winners(self):
        effects = self.results()
        for r in effects:
            if r["unit_id"]=="GLM_M0032" and r["comparison_id"]=="Ba_q1":
                r["outcome"] = "tie"
        found = next(r for r in crossing_rows(DESIGN,effects) if r["comparison_id"]=="Ba_q1")
        self.assertEqual(found["coverage"],"unresolved")
        self.assertEqual(found["intervals"][0]["kind"],"loose")

    def test_device_result_requires_gate(self):
        self.run["origin"] = "device_measurement"
        gate = self.path/"hw_gate.json"
        gate.write_text(json.dumps(dict(verdict="FAIL")))
        self.run["hardware_sha256"] = sha(gate)
        with self.assertRaisesRegex(ValueError,"passing hardware gate"):
            self.results()


    def test_combined_identity_and_preboot_seal(self):
        self.assertEqual(len(self.run["inputs"]),35)
        self.assertEqual(len(self.run["builds"]),12)
        rows,choices = load_prediction(ROOT/"preboot_prediction",DESIGN,"balanced")
        self.assertEqual(len(choices),11)
        self.assertEqual(set(choices.values()),{"SLT"})
        self.assertEqual([(r["m_low"],r["m_high"]) for r in rows if r["kind"]=="loose"],
                         [("4","16"),("16","128")])

    def test_optional_bridge_gate_depends_on_bnew_admission(self):
        from types import SimpleNamespace
        from replay import lock_admission
        a = SimpleNamespace(run=self.path,design=DESIGN)
        (a.run/"RUN.json").write_text(json.dumps(self.run))
        admission = [r for r in self.admission if r["arm_id"] not in BRIDGE]
        def emit(rows):
            (a.run/"admission.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        emit(admission)
        lock_admission(a)
        receipt = a.run/"ADMISSION_LOCK.json"
        self.assertFalse(json.loads(receipt.read_text())["bridge_enabled"])
        receipt.unlink()
        bnew = next(r for r in admission if r["arm_id"]=="BNEW_Q1_G2")
        bnew["fingerprint_ok"] = False
        emit(admission)
        with self.assertRaisesRegex(ValueError,"finish the g1 bridge"):
            lock_admission(a)
        emit(self.admission)
        lock_admission(a)
        self.assertTrue(json.loads(receipt.read_text())["bridge_enabled"])

    def test_one_snapshot_admission_failure_does_not_stop_other_arms(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from replay import admit_jobs, append_row, record_identity

        a = SimpleNamespace(run=self.path,design=DESIGN,unit="GLM_PREFILL",timeout=1,snapshots=self.path)
        (a.run/"RUN.json").write_text(json.dumps(self.run))
        active = []
        def child(_,args,timeout):
            arm = args[-1]
            active.append(arm)
            if arm=="BNEW_Q1_G2_R":
                return 1
            append_row(a.run/"admission.jsonl",dict(**record_identity(self.run,a.unit,"real",arm),
                status="ok",input_id="real",scaled_max_err=.001,nonfinite=0,fingerprint_ok=True))
            return 0
        with patch("replay.launch_job",side_effect=child):
            admit_jobs(a)
            admit_jobs(a)
        self.assertEqual(len(active),5)
        rows = [json.loads(line) for line in (a.run/"admission.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["status"]=="runtime_error" for row in rows),1)

    def test_missing_snapshots_fail_before_any_launch_or_claim(self):
        import contextlib
        import io
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from replay import admit_jobs, cli, snapshot_receipts, time_all

        for snapshots in (None,self.path/"absent"):
            with self.subTest(snapshots=snapshots):
                a = SimpleNamespace(snapshots=snapshots,design=DESIGN)
                with self.assertRaisesRegex(ValueError,"snapshot input directory"):
                    snapshot_receipts(a,self.run,"GLM_PREFILL")
        grid_only = dict(self.run,inputs={k:v for k,v in self.run["inputs"].items() if not k.endswith(":real")})
        with self.assertRaisesRegex(ValueError,"without snapshots"):
            snapshot_receipts(SimpleNamespace(snapshots=self.path,design=DESIGN),grid_only,"GLM_PREFILL")
        (self.path/"RUN.json").write_text(json.dumps(self.run))
        a = SimpleNamespace(run=self.path,design=DESIGN,unit=None,timeout=1,snapshots=None)
        with patch("replay.launch_job",side_effect=AssertionError("launched")):
            with self.assertRaisesRegex(ValueError,"--snapshots"):
                admit_jobs(a)
            (self.path/"ADMISSION_LOCK.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"--snapshots"):
                time_all(a)
        self.assertFalse((self.path/"process-started").exists())
        self.assertFalse((self.path/"admission.jsonl").exists())
        for argv in (["verify"],["prepare","--gate","g","--smoke-receipt","s","--prediction","p"],
                     ["--snapshots","x","prepare","--gate","g","--smoke-receipt","s","--prediction","p",
                      "--without-snapshots","capture failed"]):
            with self.subTest(argv=argv), patch.object(sys,"argv",["replay.py","--run",str(self.path),
                    "--library",str(self.path/"lib.so"),*argv]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli()

    def test_grid_runs_without_snapshots(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from replay import admit_jobs, append_row, lock_admission, record_identity, time_all

        run = dict(self.run,inputs={k:v for k,v in self.run["inputs"].items() if not k.endswith(":real")})
        (self.path/"RUN.json").write_text(json.dumps(run))
        a = SimpleNamespace(run=self.path,design=DESIGN,unit=None,timeout=1,snapshots=None)
        units,retries = [],[]
        def child(_,args,timeout):
            if args[0]=="admit-arm":
                unit,rep,arm = args[2],args[4],args[6]
                units.append(unit)
                arms = [x for x in BALANCED_MAIN] if arm=="balanced-main" else [arm]
                for x in arms:
                    append_row(self.path/"admission.jsonl",dict(**record_identity(run,unit,rep,x),
                        status="ok",input_id=rep,scaled_max_err=.001,nonfinite=0,fingerprint_ok=True))
            else:
                (retries if "--retry" in args else units).append(args[2])
            return 0
        with patch("replay.launch_job",side_effect=child):
            admit_jobs(a)
            lock_admission(a)
            units.clear()
            time_all(a)
        # Only the 66 grid processes run; the fake child records no rows, so each gets its one retry.
        self.assertEqual(len(units),66)
        self.assertTrue(all(pid.startswith("B") for pid in units+retries))
        self.assertEqual(sorted(retries),sorted(units))

    def test_one_mechanical_retry_replaces_a_failed_process(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from analyze import first_attempt_failed
        from replay import time_all

        pid = self.metrics[0]["process_id"]
        arms = [m for m in self.metrics if m["process_id"]==pid]
        first = [dict(arms[0],attempt=1),dict(arms[1],attempt=1,status="runtime_error")]
        retry = [dict(m,attempt=2,median_wall_ms=m["median_wall_ms"]) for m in arms]
        self.assertTrue(first_attempt_failed(first,len(arms)))
        self.assertFalse(first_attempt_failed(arms,len(arms)))
        others = [m for m in self.metrics if m["process_id"]!=pid]
        self.metrics = first+retry+others
        effects = self.results()
        self.assertTrue(all(r["pairs_complete"]==3 for r in effects))
        # A retry after a complete first attempt is rejected, so retries cannot select results.
        self.metrics = arms+retry+others
        with self.assertRaisesRegex(ValueError,"without an infrastructure failure"):
            self.results()
        # time-all retries exactly the failed process, once, after the whole queue.
        (self.path/"RUN.json").write_text(json.dumps(self.run))
        (self.path/"ADMISSION_LOCK.json").write_text("{}")
        (self.path/"metrics.jsonl").write_text("".join(json.dumps(m)+"\n" for m in first+others))
        a = SimpleNamespace(run=self.path,design=DESIGN,timeout=1,snapshots=self.path)
        calls = []
        with patch("replay.launch_job",side_effect=lambda _,args,t:calls.append(args) or 0):
            time_all(a)
            time_all(a)
        retries = [c for c in calls if "--retry" in c]
        self.assertEqual(retries,[["time","--process",pid,"--retry"]])
        self.assertEqual(calls[-1],retries[0])

    def test_smoke_receipt_must_be_current_and_lists_failures(self):
        from make_design import BALANCED, SNAPSHOT
        from replay import SMOKE_CELLS, check_smoke, smoke_identity

        gate,library = self.path/"gate.json",self.path/"lib.so"
        gate.write_text("{}")
        library.write_bytes(b"so")
        uuids = [f"GPU-{i}" for i in range(8)]
        ok = dict(status="ok")
        receipt = dict(**smoke_identity(gate,uuids,library),arms={arm:ok for arm in SNAPSHOT},
                       balanced={u:{arm:ok for arm in BALANCED} for u in SMOKE_CELLS})
        self.assertEqual(check_smoke(receipt,gate,uuids,library),[])
        receipt["arms"]["NATIVE"] = dict(status="runtime_error")
        receipt["balanced"]["SMOKE_M0064"]["SLT"] = dict(status="not_run")
        del receipt["balanced"]["SMOKE_M0001"]["BOLD_Q1_G1"]
        self.assertEqual(check_smoke(receipt,gate,uuids,library),
                         ["NATIVE","SMOKE_M0001:BOLD_Q1_G1","SMOKE_M0064:SLT"])
        for field in ("library_sha256","balanced_sha256","hardware_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"smoke receipt"):
                check_smoke(dict(receipt,**{field:"b"*64}),gate,uuids,library)

    def test_microbench_receipt_and_seven_peer_partition(self):
        from microbench import SCHEMA, KINDS, check_receipt, reader_slot

        for owner in range(8):
            self.assertEqual(sorted(reader_slot(r,owner) for r in range(8) if r!=owner),list(range(7)))
        m_values = [2**i for i in range(11)]
        readings = [dict(kind=k,seconds=1e-5) for k in KINDS if k!="all_to_all"]
        readings += [dict(kind="all_to_all",m=m,seconds=1e-5) for m in m_values]
        uuids = [f"GPU-{i}" for i in range(8)]
        micro = dict(schema=SCHEMA,hardware_sha256=DUMMY,gpu_uuids=uuids,
                     implementation_sha256=sha(ROOT/"microbench.py"),rank_measurements=[readings]*8)
        self.assertEqual(check_receipt(micro,DUMMY,uuids,m_values),[])
        failed = [dict(kind="peer_read_7",error="CompilationError()") if r["kind"]=="peer_read_7" else r
                  for r in readings]
        self.assertEqual(check_receipt(dict(micro,rank_measurements=[failed]*8),DUMMY,uuids,m_values),
                         ["peer_read_7"])
        for broken in ([r for r in readings if r["kind"]!="peer_read_7"],
                       [r for r in readings if r.get("m")!=1024],
                       readings[:-1]+[dict(readings[-1],seconds=float("nan"))]):
            with self.assertRaisesRegex(ValueError,"missing a probe"):
                check_receipt(dict(micro,rank_measurements=[broken]*8),DUMMY,uuids,m_values)

    def test_bridge_signatures_are_declared_when_the_library_loads(self):
        import ctypes
        from types import SimpleNamespace
        from unittest.mock import patch
        from balanced import load_library

        class Fake:
            def __getattr__(self, name):
                fn = SimpleNamespace(argtypes=None,restype=None)
                setattr(self,name,fn)
                return fn
        with patch("balanced.ct.CDLL",return_value=Fake()):
            library = load_library(self.path/"lib.so")
        # BOLD can run in a process that never built a cuBLASLt plan.
        self.assertEqual(library.edmir_bridge_cutlass.argtypes[3:],(ctypes.c_void_p,)*4)
        self.assertEqual(library.edmir_lt_run.argtypes,(ctypes.c_void_p,)*7)

    def test_timing_claim_cannot_be_replaced(self):
        from replay import claim_process

        pid = read(DESIGN/"blocks.csv")[0]["process_id"]
        claim_process(self.path,pid)
        with self.assertRaises(FileExistsError):
            claim_process(self.path,pid)

    def test_device_event_samples_are_validated(self):
        gate = self.path/"hw_gate.json"
        from test_hw_gate import healthy_gate
        hw = healthy_gate()
        gate.write_text(json.dumps(hw))
        self.run.update(origin="device_measurement",hardware_sha256=sha(gate),gpu_uuids=hw["gpu_uuids"],
                        artifacts={},prediction={})
        r = self.metrics[0]
        self.metrics = [r]
        for a in self.admission:
            a["hardware_sha256"] = sha(gate)
        r.update(hardware_sha256=sha(gate),events_file=r["samples_file"],events_sha256=r["samples_sha256"],
                 median_event_ms=r["median_wall_ms"],enqueue_file=r["samples_file"],
                 enqueue_sha256=r["samples_sha256"],median_enqueue_ms=r["median_wall_ms"],rank_counts=[
                     dict(input_rows=2,live_assignments=2,projection_rows=2,projection_live_rows=2,padding_rows=0)
                     for _ in range(8)])
        self.results()
        # Device rows always need valid per-rank counts, with or without phase files.
        self.metrics = [dict(r,rank_counts=r["rank_counts"][:7])]
        with self.assertRaisesRegex(ValueError,"row counts"):
            self.results()
        # Phase durations are validated and feed the release-barrier sensitivity.
        from analyze import release_sensitivity
        phases = self.path/"phases.json"
        phases.write_text(json.dumps([[[.1,.5,.2]]*8]*100))
        good = dict(r,phase_names=["transport","consumer","release"],phases_file=phases.name,
                    phases_sha256=sha(phases),phase_ms=dict(transport=.1,consumer=.5,release=.2))
        self.metrics = [good]
        self.results()
        for bad in (dict(phase_ms=dict(transport=.1,consumer=.5,release=.3)),dict(phases_sha256="b"*64),
                    dict(phase_names=["transport","consumer"])):
            with self.subTest(bad=list(bad)):
                self.metrics = [dict(good,**bad)]
                with self.assertRaisesRegex(ValueError,"phase"):
                    self.results()
        self.metrics = [r]
        for field in ("median_event_ms","median_enqueue_ms"):
            with self.subTest(field=field):
                bad = dict(r,**{field:.1})
                self.metrics = [bad]
                with self.assertRaisesRegex(ValueError,field):
                    self.results()


def synthetic_microbench(barrier=2e-5,launch=3e-6,rate=1.5e14,a2a_latency=4e-5):
    """A node microbenchmark with every probe present, in the receipt format of microbench.py."""
    from microbench import PEER_CTAS, SEVEN_PEER_CTAS, RATE_SHAPE
    m,k,n = RATE_SHAPE
    readings = [dict(kind="peer_copy",peers=1,ctas=c,bytes=2**26,seconds=2**26/min(3.5e9*c,1.9e11)) for c in PEER_CTAS]
    readings += [dict(kind="peer_read_7",peers=7,ctas=7*c,bytes=7*2**23,seconds=7*2**23/min(3.5e9*7*c,2.3e11))
                 for c in SEVEN_PEER_CTAS]
    readings += [dict(kind="hbm_copy",bytes=2**27,seconds=2**27/1.6e12),dict(kind="launch",seconds=3e-5),
                 dict(kind="graph_launch",launches=8,seconds=launch),dict(kind="peer_barrier",seconds=barrier)]
    readings += [dict(kind="consumer_rate",g=g,ctas=16*(24//g),flops=2*m*k*n,seconds=2*m*k*n/rate) for g in (1,2)]
    readings += [dict(kind="all_to_all",m=2**i,bytes=8*2**i*4096,remote_bytes=7*2**i*4096,
                      seconds=a2a_latency+7*2**i*4096/2e11) for i in range(11)]
    return dict(rank_measurements=[readings]*8)


class TestCalibratedPrediction(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)

    def seal(self, micro, name="cal"):
        import subprocess
        import sys
        source = self.path/f"{name}-microbench.json"
        source.write_text(json.dumps(micro))
        code = subprocess.run([sys.executable,"-B",str(ROOT/"calibrate_node.py"),"--microbench",str(source),
                               "--out",str(self.path/name)],capture_output=True,text=True).returncode
        return code,source,self.path/name

    def test_node_parameters_seal_a_second_prediction(self):
        from analyze import predicted_classes
        from replay import check_calibrated
        code,source,out = self.seal(synthetic_microbench())
        self.assertEqual(code,0)
        params = json.loads((out/"params.json").read_text())
        self.assertEqual(params["source"],"microbench")
        self.assertAlmostEqual(params["F_direct"],4e-5)
        self.assertAlmostEqual(params["G_launch"],3e-6)
        self.assertAlmostEqual(params["F_staged"],1e-6)
        self.assertAlmostEqual(params["mma_peak"]/1.5e14,1,places=6)
        self.assertAlmostEqual(params["a2a"]["latency"]/4e-5,1,places=4)
        self.assertAlmostEqual(params["b_hbm"]/1.6e12,1,places=6)
        check_calibrated(out,DESIGN,source)
        rows,choices = load_prediction(out,DESIGN,"balanced")
        self.assertEqual(len(choices),11)
        self.assertEqual(len(predicted_classes(out)),44)
        # The scenario seal is untouched and differs: launches no longer cost 40 us each.
        scenario = {(r["unit_id"],r["arm_id"]):float(r["predicted_ms"]) for r in read(ROOT/"preboot_prediction"/"predictions.csv")}
        node = {(r["unit_id"],r["arm_id"]):float(r["predicted_ms"]) for r in read(out/"predictions.csv")}
        self.assertLess(node["GLM_M0001","D_Q1_G2"],scenario["GLM_M0001","D_Q1_G2"]/3)
        other = self.path/"other.json"
        other.write_text(json.dumps(synthetic_microbench(barrier=3e-5)))
        with self.assertRaisesRegex(ValueError,"does not belong"):
            check_calibrated(out,DESIGN,other)
        (out/"params.json").write_text((out/"params.json").read_text().replace("microbench","edited"))
        with self.assertRaisesRegex(ValueError,"does not belong"):
            check_calibrated(out,DESIGN,source)

    def test_a_failed_required_probe_leaves_only_the_scenario_prediction(self):
        micro = synthetic_microbench()
        micro["rank_measurements"] = [[dict(kind="consumer_rate",g=r["g"],error="CompilationError()")
                                       if r["kind"]=="consumer_rate" else r for r in micro["rank_measurements"][0]]]*8
        code,_,out = self.seal(micro,"failed")
        self.assertEqual(code,3)
        receipt = json.loads((out/"CALIBRATION.json").read_text())
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("mma",receipt["reason"])
        self.assertFalse((out/"PREDICTION_SEAL.json").exists())


class TestSecondaryR1(unittest.TestCase):
    def effects(self, classes):
        return [dict(comparison_id=c,unit_id=u,outcome=k) for (c,u),k in classes.items()]

    def test_reality_equal_to_the_prediction_scores(self):
        from analyze import predicted_classes, secondary_r1
        classes = predicted_classes(ROOT/"preboot_prediction")
        rows = {r["comparison_id"]:r for r in secondary_r1(DESIGN,self.effects(classes),classes)}
        self.assertEqual(set(rows),{"Ba_q1","Ba_q4","Bb_q1","Bb_q4"})
        for cid,(lo,hi) in (("Ba_q1",(4,16)),("Ba_q4",(16,128))):
            self.assertEqual((rows[cid]["measured_bracket"]["m_low"],rows[cid]["measured_bracket"]["m_high"]),(lo,hi))
            self.assertEqual(rows[cid]["bracket_verdict"],"contained")
            self.assertEqual((rows[cid]["decided"],rows[cid]["class_agree"]),(11,11))
        self.assertEqual(rows["Bb_q1"]["bracket_verdict"],"agree_none")

    def test_bracket_verdicts(self):
        from analyze import bracket, score_bracket
        W,L,T,U = "candidate_wins","baseline_wins","tie","undetermined"
        predicted = bracket([(1,W),(2,W),(4,W),(8,T),(16,L),(32,L)])
        cases = (([(1,W),(2,W),(4,W),(8,U),(16,L),(32,L)],"contained"),
                 ([(1,W),(2,W),(4,W),(8,W),(16,L),(32,L)],"contained"),
                 ([(1,W),(2,W),(4,T),(8,T),(16,T),(32,L)],"overlap"),
                 ([(1,W),(2,L),(4,L),(8,L),(16,L),(32,L)],"disjoint"),
                 ([(1,L),(2,L),(4,L),(8,L),(16,L),(32,L)],"miss_existence"),
                 ([(1,L),(2,L),(4,L),(8,T),(16,W),(32,W)],"miss_direction"),
                 ([(1,W),(2,L),(4,W),(8,L),(16,L),(32,L)],"unscorable"),
                 ([(1,T),(2,U),(4,T),(8,T),(16,"incomplete"),(32,T)],"unscorable"))
        for seq,verdict in cases:
            with self.subTest(verdict=verdict,seq=seq):
                self.assertEqual(score_bracket(bracket(seq),predicted),verdict)
        none = bracket([(1,L),(2,L)])
        self.assertEqual(score_bracket(bracket([(1,L),(2,T)]),none),"agree_none")
        self.assertEqual(score_bracket(bracket([(1,W),(2,W)]),none),"miss_winner")


class TestPreflight(unittest.TestCase):
    def test_static_checks_and_stale_cpu_receipt(self):
        import sys
        import preflight
        self.assertEqual(preflight.design(),"glm-path-replay-2026-11-23.v2")
        self.assertIn("44 cell classes",preflight.prediction())
        preflight.files()
        sys.path.insert(0,str(preflight.REPO/"EDM-IR"))
        from edmir.benchmark.real_moe_config import collection_files
        root = preflight.REPO/"EDM-IR"
        hashes = {str(f.relative_to(root)):sha(f) for f in collection_files(root)}
        with tempfile.TemporaryDirectory() as d:
            receipt = Path(d)/"cpu-receipt.json"
            body = dict(schema="edmir.real-moe-cpu-receipt.v1",status="passed",collection_source_hashes=hashes)
            receipt.write_text(json.dumps(body))
            self.assertIn("7 collector files",preflight.collector(receipt))
            for broken in (dict(status="failed"),
                           dict(collection_source_hashes=dict(hashes,**{"tools/run_real_moe.py":"0"*64}))):
                receipt.write_text(json.dumps(dict(body,**broken)))
                with self.subTest(broken=list(broken)), self.assertRaises(ValueError):
                    preflight.collector(receipt)


class TestConsumerGraphs(unittest.TestCase):
    def fake_torch(self):
        from contextlib import contextmanager
        from types import SimpleNamespace

        state = SimpleNamespace(capturing=False,modes=[])

        class Graph:
            def __init__(self):
                self.replays = 0

            def replay(self):
                self.replays += 1

        @contextmanager
        def graph(g, capture_error_mode):
            state.modes.append(capture_error_mode)
            state.capturing = True
            try:
                yield
            finally:
                state.capturing = False
        cuda = SimpleNamespace(synchronize=lambda:None,CUDAGraph=Graph,graph=graph)
        return SimpleNamespace(cuda=cuda),state

    def test_all_ranks_capture_then_replay(self):
        from balanced import ConsumerGraphs
        torch,state = self.fake_torch()
        calls = []
        issue = lambda: calls.append(state.capturing)
        graphs = ConsumerGraphs(torch,lambda error:[error]*8)
        graphs.prepare({"D_Q1_G2":issue,"BNEW_Q1_G2":issue})
        # One warm launch per arm outside capture, then one captured launch per arm.
        self.assertEqual(calls,[False,False,True,True])
        self.assertEqual(state.modes,["thread_local"]*2)
        calls.clear()
        graphs.run("D_Q1_G2",issue)
        self.assertEqual(calls,[])
        self.assertEqual(graphs.graphs["D_Q1_G2"].replays,1)
        self.assertEqual((graphs.launch_mode("D_Q1_G2"),graphs.launch_mode("SLT")),("graph","eager"))

    def test_one_failing_rank_makes_every_rank_eager(self):
        from balanced import ConsumerGraphs
        torch,_ = self.fake_torch()
        calls = []
        issue = lambda: calls.append(1)
        graphs = ConsumerGraphs(torch,lambda error:[None]*7+["RuntimeError('capture')"])
        graphs.prepare({"D_Q1_G2":issue,"BNEW_Q1_G2":issue})
        self.assertEqual((graphs.modes,graphs.graphs),({"consumer":"eager"},{}))
        calls.clear()
        graphs.run("D_Q1_G2",issue)
        self.assertEqual(calls,[1])
        # Once eager, later arms in the process stay eager so the arms share one mode.
        graphs.prepare({"D_Q4_G2":issue})
        self.assertEqual(graphs.launch_mode("D_Q4_G2"),"eager")
        no_channel = ConsumerGraphs(torch,None)
        no_channel.prepare({"D_Q1_G2":issue})
        self.assertEqual(no_channel.modes["consumer"],"eager")

    def test_native_group_fails_alone(self):
        from balanced import ConsumerGraphs
        torch,_ = self.fake_torch()
        outcome = iter([[None]*8,["RuntimeError('native capture')"]+[None]*7])
        graphs = ConsumerGraphs(torch,lambda error:next(outcome))
        graphs.prepare({"D_Q1_G2_R":lambda:None,"BNEW_Q1_G2_R":lambda:None})
        graphs.prepare({"NATIVE":lambda:None},group="native")
        self.assertEqual(graphs.modes,{"consumer":"graph","native":"eager"})
        self.assertEqual([graphs.launch_mode(a) for a in ("D_Q1_G2_R","BNEW_Q1_G2_R","NATIVE")],
                         ["graph","graph","eager"])


class TestSnapshot(unittest.TestCase):
    def test_source_tile_table_handles_expert_tail_and_empty_ranges(self):
        self.assertEqual(tiles([0,1,65,0,2]),
                         [(0,1,1),(1,66,2),(65,66,2),(66,68,4)])

    def test_bf16_route_replay_and_numeric_gate(self):
        import torch

        states = [torch.arange(64,dtype=torch.bfloat16).view(4,16)+rank for rank in range(8)]
        packed = [torch.tensor([[0,9,17,27],[1,8,29,41],[1,8,51,59],[3,13,19,29]])
                  for _ in range(8)]
        weight = (torch.arange(8*16*16).reshape(8,16,16) % 3).to(torch.bfloat16)
        payload = dict(hidden=states[0], expert_weights=weight, ids=packed[0],
                       router_weights=torch.full((4,4),.25))
        validate_tensors(payload,4,3,16,16)
        layout_ = layout([v.tolist() for v in packed],[3,2,1,0,2,2,1,1])
        self.assertEqual(layout_[0][0][1],[(1,0),(2,0)])
        self.assertEqual(sum(len(e) for r in layout_ for d in r for e in d),48)
        ref = reference(states,weight,layout_,0)
        self.assertEqual(ref.shape,(12,16))
        actual = []
        for src,source in enumerate(layout_):
            for ex,rows in enumerate(source[0]):
                if rows:
                    x = states[src][[row for row,_ in rows]].float()
                    w = weight[ex].float()
                    actual.append(sum(xs @ ws.T for xs,ws in zip(x.chunk(4,1),w.chunk(4,1))).bfloat16())
        bf16 = torch.cat(actual)
        err,nans = correctness(bf16,ref)
        self.assertLessEqual(err,.02)
        self.assertEqual(nans,0)
        bf16[0,0] = torch.inf
        self.assertEqual(correctness(bf16,ref),(None,1))
        broken = copy.deepcopy(payload)
        broken["ids"][0,0] = broken["ids"][0,1]
        with self.assertRaisesRegex(ValueError,"invalid expert"):
            validate_tensors(broken,4,3,16,16)

    def test_captured_alignment_and_balanced_seed(self):
        import torch
        from balanced import input_digest

        payload = dict(sorted_token_ids=torch.tensor([0,2,4,4,1,3,4,4],dtype=torch.int32),
                       expert_ids=torch.tensor([0,-1],dtype=torch.int32),
                       num_tokens_post_padded=torch.tensor([8],dtype=torch.int32))
        validate_alignment(payload,{"BLOCK_SIZE_M":4},4)
        payload["sorted_token_ids"][2] = 5
        with self.assertRaisesRegex(ValueError,"alignment extent"):
            validate_alignment(payload,{"BLOCK_SIZE_M":4},4)
        digests = [input_digest(torch,"GLM_M0001",rep,(1,32,16)) for rep in ("rep1","rep1","rep2")]
        self.assertEqual(digests[0],digests[1])
        self.assertNotEqual(digests[0],digests[2])

    def test_receipt_phase_and_rank_consensus(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            snap = base/"GLM_DECODE"
            snap.mkdir()
            for rank in range(8):
                record = dict(schema="edmir.glm-gate-up-snapshot.v2",origin="real_vllm",snapshot=snap.name,
                    model_revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",rank=rank,layer=23,
                    phase="pure_decode",phase_ordinal=8,step=9,run_id="fixture",padded_sizes=[1]*8,
                    expert_map=[i%8 if i//8==rank else -1 for i in range(64)],tokens=[{"phase":"decode"}],
                    source_lock_sha256=DUMMY,request_sha256=DUMMY,snapshot_sha256=DUMMY,
                    native_config={"BLOCK_SIZE_M":16},native_tune={"kind":"default"},
                    native_strides={"A":[2048,1],"B":[3072*2048,2048,1],"C":[4*3072,3072,1]},
                    gate_up_accumulator="float32")
                path = snap/f"rank-{rank}.json"
                path.write_text(json.dumps(record))
            self.assertEqual(len(load_receipts(base,DESIGN,snap.name)),8)
            path.write_text(json.dumps(dict(record,phase_ordinal=7)))
            with self.assertRaisesRegex(ValueError,"differs from design"):
                load_receipts(base,DESIGN,snap.name)


if __name__ == "__main__":
    unittest.main()
