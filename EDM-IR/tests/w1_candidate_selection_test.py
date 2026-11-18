from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from edmir.compiler import w1_candidate_selection as s
from edmir.compiler.target import FusedScheduleContract
from prepare_selector_gpu import program_artifact, implementation_id
from run_w1_gpu import validate_rows


INPUTS={str(seed):dict(input_sha256=f"input-{seed}",reference_sha256=f"ref-{seed}") for seed in (1,2,3)}


def fixture(shape, family, status="passed"):
    row=dict(shape=shape,shape_tag=str(shape),candidate_id=family,execution_context="synthetic-W1-test-only",
             implementation_id="test-"+family,source_id="test-build",precision=s.precision(family),
             error_contract=s.ERROR_CONTRACT,phase="measure",sm_count=108,checks=[],
             split="train",measurement_kind="observed",latency_us=100. if family in s.HOSTS else 80.)
    row["features"]=s.original_geometry_features(row)
    if status!="unknown":
        for seed,inp in INPUTS.items():
            row["checks"].append(dict(seed=int(seed),input=inp,passed=(status!="failed" or seed!="3"),
                checked_elements=shape[0]**2*shape[1]*shape[3],contract_sha256=s.contract(row),record_sha256="test-only"))
    return row


class AdmissionTests(unittest.TestCase):
    def setUp(self):self.row=fixture([8,512,8192,512],"splitk8")

    def test_all_seeds_and_failure_dominates(self):
        self.assertEqual(s.admission(self.row,INPUTS),"passed")
        self.row["checks"].pop()
        self.assertEqual(s.admission(self.row,INPUTS),"unknown")
        self.row["checks"][0]["passed"]=False
        self.assertEqual(s.admission(self.row,INPUTS),"failed")

    def test_binding_and_reference(self):
        for field,value in (("implementation_id","different-kernel"),("execution_context","other-run"),
                            ("source_id","other-build")):
            row=deepcopy(self.row);row[field]=value
            with self.assertRaisesRegex(ValueError,"bound"):s.admission(row,INPUTS)
        row=deepcopy(self.row);row["checks"][0]["input"]={}
        with self.assertRaisesRegex(ValueError,"identity"):s.admission(row,INPUTS)

    def test_sampled_duplicate_wrong_precision(self):
        row=deepcopy(self.row);row["checks"][0]["checked_elements"]//=64
        with self.assertRaisesRegex(ValueError,"sampled"):s.admission(row,INPUTS)
        row=deepcopy(self.row);row["checks"].append(row["checks"][0])
        with self.assertRaisesRegex(ValueError,"duplicate"):s.admission(row,INPUTS)
        row=deepcopy(self.row);row["precision"]["input"]="bf16"
        with self.assertRaisesRegex(ValueError,"unsupported"):s.admission(row,INPUTS)

    def test_new_precision_does_not_inherit_status(self):
        row=fixture([8,512,8192,512],"splitk8_fp32reduce","unknown")
        self.assertEqual(s.admission(row,INPUTS),"unknown")
        row["checks"]=self.row["checks"]
        with self.assertRaisesRegex(ValueError,"bound"):s.admission(row,INPUTS)

    def test_q8_is_not_globally_banned(self):
        bad=fixture([8,512,8192,512],"splitk8","failed")
        good=fixture([8,256,8192,512],"splitk8")
        self.assertEqual(s.admission(bad,INPUTS),"failed")
        self.assertEqual(s.admission(good,INPUTS),"passed")

    def test_pair_phase_refused(self):
        self.row["phase"]="pair"
        with self.assertRaisesRegex(ValueError,"paired"):s.admission(self.row,INPUTS)


class GeometryTests(unittest.TestCase):
    def test_grouping_keeps_grid_warps_but_changes_block_pressure(self):
        a=s.coupling(fixture([8,64,8192,1024],"splitk8"))["geometry"]
        b=s.coupling(fixture([8,64,8192,1024],"splitk8_groupedn2"))["geometry"]
        self.assertEqual(a["ctas"],2*b["ctas"])
        self.assertEqual(a["grid_warps_per_sm"],b["grid_warps_per_sm"])
        self.assertEqual(b["warps_per_cta"],2*a["warps_per_cta"])
        self.assertGreater(b["operand_staging_lower_bound_bytes"],a["operand_staging_lower_bound_bytes"])
        self.assertIsNone(b["observed_residency"])

    def test_q_changes_depth_and_feedback(self):
        a=s.coupling(fixture([8,256,8192,512],"splitk4"))["geometry"]
        b=s.coupling(fixture([8,256,8192,512],"splitk8"))["geometry"]
        self.assertEqual(a["k_iterations_per_partition"],2*b["k_iterations_per_partition"])
        self.assertGreater(b["serial_feedback_bytes"],a["serial_feedback_bytes"])

    def test_library_geometry_is_unknown(self):
        c=s.coupling(fixture([8,64,8192,1024],"host_merged_cublaslt"))
        self.assertIsNone(c["geometry"])
        self.assertEqual(c["program_gemm_launches"],8)
        self.assertEqual(c["host_nccl_groups"],56)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.shapes=([8,64,8192,1024],[8,512,8192,512])
        self.rows=[fixture(shape,f) for shape in self.shapes for f in ("host","host_merged_cutlass","direct","splitk4")]
        self.inputs={str(shape):INPUTS for shape in self.shapes}
        self.model=s.fit(self.rows,self.inputs,variant="coupled_geometry")

    def test_permutation_labels_and_ties(self):
        queries=deepcopy(self.rows[:4])
        expected=s.predict(self.model,queries,INPUTS)
        for row in queries:row["latency_us"]=-1e99;row["measured_best"]=True
        self.assertEqual(expected,s.predict(self.model,list(reversed(queries)),INPUTS))
        tied=deepcopy(self.model);tied["theta"]=[0.]*len(tied["theta"])
        self.assertEqual(s.predict(tied,queries,INPUTS)["chosen"],"direct")

    def test_invalid_never_scored_and_new_identity_unsupported(self):
        queries=deepcopy(self.rows[:4]);queries[0]["checks"][0]["passed"]=False
        pred=s.predict(self.model,queries,INPUTS)
        self.assertNotIn("host",pred["scores_us"])
        row=fixture(self.shapes[0],"splitk8")
        pred=s.predict(self.model,[row],INPUTS)
        self.assertIsNone(pred["chosen"])
        self.assertFalse(pred["complete_eligible_coverage"])

    def test_training_rejects_bad_and_missing_labels(self):
        for value in (None,-1.,float("nan")):
            rows=deepcopy(self.rows);rows[0]["latency_us"]=value
            with self.assertRaises((ValueError,TypeError)):s.fit(rows,self.inputs)
        rows=deepcopy(self.rows);rows[0]["checks"][0]["passed"]=False
        with self.assertRaisesRegex(ValueError,"no training label"):s.fit(rows,self.inputs)

    def test_feature_poison_and_bound(self):
        row=deepcopy(self.rows[0]);row["features"][0]=0.
        with self.assertRaisesRegex(ValueError,"source-derived"):s.predict(self.model,[row],INPUTS)
        with self.assertRaisesRegex(ValueError,"unique"):s.predict(self.model,self.rows[:1]*10000,INPUTS)


class LoweringTests(unittest.TestCase):
    def test_collector_requires_new_precision_on_failures_too(self):
        family="splitk8_fp32reduce"
        row=dict(family=family,input_seed=1,checked_elements=8*8*64*512,correctness_passed=False,
                 failure="numerical",median_ms=None)
        job=dict(seed=1,phase="check")
        with self.assertRaisesRegex(ValueError,"precision"):validate_rows([row],job,[family],[8,64,8192,512],{})
        row.update(precision=s.precision(family),error_contract=s.ERROR_CONTRACT,
                   implementation_id=implementation_id(family,"prepared",dependency_policy="uniform_immutable_peer_v1"))
        validate_rows([row],job,[family],[8,64,8192,512],{})
        row["implementation_id"]=implementation_id("splitk8","prepared",dependency_policy="uniform_immutable_peer_v1")
        with self.assertRaisesRegex(ValueError,"identity"):validate_rows([row],job,[family],[8,64,8192,512],{})

    def test_old_manifest_default_and_new_identity(self):
        self.assertEqual(FusedScheduleContract(2).to_dict(),{"n_tiles_per_cta":2})
        original=program_artifact([8,512,8192,512],"splitk8")
        fp32=program_artifact([8,512,8192,512],"splitk8_fp32reduce")
        self.assertNotIn("kSerialFP32",original.files["edmir_program.hpp"])
        self.assertIn("kSerialFP32",fp32.files["edmir_program.hpp"])
        self.assertNotEqual(implementation_id("splitk8","prepared"),implementation_id("splitk8_fp32reduce","prepared"))
        with self.assertRaises(ValueError):program_artifact([8,512,8192,512],"direct_fp32reduce")
        with self.assertRaises(ValueError):program_artifact([8,512,8192,512],"host_fp32reduce")


if __name__=="__main__":unittest.main()
