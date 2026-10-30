"""Semantic tests for finer counts and the explicit M/q/g cost hypothesis."""
import copy
import json
from itertools import permutations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
import audit_fine_mechanisms as audit
from clean_mechanism_data import clean_collection
from evaluate_mechanism_prediction import DEFAULT_DATA, comparisons, folds, fused_leave_M_out
from edmir.compiler.mechanism_prediction import (
    MechanismModel, basis, detailed_geometry, geometry, select_prediction,
)


class FineGeometryTests(unittest.TestCase):
    def test_shared_allocation_is_dynamic_and_adds_calls(self):
        a = detailed_geometry((4,64,8192,512),"splitk8")
        b = detailed_geometry((4,64,8192,512),"splitk8_groupedn2")
        self.assertEqual(a["dynamic_smem_optin_calls_per_execution"],0)
        self.assertEqual(b["dynamic_smem_optin_calls_per_execution"],16)
        self.assertEqual(a["cudaGetLastError_calls_per_execution"],16)
        self.assertEqual(a["combined_resident_block_upper_bound"],3)
        self.assertEqual(b["combined_resident_block_upper_bound"],1)
        self.assertIsNone(a["achieved_occupancy"])
        self.assertIsNone(b["actual_smem_carveout"])

    def test_equal_total_warps_do_not_imply_equal_sm_spread(self):
        a = detailed_geometry((4,64,8192,512),"splitk8")
        b = detailed_geometry((4,64,8192,512),"splitk8_groupedn2")
        self.assertEqual(a["grid_warps"],b["grid_warps"])
        self.assertEqual(a["possible_active_SM_upper_bound"],32)
        self.assertEqual(b["possible_active_SM_upper_bound"],16)

    def test_partition_footprint_and_total_reads_are_not_confused(self):
        shape=(4,192,3072,1536)
        a,b=[detailed_geometry(shape,"splitk"+str(q)) for q in (2,4)]
        self.assertEqual(a["partition_A_bytes"],2*b["partition_A_bytes"])
        self.assertEqual(a["partition_B_bytes"],2*b["partition_B_bytes"])
        self.assertEqual(a["logical_B_replay_bytes"],b["logical_B_replay_bytes"])
        self.assertEqual(geometry(shape,"splitk2")["logical_A_replay_bytes"],
                         geometry(shape,"splitk4")["logical_A_replay_bytes"])
        self.assertEqual(a["ordered_boundaries_per_output_tile"],1)
        self.assertEqual(b["ordered_boundaries_per_output_tile"],3)

    def test_edge_tiles_count_valid_B_elements_independently(self):
        # Independently enumerate valid rectangles and all K partitions.
        for m,k,n,q,g in ((65,1024,264,4,2),(192,3072,512,2,1)):
            family=f"splitk{q}"+("_groupedn2" if g==2 else "")
            expected=0
            for _ in range(0,m,64):
                for col in range(0,n,128*g):
                    for _ in range(q):
                        expected += 2*min(128*g,n-col)*(k//q)
            self.assertEqual(detailed_geometry((4,m,k,n),family)["logical_B_replay_bytes"],expected)

    def test_same_parameter_count_but_separate_row_and_column_tiles(self):
        a,b=(4,64,4096,2048),(4,256,4096,512)
        fa=dict(zip(*basis(a,"splitk2_groupedn2","row_service")))
        fb=dict(zip(*basis(b,"splitk2_groupedn2","row_service")))
        self.assertEqual(fa["ctas_over_108"],fb["ctas_over_108"])
        self.assertEqual(fa["logical_A_replay_MiB"],fb["logical_A_replay_MiB"])
        self.assertEqual(fa["grouped_depth_per_M_tile"],4*fb["grouped_depth_per_M_tile"])
        self.assertEqual(len(fa),12)
        self.assertEqual(len(basis(a,"splitk2_groupedn2","service_detail")[0]),12)

    def test_more_available_ctas_reduce_sm_deficit(self):
        names,a=basis((4,64,4096,512),"splitk2","service_detail")
        _,b=basis((4,512,4096,512),"splitk4","service_detail")
        i=names.index("SM_spread_deficit")
        self.assertGreater(a[i],0.)
        self.assertEqual(b[i],0.)


class FineEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.configs,_=clean_collection(DEFAULT_DATA)
        cls.primary=[r for r in cls.configs if r["primary_scope"]]
        cls.stable=[r for r in cls.primary if r["eligible_for_calibration"]]
        cls.resources=audit.audit_resources(DEFAULT_DATA,cls.configs)

    def test_resource_provenance_and_affine_confounding(self):
        self.assertEqual(self.resources["g_only_resource_design"]["rank"],2)
        self.assertEqual(self.resources["g_only_resource_design"]["column_count"],5)
        self.assertEqual(self.resources["resource_round_values"],[1])
        self.assertEqual([k["compiled_resources"]["REG"] for k in self.resources["kernels"]],[140,134])
        self.assertEqual([k["compiled_resources"]["SHARED"] for k in self.resources["kernels"]],[0,0])
        self.assertTrue(all(k["type_sizes"]["shared_storage_bytes"]>0 for k in self.resources["kernels"]))

    def test_resource_artifact_mutation_fails(self):
        original=audit.sha
        with patch.object(audit,"sha",side_effect=lambda p: "0"*64 if p.name=="resources-demangled.txt" else original(p)):
            with self.assertRaisesRegex(ValueError,"resource evidence changed"):
                audit.audit_resources(DEFAULT_DATA,self.configs)

    def test_new_fit_is_order_invariant_and_kkt_certified(self):
        a=MechanismModel.fit(self.stable,variant="row_service")
        b=MechanismModel.fit(self.stable[::-1],variant="row_service")
        self.assertEqual(a.to_dict(),b.to_dict())
        f=a.models["fused"]
        self.assertEqual(f["feature_count"],12)
        self.assertLess(f["kkt_relative_violation"],1e-8)
        self.assertLessEqual(f["enumerated_faces"],4096)
        self.assertTrue(all(v>=0 for v in f["coefficients_us"]))

    def test_previous_baseline_predictions_are_numerically_unchanged(self):
        archived=json.loads((ROOT/"experiments/2026-10-24-mechanism-prediction/final/calibration-predictions.json").read_text())["cutlass_depth"]
        model=MechanismModel.fit(self.stable,variant="cutlass_depth")
        for r in archived:
            actual=model.predict(r["shape"],r["family"],context_id=model.context_id)
            self.assertEqual(actual["predicted_us"],r["predicted_us"])

    def test_new_basis_heldout_label_noninterference(self):
        before,models=folds(self.primary,"row_service")
        changed=copy.deepcopy(self.primary)
        target=[4,64,8192,512]
        for r in changed:
            if r["shape"]==target:
                r["observed_us"]*=10
        after,changed_models=folds(changed,"row_service")
        a=next(m for m in models if list(m["held_out_shape"])==target)
        b=next(m for m in changed_models if list(m["held_out_shape"])==target)
        self.assertEqual(a,b)
        self.assertEqual([r["predicted_us"] for r in before if r["shape"]==target],
                         [r["predicted_us"] for r in after if r["shape"]==target])

    def test_leave_M_out_has_no_same_M_candidate_and_no_fake_host(self):
        result=fused_leave_M_out(self.primary,("row_service",))
        for fold in result["variants"]["row_service"]:
            self.assertTrue(all(s[1]!=fold["held_out_M"] for s in fold["model"]["training_shapes"]))
            self.assertTrue(all(not r["family"].startswith("host") for r in fold["predictions"]))
            decisions,_=comparisons(fold["predictions"])
            self.assertFalse(any(d["scope"]=="compiler_host_vs_fused" for d in decisions))


if __name__=="__main__":
    unittest.main()
