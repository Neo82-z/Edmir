"""Independent geometry enumeration, dependency behavior and partition checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"EDM-IR"), str(ROOT/"EDM-IR/tools")]
from edmir.compiler.execution_ledger import execution_ledger, digest
from edmir.compiler.execution_policy import LEGACY_POLICY, UNIFORM_PEER_POLICY
from edmir.compiler.tile_execution_model import (describe_execution, structure_from_ledger,
                                                packing_reference, conditional_dependency_span)
from edmir.compiler.structured_validation import validate_partition_records, development_folds
from edmir.compiler.scale_aware_prediction import basis


def example_ledger(policy=UNIFORM_PEER_POLICY):
    return execution_ledger((8,64,4096,2048), "splitk4_groupedn2", environment_id="unit-test",
                            metadata_mode="prepared", dependency_policy=policy)


def reseal_graph(ledger):
    ledger["identity"]["execution_graph_digest"] = digest(ledger["execution_graph"])
    ledger["identity_sha256"] = digest(ledger["identity"])


class TileExecutionTests(unittest.TestCase):
    def test_partial_tiles_match_independent_enumeration(self):
        for m,n in ((32,128),(65,264),(512,2048)):
            for family,q,g in (("direct",1,1),("splitk8",8,1),("splitk16_groupedn2",16,2)):
                d=describe_execution((8,m,8192,n),family,environment_id="unit-test")["geometry"]
                slots=valid=ctas=edges=0
                for lo_m in range(0,m,64):
                    for lo_n in range(0,n,128*g):
                        valid += min(m-lo_m,64)*min(n-lo_n,128*g)
                        slots += 64*128*g
                        for part in range(q):
                            ctas += 1; edges += part > 0
                self.assertEqual((d["valid_output_elements"],d["tile_output_slots"],d["ctas"]),(valid,slots,ctas))
                self.assertEqual(d["epilogue_order"]["ordered_edges"],edges)
                self.assertEqual(d["valid_output_fraction"],valid/slots)

    def test_new_representation_distinguishes_old_alias(self):
        a,b=(8,32,8192,8192),(8,64,8192,4096);f="splitk8_groupedn2"
        self.assertEqual(basis(a,f),basis(b,f))
        left,right=(describe_execution(s,f,environment_id="unit-test")["geometry"] for s in (a,b))
        self.assertEqual((left["ctas"],right["ctas"]),(256,128))
        self.assertEqual((left["valid_output_fraction"],right["valid_output_fraction"]),(.5,1))
        self.assertNotEqual(left,right)

    def test_round_boundaries_and_unknown_residency(self):
        self.assertEqual([packing_reference(n,108)["rounds"] for n in (107,108,109,216,217)],[1,1,2,2,3])
        self.assertEqual(packing_reference(216,108)["last_round_ctas"],108)
        ledger=example_ledger();d=structure_from_ledger(ledger)
        self.assertIsNone(d["geometry"]["resource_bound_reference"])
        self.assertIsNone(d["predicted_latency_us"])
        bound={"ctas_per_sm":2,"source":"hypothetical unit test, not a compiled report", "identity_sha256":ledger["identity_sha256"]}
        d=structure_from_ledger(ledger,residency_bound=bound)
        self.assertFalse(d["geometry"]["resource_bound_reference"]["observed_residency"])
        for bad in (0,-1,True,1.5,33):
            with self.assertRaises(ValueError):packing_reference(108,108,bad)
        bound["identity_sha256"]="wrong"
        with self.assertRaises(ValueError):structure_from_ledger(ledger,residency_bound=bound)

    def test_merged_host_geometry_and_library_unknowns(self):
        per=describe_execution((8,64,4096,2048),"host",environment_id="unit-test")
        merged=describe_execution((8,64,4096,2048),"host_merged_cutlass",environment_id="unit-test")
        lib=describe_execution((8,64,4096,2048),"host_merged_cublaslt",environment_id="unit-test")
        self.assertEqual(per["geometry"]["gemm_mnk"][0],64)
        self.assertEqual(merged["geometry"]["gemm_mnk"][0],512)
        self.assertEqual((per["program"]["gemm_launches"],merged["program"]["gemm_launches"]),(64,8))
        self.assertIsNone(lib["geometry"])

    def test_uniform_dag_has_eight_parallel_destination_chains(self):
        ledger=example_ledger()
        durations={op["id"]:1. for op in ledger["execution_graph"]["execution"]["ops"]}
        result=conditional_dependency_span(ledger,durations,identity_sha256=ledger["identity_sha256"],source="synthetic unit weights")
        # Independently: 8 destinations in parallel, 8 serial launches each.
        self.assertEqual(result["dependency_span_us"],8)
        self.assertIsNone(result["program_wall_time_us"])
        legacy=example_ledger(LEGACY_POLICY)
        old=conditional_dependency_span(legacy,durations,identity_sha256=legacy["identity_sha256"],source="synthetic unit weights")
        self.assertEqual(old["dependency_span_us"],33)
        shuffled=deepcopy(ledger)
        for key in ("ops","dependencies"):shuffled["execution_graph"]["execution"][key].reverse()
        reseal_graph(shuffled)
        other=conditional_dependency_span(shuffled,dict(reversed(list(durations.items()))),identity_sha256=shuffled["identity_sha256"],source="synthetic unit weights")
        self.assertEqual(other["dependency_span_us"],result["dependency_span_us"])

    def test_missing_invalid_and_cyclic_durations_fail_closed(self):
        ledger=example_ledger();identity=ledger["identity_sha256"]
        result=conditional_dependency_span(ledger,{},identity_sha256=identity,source="missing")
        self.assertIsNone(result["dependency_span_us"])
        self.assertEqual(len(result["missing_nodes"]),64)
        first=ledger["execution_graph"]["execution"]["ops"][0]["id"]
        for bad in (-1,float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):conditional_dependency_span(ledger,{first:bad},identity_sha256=identity,source="test")
        with self.assertRaises(ValueError):conditional_dependency_span(ledger,{},identity_sha256="wrong",source="test")
        cyclic=deepcopy(ledger);edges=cyclic["execution_graph"]["execution"]["dependencies"]
        edges.append({"src":edges[0]["dst"],"dst":edges[0]["src"]});reseal_graph(cyclic)
        with self.assertRaises(ValueError):conditional_dependency_span(cyclic,{},identity_sha256=cyclic["identity_sha256"],source="cycle")


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.required={"p8-test":("host","fused")}
        self.rows=[{"shape":[8,m,8192,512],"execution_context":"p8-test","candidate_id":c,
                    "replicate_id":r,"split":"train"}
                   for m in (64,256,512) for c in ("host","fused") for r in ("0","1","2")]

    def test_whole_shape_folds_are_order_invariant(self):
        a=development_folds(self.rows,self.required)
        self.assertEqual(a,development_folds(self.rows[::-1],self.required))
        self.assertEqual(len(a),3)
        for fold in a:
            self.assertEqual(fold["audit"]["shapes_per_split"],{"train":2,"calibration":0,"evaluation":1})
            self.assertEqual(sum(r["split"]=="evaluation" for r in fold["records"]),6)

    def test_leakage_duplicate_and_incomplete_candidate_blocks_rejected(self):
        for rows in (self.rows+[self.rows[0]], self.rows[1:],
                     [{**self.rows[0],"split":"evaluation"}]+self.rows[1:]):
            with self.assertRaises(ValueError):validate_partition_records(rows,self.required)
        with self.assertRaises(ValueError):validate_partition_records([{**r,"latency":1.} for r in self.rows],self.required)

    def test_input_budget_is_checked_before_fold_expansion(self):
        with self.assertRaises(ValueError):development_folds(self.rows*1000,self.required)
        mixed=[{**r,"shape":[4,*r["shape"][1:]]} for r in self.rows]+self.rows
        with self.assertRaises(ValueError):validate_partition_records(mixed,self.required)


if __name__ == "__main__":
    unittest.main()
