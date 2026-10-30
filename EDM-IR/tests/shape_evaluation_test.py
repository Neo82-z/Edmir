from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from edmir.compiler.shape_decision_model import (FEATURES, features, objective,
    fit, evaluate, predict, validate_rows)
from edmir.compiler.shared_candidate_selection import parameter_count
from edmir.compiler.tile_execution_model import describe_execution
from edmir.compiler.execution_policy import UNIFORM_PEER_POLICY
from edmir.benchmark.moe_shape_profile import router_histogram, summarize


def numerical_rows():
    rows=[]
    for m in (16,24,40):
        for name,t in (("host",10+m/20),("splitk2",11+m/40)):
            s=describe_execution((4,m,512,128),name,environment_id="unit_test")
            rows.append({"shape":[4,m,512,128],"execution_context":"unit_test",
                         "candidate_id":name,"is_host":name=="host","features":features(s),
                         "latency_us":t,"measurement_kind":"observed","source_id":"synthetic_unit_test",
                         "split":"train"})
    return rows


def router_fixture():
    selections=[[[0,1],[0,2]],[[1,3]],[],[[2,3]]]
    return {"origin":"synthetic_test","diagnostic_only":True,"world_size":4,
            "num_experts":4,"top_k":2,"expected_layers":[0],
            "provenance":{k:"synthetic_unit_test" for k in ("model_id","model_revision","engine_revision",
                "run_id","workload_id","request_corpus_sha256","dtype","sampling_rule","router_hook_source")},
            "projections":[{"name":"gate_up","K":4,"N":8},{"name":"down","K":4,"N":4}],
            "records":[{"step":0,"layer":0,"phase":"prefill","source_rank":r,
                        **router_histogram(x,num_experts=4,top_k=2,diagnostic_only=True)}
                       for r,x in enumerate(selections)]}


class DecisionTests(unittest.TestCase):
    def test_gradients_with_missing_candidates_and_tied_best(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(2,3,4));target=np.array([[1.,1.,2.],[2.,1.,0.]])
        mask=np.array([[True,True,True],[True,True,False]])
        weights=np.array([[.5,.25,.25],[.5,.5,0.]])
        for width in (0,2):
            theta=rng.normal(size=parameter_count(4,width))*.15
            value,grad=objective(theta,x,target,mask,weights,width,1.,1.)
            for i in range(len(theta)):
                delta=np.zeros_like(theta);delta[i]=1e-6
                plus=objective(theta+delta,x,target,mask,weights,width,1.,1.)[0]
                minus=objective(theta-delta,x,target,mask,weights,width,1.,1.)[0]
                self.assertAlmostEqual(grad[i],(plus-minus)/2e-6,places=5)
            perm=[2,0,1]
            vp,gp=objective(theta,x[:,perm],target[:,perm],mask[:,perm],weights[:,perm],width,1.,1.)
            self.assertAlmostEqual(value,vp,places=12)
            np.testing.assert_allclose(grad,gp,atol=1e-12)
            changed=target.copy();changed[1,2]=10000
            self.assertEqual(value,objective(theta,x,changed,mask,weights,width,1.,1.)[0])

    def test_geometry_unknown_and_policy(self):
        s=describe_execution((8,64,512,128),"host_merged_cublaslt",environment_id="test")
        f=features(s)
        self.assertEqual(len(f),len(FEATURES));self.assertEqual(f[2],1)
        self.assertEqual(f[17:],[0.]*14)
        s=describe_execution((8,64,512,128),"splitk2",environment_id="test",dependency_policy=UNIFORM_PEER_POLICY)
        self.assertEqual(features(s)[3],1)

    def test_local_fit_order_and_partitions(self):
        rows=numerical_rows(); model=fit(rows[:4],"mlp_decision")
        test=[{**r,"split":"evaluation"} for r in rows[4:]]
        result=evaluate(model,test)
        self.assertTrue(result["all_order_invariant"])
        self.assertEqual(predict(model,test),predict(model,list(reversed(test))))
        with self.assertRaises(ValueError):evaluate(model,[{**r,"split":"evaluation"} for r in rows[:2]])
        wrong=deepcopy(test);wrong[0]["execution_context"]="different_binary"
        with self.assertRaises(ValueError):predict(model,wrong)
        self.assertEqual(len(model["theta"]),296)

    def test_missing_fake_duplicate_and_context_labels_rejected(self):
        rows=numerical_rows()
        for field,value in (("latency_us",0),("latency_us",float("nan")),("split","evaluation"),
                            ("measurement_kind","predicted"),("execution_context","other")):
            wrong=deepcopy(rows);wrong[0][field]=value
            with self.assertRaises(ValueError):validate_rows(wrong,split="train")
        with self.assertRaises(ValueError):validate_rows(rows+rows[:1],split="train")

    def test_training_candidate_permutation(self):
        rows=numerical_rows()
        a,b=fit(rows,"linear_decision"),fit(list(reversed(rows)),"linear_decision")
        self.assertEqual(a["theta"],b["theta"])
        self.assertEqual(a["means"],b["means"])


class MoETests(unittest.TestCase):
    def test_actual_counts_and_coverage_denominator(self):
        f=router_fixture();r=summarize(f,evaluated_shapes=[(4,2,4,8)])
        self.assertEqual(r["routed_rows"],8)
        self.assertEqual(r["logical_totals"],{"calls":8,"rows":16,"flops":768})
        self.assertEqual(r["logical_exact_MKN_coverage"]["calls"],.5)
        self.assertAlmostEqual(r["logical_exact_MKN_coverage"]["flops"],2/3)
        self.assertEqual(sum(r["source_expert_M_histogram"].values()),16)
        self.assertIsNone(r["kernel_shapes"]);self.assertIsNone(r["measured_time_coverage"])
        self.assertEqual(r["origin"],"synthetic_test")

    def test_missing_ranks_layers_duplicates_and_conservation(self):
        cases=[];f=router_fixture();f["records"].pop();cases.append(f)
        f=router_fixture();f["records"].append(f["records"][0]);cases.append(f)
        f=router_fixture();f["records"][0]["expert_counts"][0]+=1;cases.append(f)
        f=router_fixture();f["expected_layers"].append(1);cases.append(f)
        for f in cases:
            with self.assertRaises(ValueError):summarize(f)

    def test_capture_outside_timing_and_unique_topk(self):
        with self.assertRaises(ValueError):router_histogram([[0,1]],num_experts=4,top_k=2,diagnostic_only=False)
        with self.assertRaises(ValueError):router_histogram([[0,0]],num_experts=4,top_k=2,diagnostic_only=True)
        with self.assertRaises(ValueError):router_histogram([[0,4]],num_experts=4,top_k=2,diagnostic_only=True)

    def test_record_order_and_empty_experts(self):
        f=router_fixture();a=summarize(f)
        f["records"].reverse();b=summarize(f)
        self.assertEqual(a["logical_expert_shapes"],b["logical_expert_shapes"])
        f=router_fixture()
        for r in f["records"]:r.update(token_count=0,expert_counts=[0]*4)
        z=summarize(f)
        self.assertEqual(z["empty_expert_step_layers"],4)
        self.assertIsNone(z["logical_exact_MKN_coverage"]["calls"])


if __name__=="__main__":unittest.main()
