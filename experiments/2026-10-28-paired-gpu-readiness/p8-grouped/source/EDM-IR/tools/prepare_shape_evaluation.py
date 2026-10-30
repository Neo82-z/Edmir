#!/usr/bin/env python3
"""Seal an unused-shape protocol; fitted model/binary sealing is a later gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"EDM-IR"))
from edmir.compiler.execution_ledger import FAMILIES, source_provenance
from edmir.compiler.execution_policy import LEGACY_POLICY, UNIFORM_PEER_POLICY
from edmir.compiler.structured_validation import BASELINES
from edmir.compiler.tile_execution_model import describe_execution
from edmir.compiler.shape_decision_model import FEATURES, VARIANTS, SETTINGS, features
from edmir.compiler.measurement_dataset import load_identified_selector_dataset

DIAGNOSTIC=((8,64,4096,2048),(8,64,8192,1024),(8,256,8192,512),(8,512,8192,512))
BOUNDARY=((8,56,3072,776),(8,72,3072,776),(8,64,3072,3328),
          (8,64,3072,3584),(8,320,6144,768),(8,96,6144,2560))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inventory):
    exposure=json.loads(inventory.read_text())
    if (exposure["numeric_hits"] or exposure["skipped"]
        or exposure["proposed_shapes"] != [list(s) for s in BOUNDARY]):
        raise ValueError("resolve exposure audit before freezing any shapes")
    candidates=BASELINES+FAMILIES[1:]; provenance=source_provenance(); descriptions=[]
    for shape in DIAGNOSTIC+BOUNDARY:
        for name in candidates:
            host=name in BASELINES
            s=describe_execution(shape,name,environment_id="planned-p8-shape-closure-v1",
                metadata_mode="merged" if name.startswith("host_merged") else "prepared",
                dependency_policy=LEGACY_POLICY if host else UNIFORM_PEER_POLICY,provenance=provenance)
            descriptions.append({"partition":"diagnostic" if shape in DIAGNOSTIC else "boundary",
                                 "structure":s,"numerical_features":features(s)})
    comparisons=[]
    for a,b in ((DIAGNOSTIC[0],DIAGNOSTIC[1]),(DIAGNOSTIC[2],DIAGNOSTIC[3]),
                (BOUNDARY[0],BOUNDARY[1]),(BOUNDARY[2],BOUNDARY[3]),(BOUNDARY[4],BOUNDARY[5])):
        name="splitk4"
        ss=[next(d["structure"] for d in descriptions if d["structure"]["identity"]["shape"]==list(s)
                 and d["structure"]["identity"]["family"]==name) for s in (a,b)]
        fields=("m_tiles","n_tiles","output_tiles","ctas","partition_k_iterations", "logical_flops",
                "valid_output_fraction","last_m_tile_rows","last_n_tile_columns","one_cta_per_sm_reference")
        comparisons.append({"a":a,"b":b,"example_candidate":name,
            "changed_geometry":{k:[ss[0]["geometry"][k],ss[1]["geometry"][k]] for k in fields
                                if ss[0]["geometry"][k]!=ss[1]["geometry"][k]},
            "controlled_single_mechanism_experiment":False,"observed_latency_difference":None})
    preserved=["experiments/2026-10-21-shared-candidate-review/verified/decision.json",
               "experiments/2026-10-21-decision-loss-iterations/README.md",
               "experiments/2026-10-26-tile-model-revision/validated/plan.json"]
    model=json.loads((ROOT/"experiments/2026-10-25-moe-model-selection/comparison.json").read_text())["models"][0]
    _,cal,ev,_=load_identified_selector_dataset(ROOT/"experiments/2026-10-20-offline-predictor/data")
    archived_p8=sorted({tuple(key[-4:]) for dataset in (cal,ev) for key in dataset.validate()})
    return {"schema":"edmir.shape-evaluation-protocol.v1","date":"2026-10-28",
        "state":"protocol_frozen; model_binary_and_environment_not_yet_frozen",
        "shape_order":"P,M,K,N","diagnostic_shapes":DIAGNOSTIC,"frozen_boundary_shapes":BOUNDARY,
        "boundary_questions":["M=56/72 crosses a 64-row tile edge with a fixed N tail",
                              "q4/g1 CTA count 104/112 crosses the 108-CTA reference capacity; not actual residency",
                              "equal MN with different M/N aspect ratios; host/fused winner not preassigned"],
        "exposure_inventory_sha256":digest(inventory),
        "exposure_review":"no explicit M,K,N match in searchable repositories; old P4/P8 finite sweep axes checked; implicit/external exposure cannot be excluded by regex alone",
        "candidates":candidates,"comparisons":comparisons,
        "representativeness_reference_sets":{
            "archived_p8":{"shapes":archived_p8,"status":"measured_uniform_FP16; MKN coincidence only, not routed_BF16 execution coverage"},
            "diagnostic":{"shapes":DIAGNOSTIC,"status":"planned_new_runtime; known design shapes"},
            "boundary":{"shapes":BOUNDARY,"status":"planned_new_runtime; no measured labels"}},
        "training":{"features":FEATURES,"variants":VARIANTS,"settings":SETTINGS,
                    "parameters":{"linear":32,"width8_mlp":296},
                    "loss":"balanced log-time MSE + expected relative regret + observed-best log-gap hinge squared; AdamW decay",
                    "physical_prior":"not yet identified; current numerical head is empirical, service model remains separate",
                    "observed_candidates_only":True,"selection_weights_are_calibrated_probabilities":False,
                    "language_encoder":False,"Jev_RLCD_reproduced":False,
                    "development_only":"all previously measured or design-exposed shapes; whole-shape folds; no promotion from historical pilot",
                    "P4_P8_BF16_contexts_may_be_pooled":False},
        "evaluation":{"model_sha256":None,"binary_sha256":None,"environment_manifest_sha256":None,
                    "independent_evaluation_ready":False,"must_freeze_before_labels":True,
                    "candidate_order":"score canonical candidates; exact ties use host then stable candidate ID",
                    "missing_or_failed_candidates":"preserve status and failed numerical checks; incomplete surface cannot pass full-candidate oracle gate",
                    "selection_blocks":"separate block IDs from all verification blocks; k=1/2/3",
                    "verification_blocks":3,"shortlist_selection_blocks_per_candidate":1,
                    "metrics":["mean/max hard regret","host-to-fused errors","fused-to-host errors",
                               "score order invariance","shape-balanced latency MAPE","top1/2/3 oracle retention",
                               "amortized search+execution cost"],
                    "primary_gate":{"mean_regret_max":.01,"worst_regret_max":.03,
                        "no_worse_mean_and_max_than_training_fixed_order":True,
                        "no_worse_mean_and_max_than_strong_host_control":True,
                        "all_candidate_permutations_invariant":True,
                        "both_directions_observed_with_over_2_percent_margin":True},
                    "missing_direction":"mark bidirectional coverage unestablished; do not replace shapes after observing labels",
                    "failure_policy":"retain every shape and failed model; fresh labels become development-only after any retuning"},
        "budget":{"diagnostic_primary_blocks":4*13*3,"boundary_oracle_verification_blocks":6*13*3,
                  "boundary_shortlist_selection_blocks_max":6*3,"matched_pipeline_control_blocks":12,
                  "two_host_selection_control_blocks_max":12,"equal_budget_search_blocks_max":18,
                  "candidate_block_total_max":450,"new_hardware_or_model_sweeps":0,
                  "deadline":"estimate from first smoke run; preserve partial outputs, no automatic rental or overtime"},
        "moe":{"model_id":model["id"],"revision":model["revision"],"dtype":"BF16","EP":8,"TP":1,
               "num_experts":512,"top_k":10,"projections":[{"name":"gate_up","K":2048,"N":1024},
                                                            {"name":"down","K":512,"N":2048}],
               "projection_status":"config-derived logical projections; actual backend fusion/padding must be captured",
               "workloads":{"input_tokens":[512,4096],"concurrency":[1,128],"max_output_tokens":128},
               "sampling":"diagnostic run only; first three complete prefill and first three complete decode steps per workload, all MoE layers and EP ranks; record missing phases",
               "request_corpus_sha256":None,"engine_revision":None,"real_trace_sha256":None,
               "required_statistics":["source-expert token histogram including zeros","global expert token histogram",
                                      "logical projection dimensions/frequency by phase","actual backend padded MKN when observed",
                                      "exact logical MKN occurrence/row/FLOP coverage with denominator and sample scope"],
               "claims_forbidden_without_measurement":["kernel-time share from FLOPs","uniform M equals expert M",
                                                      "FP16 proxy implies BF16 end-to-end gain"],
               "runtime_hook_validated":False,"representation_validation_passed":False},
        "preserved_failures_sha256":{p:digest(ROOT/p) for p in preserved},
        "implementation_sha256":{str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),ROOT/"EDM-IR/edmir/compiler/shape_decision_model.py",
             ROOT/"EDM-IR/edmir/benchmark/moe_shape_profile.py")},
        "compiler_admission":False,"new_GPU_measurements":0},descriptions


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    plan,desc=prepare(a.inventory)
    for name,value in (("protocol.json",plan),("descriptions.json",desc)):
        target=a.out/name;target.write_text(json.dumps(value,indent=2)+"\n")
        (a.out/(name+".sha256")).write_text(digest(target)+"  "+name+"\n")
    print(json.dumps({"diagnostic_shapes":len(DIAGNOSTIC),"frozen_boundary_shapes":len(BOUNDARY),
                      "descriptions":len(desc),"independent_evaluation_ready":False}))
