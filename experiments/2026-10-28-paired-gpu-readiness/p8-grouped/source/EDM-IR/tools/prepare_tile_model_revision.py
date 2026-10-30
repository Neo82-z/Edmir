#!/usr/bin/env python3
"""Prepare a bounded local representation/ablation review, not a GPU package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EDM-IR"))
from edmir.compiler.execution_ledger import FAMILIES, execution_ledger, source_provenance
from edmir.compiler.execution_policy import LEGACY_POLICY, UNIFORM_PEER_POLICY
from edmir.compiler.scale_aware_prediction import basis
from edmir.compiler.structured_validation import BASELINES, development_folds
from edmir.compiler.tile_execution_model import structure_from_ledger, conditional_dependency_span

SHAPES = ((8, 64, 4096, 2048), (8, 64, 8192, 1024),
          (8, 256, 8192, 512), (8, 512, 8192, 512))
CANDIDATES = BASELINES + FAMILIES[1:]
CONTEXT = "planned-a100-sxm4-80gb-p8-tile-structure-v1"


def prepare():
    start = time.monotonic()
    provenance = source_provenance()

    def ledger(shape, family):
        host = family in BASELINES
        return execution_ledger(shape, family, environment_id=CONTEXT,
            metadata_mode="merged" if family.startswith("host_merged") else "prepared",
            dependency_policy=LEGACY_POLICY if host else UNIFORM_PEER_POLICY,
            provenance=provenance)

    descriptions, records = [], []
    for shape in SHAPES:
        for family in CANDIDATES:
            descriptions.append(structure_from_ledger(ledger(shape, family)))
            for rep in range(3):
                records.append({"shape": list(shape), "execution_context": CONTEXT,
                    "candidate_id": family, "replicate_id": f"planned-block-{rep}", "split": "train"})
    folds = development_folds(records, {CONTEXT: CANDIDATES})
    aliases = []
    for a, b, family in (
        ((8,32,8192,8192), (8,64,8192,4096), "splitk8_groupedn2"),
        ((8,32,4096,8192), (8,64,4096,4096), "splitk8"),
    ):
        left, right = (structure_from_ledger(ledger(s, family))["geometry"] for s in (a,b))
        same_old = basis(a, family) == basis(b, family)
        distinguishing = [k for k in ("m_tiles", "n_tiles", "output_tiles", "ctas",
            "valid_output_fraction", "padded_tile_work_flops", "one_cta_per_sm_reference")
            if left[k] != right[k]]
        if not same_old or not distinguishing:
            raise ValueError("recheck the previous collision and new representation")
        aliases.append({"a": list(a), "b": list(b), "family": family,
            "old_complete_basis_equal": same_old, "new_distinguishing_fields": distinguishing,
            "latency_difference_established": False, "added_to_GPU_shapes": False})
    sample = ledger(SHAPES[-1], "splitk16_groupedn2")
    unknown = conditional_dependency_span(sample, {}, identity_sha256=sample["identity_sha256"],
                                           source="no node-duration measurements supplied")
    paths = [Path(__file__), ROOT/"EDM-IR/edmir/compiler/tile_execution_model.py",
             ROOT/"EDM-IR/edmir/compiler/structured_validation.py",
             ROOT/"EDM-IR/edmir/compiler/scale_aware_prediction.py"]
    plan = {
        "schema": "edmir.tile-model-revision-plan.v1", "date": "2026-10-26",
        "status": "local_structure_and_development_design; not a fitted latency predictor",
        "shape_order": "P,M,K,N", "shapes": SHAPES, "candidates": CANDIDATES,
        "planned_configurations": len(descriptions), "planned_repeated_runs": len(records),
        "planned_repetitions_are_observations": False, "new_measured_rows": 0,
        "models": [
            {"id":"A", "change":"simple baseline adapted to the new execution identity",
             "coefficients":"not fitted; old coefficients cannot be transferred silently"},
            {"id":"B", "change":"A plus explicit tile geometry, capacity reference and tail",
             "coefficients":"not fitted; functional form must be fixed before final evaluation"},
            {"id":"C", "change":"B plus independently constrained service and dependency composition",
             "coefficients":"unidentified pending component evidence"}],
        "ablation_rules": {"same_development_shapes":True, "same_candidates":True,
            "same_calibration_measurement_budget":True, "fixed_model_variants":3,
            "old_replay_is_separate":True, "maximum_fit_seconds_per_fit":10,
            "maximum_replay_seconds":120, "retry_on_model_failure":False,
            "final_form_and_parameter_count_frozen":False,
            "independent_evaluation_ready":False},
        "baselines": [*BASELINES, "measure_two_merged_hosts_within_budget",
                      "training_fixed_order", "equal_budget_search", "measured_set_oracle"],
        "metrics": ["mean_and_max_top1_regret", "pair_order_errors", "latency_error",
                    "top_k_retention_k1_k2_k3", "amortized_total_cost"],
        "validation": {"folds":folds, "purpose":"development_only",
            "independent_shapes":None, "selection_and_verification_blocks_must_differ":True,
            "missing_or_failed_candidates_are_reported_not_dropped":True},
        "constructed_alias_checks": aliases, "unknown_duration_check":unknown,
        "runtime_source_sha256": {k:v["sha256"] for k,v in provenance["project"].items()},
        "implementation_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "model_fits":0, "GPU_runs":0, "compiler_admission":False,
        "duration_seconds":round(time.monotonic()-start, 6),
    }
    return plan, descriptions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args=parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve prior review")
    plan, descriptions=prepare()
    args.out.mkdir(parents=True, exist_ok=False)
    for name,value in (("plan.json",plan),("execution-structures.json",descriptions)):
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"out":str(args.out),"configurations":len(descriptions),
                      "development_folds":len(plan["validation"]["folds"]),
                      "seconds":plan["duration_seconds"],"GPU_runs":0}))


if __name__ == "__main__":
    main()
