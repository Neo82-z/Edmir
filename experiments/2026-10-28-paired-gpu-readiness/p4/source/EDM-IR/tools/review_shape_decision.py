#!/usr/bin/env python3
"""Bounded historical development pilot. Never a fresh-shape result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"EDM-IR"))
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.tile_execution_model import describe_execution
from edmir.compiler.execution_ledger import source_provenance
from edmir.compiler.shape_decision_model import features, fit, evaluate, VARIANTS, SETTINGS, FEATURES


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    data,*_=load_identified_selector_dataset(ROOT/"experiments/2026-10-20-offline-predictor/data")
    provenance=source_provenance(); rows=[]
    source=str(ROOT/"experiments/2026-10-20-offline-predictor/data/manifest.json")
    for key,surface in sorted(data.validate().items()):
        shape=list(key[-4:])
        for name,row in sorted(surface.items()):
            # Only unchanged tile/work formulas are used for the legacy archive.
            # This does not relabel its binary, timing scope or runtime identity.
            structure=describe_execution(shape,name,environment_id=data.environment_id,
                                         metadata_mode="legacy",provenance=provenance)
            rows.append({"shape":shape,"execution_context":data.environment_id,
                         "candidate_id":name,"is_host":name=="host","features":features(structure),
                         "latency_us":row.measured_ms*1000,"measurement_kind":"observed",
                         "source_id":sha256(Path(source)),"split":"train"})
    shapes=sorted({tuple(r["shape"]) for r in rows})
    # Label-free whole-shape allocation; all archive data already exposed to design.
    order=sorted(shapes,key=lambda s:hashlib.sha256(json.dumps(s).encode()).hexdigest())
    fold={shape:i%4 for i,shape in enumerate(order)}
    design={"purpose":"retrospective_development_only","settings":SETTINGS,"variants":VARIANTS,
            "features":FEATURES,"shape_folds":[{"shape":s,"fold":fold[s]} for s in shapes],
            "dataset_manifest_sha256":sha256(Path(source)),"compiler_admission":False,
            "feature_provenance":"legacy 64x128x32 geometry; current ledger service/identity claims are not transferred"}
    (out/"design-before-fit.json").write_text(json.dumps(design,indent=2)+"\n")
    results={}
    for variant in VARIANTS:
        decisions=[]; audits=[]
        for held in range(4):
            if time.monotonic()-started>120:raise TimeoutError("pilot exceeds 120 seconds; preserve partial outputs")
            train=[r for r in rows if fold[tuple(r["shape"]) ]!=held]
            test=[{**r,"split":"evaluation"} for r in rows if fold[tuple(r["shape"]) ]==held]
            model=fit(train,variant); evaluated=evaluate(model,test)
            decisions.extend(evaluated["decisions"])
            audits.append({"fold":held,"runs":model["runs"],"seconds":model["seconds"],"parameters":len(model["theta"])})
            (out/f"{variant}-fold{held}.json").write_text(json.dumps({"model":model,"evaluation":evaluated},indent=2)+"\n")
        results[variant]={"mean_regret":sum(r["regret"] for r in decisions)/len(decisions),
                          "max_regret":max(r["regret"] for r in decisions),
                          "wrong_host_to_fused":sum(r["wrong_host_to_fused"] for r in decisions),
                          "wrong_fused_to_host":sum(r["wrong_fused_to_host"] for r in decisions),
                          "host_winning_shapes":sum(r["host_wins"] for r in decisions),
                          "all_order_invariant":all(r["order_invariant"] for r in decisions),
                          "mean_shape_mape":sum(r["mape"] for r in decisions)/len(decisions),
                          "fits":audits,"decisions":decisions}
        (out/"progress.json").write_text(json.dumps({"finished":list(results),"seconds":time.monotonic()-started})+"\n")
    report={"schema":"edmir.shape-decision-development.v1","results":results,"design":design,
            "seconds":time.monotonic()-started,"new_gpu_measurements":0,
            "independent_generalization_claim":False,"compiler_admission":False,
            "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in
                (Path(__file__),ROOT/"EDM-IR/edmir/compiler/shape_decision_model.py")}}
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"seconds":report["seconds"],"results":{v:{k:r[k] for k in
        ("mean_regret","max_regret","wrong_host_to_fused","wrong_fused_to_host","all_order_invariant")} for v,r in results.items()}},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
