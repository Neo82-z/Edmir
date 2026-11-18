#!/usr/bin/env python3
"""Read-only synthesis: raw receipts and SAVED coefficients; never fits a model.

Run with the existing numpy environment. No GPU, network, optimizer, simulator,
new labels, or access to the six-shape holdout. Serial timings are not IID trials.
"""
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path
import hashlib
import json
import statistics as st
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "EDM-IR"))
import numpy as np
from edmir.compiler import w1_candidate_selection as score

read = lambda p: json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

FINAL = ROOT / "experiments/2026-11-17-final-operator"
PREV = ROOT / "experiments/2026-11-16-precision-analysis/attempt-01"
TAGS = ("grouped", "m512", "large_n", "m256")

def raw_audit(run, pattern):
    rec = run / "recovered"
    manifest = read(rec / "recovery-manifest.json")["files"]
    for name, info in manifest.items():
        p = (rec / name).resolve()
        assert p.is_relative_to(rec.resolve())
        assert p.stat().st_size == info["bytes"] and sha(p) == info["sha256"], name
    check_rows, measures, blocks = [], [], []
    paths = sorted(rec.glob(pattern))
    for path in paths:
        r = read(path)
        assert r["context"] == sha(path.parent / "context.json")
        stdout = path.parent / Path(r["stdout"]).name
        assert sha(stdout) == r["stdout_sha256"]
        raw = [json.loads(line) for line in stdout.read_text().splitlines()
               if line.startswith('{"family":')]
        assert raw == r["rows"], path
        shape = r["job"]["shape"]
        medians = {}
        for pos, x in enumerate(r["rows"]):
            if r["job"]["phase"] == "check":
                check_rows.append(dict(shape=shape, family=x["family"], seed=r["job"]["seed"],
                                       passed=x["correctness_passed"]))
                continue
            assert x["correctness_passed"]
            samples = x["samples_ms"]
            assert len(samples) == 100 and all(np.isfinite(v) and v > 0 for v in samples)
            median = st.median(samples) * 1000
            assert abs(median - x["median_ms"] * 1000) < 1e-8
            medians[x["family"]] = median
            measures.append(dict(shape=shape, family=x["family"], phase=r["job"]["phase"],
                block=str(path.relative_to(run)), seed=r["job"]["seed"], position=pos,
                median_us=median, samples=len(samples),
                half_drift_pct=100*(st.median(samples[50:])/st.median(samples[:50])-1),
                implementation_id=x["implementation_id"], algorithm=x["gemm_algorithm_id"]))
        if medians:
            blocks.append(dict(shape=shape, seed=r["job"]["seed"], job=r["job"], medians_us=medians))
    return dict(verified_files=len(manifest), groups=len(paths), checks=len(check_rows),
                failed_checks=sum(not x["passed"] for x in check_rows),
                failed_combinations=len({(x["shape"], x["family"]) for x in check_rows if not x["passed"]}),
                timing_rows=len(measures), raw_samples=sum(x["samples"] for x in measures),
                timing_seeds=sorted({x["seed"] for x in measures}),
                shapes=sorted({x["shape"] for x in measures}),
                max_abs_within_process_half_drift_pct=max(abs(x["half_drift_pct"]) for x in measures),
                raw_manifest_sha256=sha(rec / "recovery-manifest.json")), measures, blocks

def saved_solution_audit(model, rows):
    train = [r for r in rows if r["admission"] == "passed" and r["shape"] in model["training_shapes"]]
    ordered, weights = [], []
    for shape in sorted(model["training_shapes"]):
        surface = sorted([r for r in train if r["shape"] == shape], key=lambda r:r["candidate_id"])
        nh = sum(r["candidate_id"] in score.HOSTS for r in surface)
        ordered.extend(surface)
        weights.extend(.5/len(model["training_shapes"])/(nh if r["candidate_id"] in score.HOSTS else len(surface)-nh) for r in surface)
    x = np.array([score.features(r, model["variant"]) for r in ordered])
    y = np.log([r["latency_us"] for r in ordered]); w = np.array(weights)
    assert np.allclose(w@x, model["means"], rtol=0, atol=1e-13)
    assert np.allclose(np.maximum(np.sqrt(w@((x-w@x)**2)), .1), model["scales"], rtol=0, atol=1e-13)
    z = np.column_stack(((x-model["means"])/model["scales"], np.ones(len(x))))
    theta = np.array(model["theta"])
    reg = model["ridge"]*np.eye(len(theta)); reg[-1,-1] = 0
    A = z.T@(w[:,None]*z)+reg; b = z.T@(w*y)
    residual = A@theta-b
    predicted = z@theta
    weighted_error = w*(predicted-y)**2
    groups = {}
    for name in ("host", "fused"):
        mask = np.array([(r["candidate_id"] in score.HOSTS) == (name == "host") for r in ordered])
        groups[name] = dict(weight=float(w[mask].sum()), log_squared_error=float(weighted_error[mask].sum()),
                           unweighted_mape_pct=float(np.mean(np.abs(np.exp(predicted[mask]-y[mask])-1))*100))
    eig = np.linalg.eigvalsh(A)
    rank = int(np.linalg.matrix_rank(z))
    assert rank == model["design_rank"] and eig.min() > 0
    assert np.linalg.norm(residual) / max(np.linalg.norm(b), 1) < 1e-12
    # No parameter changes: only SVD of the training design, and projections.
    _, _, vh = np.linalg.svd(z, full_matrices=True)
    null = vh[rank:].T
    contrasts = []
    test = {r["candidate_id"]:r for r in rows if r["shape_tag"] == "large_n" and r["admission"] == "passed"}
    if [8,64,4096,2048] not in model["training_shapes"]:
        for a,bname in (("splitk8_groupedn2", "splitk2"), ("splitk8_groupedn2", "splitk4_groupedn2")):
            dx = (np.array(score.features(test[a],model["variant"]))-score.features(test[bname],model["variant"]))/model["scales"]
            dz = np.r_[dx,0.]
            contrib = dx*theta[:-1]
            null_part = null@(null.T@dz)
            contrasts.append(dict(a=a,b=bname,predicted_log_ratio=float(dz@theta),
                predicted_time_ratio=float(np.exp(dz@theta)),
                contrast_norm=float(np.linalg.norm(dz)),
                training_nullspace_component_norm=float(np.linalg.norm(null_part)),
                training_nullspace_fraction=float(np.linalg.norm(null_part)/np.linalg.norm(dz)),
                contributions=sorted([dict(feature=f,log_ratio_contribution=float(c))
                   for f,c in zip(model["features"],contrib)],key=lambda r:abs(r["log_ratio_contribution"]), reverse=True),
                interpretation="algebraic decomposition, not causal feature importance; rank deficiency is before ridge"))
    return dict(training_rows=len(train), training_shapes=model["training_shapes"],
        design_rank=rank, coefficients=len(theta), unregularized_nullity=len(theta)-rank,
        regularized_eigenvalue_min=float(eig.min()), regularized_condition_number=float(eig.max()/eig.min()),
        normal_equation_relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(b),1)),
        max_absolute_objective_gradient=float(np.abs(2*residual).max()),
        weighted_log_squared_error=float(weighted_error.sum()), regularization=float(theta@reg@theta),
        weighted_log_rmse=float(np.sqrt(weighted_error.sum())),
        training_loss_groups=groups, constant_features=model["constant_training_features"],
        extrapolation_contrasts=contrasts)

def final_diagnostics(rawrows, blocks, prereg):
    scores = prereg["predictions"]["coupled_geometry"]["scores_us"]
    order = sorted(scores, key=lambda f:(scores[f],f))
    aggregate = {f:st.median(b["medians_us"][f] for b in blocks) for f in order}
    paired = []
    for a,b in combinations(sorted(order),2):
        ds=[x["medians_us"][a]-x["medians_us"][b] for x in blocks]
        paired.append(dict(a=a,b=b,a_faster_blocks=sum(d<0 for d in ds),
            paired_median_delta_us=st.median(ds), aggregate_delta_us=aggregate[a]-aggregate[b],
            direction_wrong=(scores[a]-scores[b])*(aggregate[a]-aggregate[b])<0))
    policy = {}
    for f in order:
        regrets=[100*(b["medians_us"][f]/min(b["medians_us"].values())-1) for b in blocks]
        policy[f]=dict(median_us=aggregate[f],wins=sum(f==min(b["medians_us"],key=b["medians_us"].get) for b in blocks),
                       aggregate_regret_pct=100*(aggregate[f]/min(aggregate.values())-1),
                       mean_block_regret_pct=st.mean(regrets),worst_block_regret_pct=max(regrets))
    shortlists=[]
    for k in range(1,len(order)+1):
        chosen=order[:k]
        losses=[100*(min(b["medians_us"][f] for f in chosen)/min(b["medians_us"].values())-1) for b in blocks]
        shortlists.append(dict(k=k,candidates=chosen,oracle_retention_blocks=sum(v==0 for v in losses),
            mean_oracle_shortlist_regret_pct=st.mean(losses),max_oracle_shortlist_regret_pct=max(losses),
            actual_search_or_timing_cost_included=False))
    sensitivity={}
    # Sensitivity only: no fitting, selecting a protocol, or discarding observations.
    for axis in ("seed", "order", "block"):
        keys=[b["seed"] if axis=="seed" else tuple(b["job"]["order"]) if axis=="order" else i for i,b in enumerate(blocks)]
        out=[]
        for key in sorted(set(keys)):
            keep=[b for b,v in zip(blocks,keys) if v!=key]
            times={f:st.median(b["medians_us"][f] for b in keep) for f in order}
            out.append(dict(omitted=key,retained_blocks=len(keep),winner=min(times,key=times.get),
                           frozen_choice_regret_pct=100*(times[order[0]]/min(times.values())-1)))
        sensitivity[axis]=out
    return dict(rank=order, policies=policy, pairwise=paired, shortlist=shortlists,
                leave_group_out_descriptive_sensitivity=sensitivity,
                order_values=sorted({tuple(b["job"]["order"]) for b in blocks}),
                rawrows=len(rawrows), inference="no IID confidence interval; post-hoc fixed policies are comparators, not validated selectors")

def main():
    protected=[ROOT/"EDM-IR/edmir/compiler/w1_candidate_selection.py"]
    protected+=sorted(PREV.glob('*geometry-*.json'))
    protected+=sorted((FINAL/"frozen-predictions").glob('*'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    audits, measurements, blocks = {}, {}, {}
    for date,run,pat in (("2026-11-11","2026-11-11-shadeform-w1","results/collect-[0-9][0-9][0-9].json"),
                         ("2026-11-16","2026-11-16-shadeform-precision","results-precision/*/collect-[0-9][0-9][0-9].json"),
                         ("2026-11-17","2026-11-17-final-operator","results-balanced-large-n/collect-[0-9][0-9][0-9].json")):
        audits[date],measurements[date],blocks[date]=raw_audit(ROOT/"experiments"/run,pat)
    for name,info in read(FINAL/"frozen-predictions-manifest.json")["files"].items():
        assert sha(FINAL/name)==info["sha256"]
    dataset=read(PREV/"dataset.json");rows=dataset["rows"]
    models={}
    for variant in ("original_geometry","coupled_geometry"):
        models[variant]={tag:saved_solution_audit(read(PREV/f'{variant}-{tag}.json')["model"],rows) for tag in TAGS}
    families=list(read(FINAL/"frozen-predictions/preregistered.json")["predictions"]["coupled_geometry"]["scores_us"])
    queries=[{k:r[k] for k in (*score.CONTRACT_FIELDS,"checks","features","sm_count")} for r in rows
             if r["shape_tag"]=="large_n" and r["candidate_id"] in families]
    order_tests={}
    for variant in models:
        m=read(FINAL/f'frozen-predictions/{variant}-large_n.json')["model"]
        canonical=score.predict(m,queries,dataset["required_inputs"]["large_n"])
        for perm in permutations(queries):
            assert score.predict(m,list(perm),dataset["required_inputs"]["large_n"])==canonical
        order_tests[variant]=dict(permutations=720,scores_and_choice_identical=True,chosen=canonical["chosen"],
            context="original Nov16 admissible queries only; new session not silently rebound")
    collisions=[]
    for r in rows:
        if not r["candidate_id"].endswith("_fp32reduce") or r["admission"]!="passed":continue
        other=next(v for v in rows if v["shape_tag"]==r["shape_tag"] and v["candidate_id"]==r["candidate_id"].removesuffix("_fp32reduce"))
        if other["admission"]!="passed":continue
        assert score.features(r,"original_geometry")==score.features(other,"original_geometry")
        collisions.append(dict(shape=r["shape_tag"],candidate=other["candidate_id"],
            fp16_us=other["latency_us"],fp32_us=r["latency_us"],
            fp32_slowdown_pct=100*(r["latency_us"]/other["latency_us"]-1),
            identical_original_features=True,
            coupled_features_distinct=score.features(r,"coupled_geometry")!=score.features(other,"coupled_geometry")))
    policies={}
    for tag in TAGS:
        candidates={r["candidate_id"]:r["latency_us"] for r in rows if r["shape_tag"]==tag and r["admission"]=="passed"}
        policies[tag]={f:dict(valid=f in candidates,latency_us=candidates.get(f),
                       regret_pct=100*(candidates[f]/min(candidates.values())-1) if f in candidates else None)
                       for f in ("splitk2","splitk4","splitk4_groupedn2","splitk8_groupedn2","host_merged_cutlass","host_merged_cublaslt")}
    final=final_diagnostics(measurements["2026-11-17"],blocks["2026-11-17"],read(FINAL/"frozen-predictions/preregistered.json"))
    result=dict(scope="read-only evidence synthesis; no fitting, no new device work; development data only",
        raw_audits=audits,saved_solution_audits=models,enumeration_order_tests=order_tests,
        precision_feature_collisions=collisions,nov16_fixed_policies_posthoc=policies,
        final_round=final,nov16_saved_shape_fold_metrics=read(PREV/"score-diagnostics.json"),
        total_recent_raw_samples=sum(a["raw_samples"] for a in audits.values()),
        protected_files_unchanged=before=={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()},
        protected_hashes=before)
    assert result["protected_files_unchanged"]
    write(HERE/"results.json",result)
    print(json.dumps(dict(verified_files=sum(a['verified_files'] for a in audits.values()),
        recent_raw_samples=result['total_recent_raw_samples'],
        max_relative_normal_equation_residual=max(v['normal_equation_relative_residual'] for m in models.values() for v in m.values()),
        enumeration_permutations=1440,precision_collisions=len(collisions),predictor_unchanged=True),ensure_ascii=False))

if __name__ == "__main__":main()
