"""Diagnostic-only routed shape accounting. No timing or speedup inference.

Capture post-top-k IDs outside timed windows. A complete global step/layer
needs every EP source rank, including empty ranks. Expert aggregate M, source
bucket M, and actual padded kernel M are distinct quantities.
"""
from collections import Counter, defaultdict
import hashlib
import json

SCHEMA = "edmir.moe-shape-profile.v1"


def router_histogram(selected_experts, *, num_experts, top_k, diagnostic_only):
    """CPU lists or an actual router tensor; never generate synthetic routes."""
    if diagnostic_only is not True:
        raise ValueError("router capture is forbidden in the timed performance window")
    if not (type(num_experts) is int and 1 <= num_experts <= 4096
            and type(top_k) is int and 1 <= top_k <= num_experts):
        raise ValueError("invalid expert dimensions")
    if hasattr(selected_experts, "detach"):
        if selected_experts.ndim != 2 or selected_experts.shape[1] != top_k or selected_experts.shape[0] > 65536:
            raise ValueError("bounded token-by-top-k tensor required")
        selected_experts = selected_experts.detach().cpu().tolist()
    if not isinstance(selected_experts, list) or len(selected_experts) > 65536:
        raise ValueError("at most 65536 tokens per capture")
    counts = [0]*num_experts
    for selected in selected_experts:
        if (len(selected) != top_k or len(set(selected)) != top_k
            or any(type(e) is not int or not 0 <= e < num_experts for e in selected)):
            raise ValueError("distinct post-top-k expert IDs required for each token")
        for e in selected:
            counts[e] += 1
    return {"token_count":len(selected_experts),"expert_counts":counts}


def summarize(envelope, *, evaluated_shapes=(), reference_status="declared_shapes_not_validated"):
    """One frozen workload/run profile. Frequencies are diagnostic sample rates.

    `kernel_shapes` are optional actual dispatch/backend observations, never
    inferred from routing. A logical shape match does not establish matching
    distribution, dtype, execution path, kernel calls, or time coverage.
    """
    provenance = envelope["provenance"]
    required = ("model_id","model_revision","engine_revision","run_id","workload_id",
                "request_corpus_sha256","dtype","sampling_rule","router_hook_source")
    if any(not isinstance(provenance.get(k),str) or not provenance[k].strip() for k in required):
        raise ValueError("model, engine, workload, sampling and hook provenance required")
    if envelope["origin"] not in ("real_router", "synthetic_test") or envelope["diagnostic_only"] is not True:
        raise ValueError("explicit diagnostic origin required")
    p,e,k = (envelope[n] for n in ("world_size","num_experts","top_k"))
    if (type(p) is not int or p not in (4,8) or type(e) is not int or not 1 <= e <= 4096
        or type(k) is not int or not 1 <= k <= e):
        raise ValueError("invalid world or expert dimensions")
    records = envelope["records"]
    if not 1 <= len(records) <= 8192:
        raise ValueError("1..8192 diagnostic snapshots required")
    projections = envelope["projections"]
    expected_layers = envelope["expected_layers"]
    if (not isinstance(expected_layers,list) or not 1 <= len(expected_layers) <= 256
        or len(set(expected_layers)) != len(expected_layers)
        or any(type(v) is not int or v < 0 for v in expected_layers)):
        raise ValueError("explicit complete MoE layer list required")
    if (not 1 <= len(projections) <= 3 or len({s["name"] for s in projections}) != len(projections)
        or any(not s["name"] or any(type(s[d]) is not int or s[d] <= 0 for d in ("K","N")) for s in projections)):
        raise ValueError("explicit logical projections required; gate/up fused or separate, never both")
    groups = defaultdict(dict)
    for r in records:
        step, layer, rank = r["step"],r["layer"],r["source_rank"]
        if (any(type(v) is not int or v < 0 for v in (step,layer,rank,r["token_count"]))
            or rank >= p or layer not in expected_layers or r["token_count"] > 65536
            or r["phase"] not in ("prefill","decode")):
            raise ValueError("invalid step/layer/rank/phase")
        counts = r["expert_counts"]
        if (len(counts) != e or any(type(v) is not int or v < 0 or v > r["token_count"] for v in counts)
            or sum(counts) != r["token_count"]*k):
            raise ValueError("router count conservation failed")
        key = (step,layer,r["phase"])
        if rank in groups[key]:
            raise ValueError("duplicate source snapshot; TP replicas must not be counted twice")
        groups[key][rank] = r
    if any(set(rs) != set(range(p)) for rs in groups.values()):
        raise ValueError("missing source ranks; partial steps cannot define global expert M")
    steps = defaultdict(set)
    for step,layer,phase in groups:
        steps[(step,phase)].add(layer)
    if any(layers != set(expected_layers) for layers in steps.values()):
        raise ValueError("missing MoE layers in a sampled step")
    logical, buckets, empty, tokens = Counter(), Counter(), 0, 0
    totals = Counter(); phase_totals = Counter(); kernel = Counter()
    for (step,layer,phase),rs in sorted(groups.items()):
        tokens += sum(r["token_count"] for r in rs.values())
        aggregate = [sum(r["expert_counts"][expert] for r in rs.values()) for expert in range(e)]
        for r in rs.values():
            buckets.update(r["expert_counts"])
        for m in aggregate:
            if not m:
                empty += 1; continue
            for proj in projections:
                key = (phase,proj["name"],m,proj["K"],proj["N"])
                logical[key] += 1
                totals["calls"] += 1; totals["rows"] += m
                totals["flops"] += 2*m*proj["K"]*proj["N"]
                phase_totals[phase] += 1
    # Actual kernel observations are supplementary, never synthesized by rounding M.
    if len(envelope.get("kernel_shapes", [])) > 8192:
        raise ValueError("kernel shape records exceed diagnostic budget")
    for row in envelope.get("kernel_shapes", []):
        if not row.get("backend_source") or any(type(row[n]) is not int or row[n] <= 0 for n in ("M","K","N","count")):
            raise ValueError("kernel geometry requires backend evidence and positive counts")
        kernel[(row["M"],row["K"],row["N"],row["backend_source"])] += row["count"]
    evaluated = {tuple(s[-3:]) for s in evaluated_shapes}
    matched = Counter(); rows = []
    for (phase,name,m,kk,n),count in sorted(logical.items()):
        match = (m,kk,n) in evaluated
        if match:
            matched["calls"] += count; matched["rows"] += count*m; matched["flops"] += count*2*m*kk*n
        rows.append({"phase":phase,"projection":name,"M":m,"K":kk,"N":n,"occurrences":count,
                     "fraction_all_logical_calls":count/totals["calls"],
                     "fraction_within_phase":count/phase_totals[phase],"exact_MKN_match":match})
    return {"schema":SCHEMA,"origin":envelope["origin"],"provenance":provenance,
            "input_sha256":hashlib.sha256(json.dumps(envelope,sort_keys=True).encode()).hexdigest(),
            "complete_step_layers":len(groups),"token_layer_observations":tokens,
            "routed_rows":tokens*k,"empty_expert_step_layers":empty,
            "source_expert_M_histogram":dict(sorted(buckets.items())),"logical_expert_shapes":rows,
            "logical_totals":dict(totals),"logical_exact_MKN_coverage":{
                metric:(matched[metric]/totals[metric] if totals[metric] else None) for metric in ("calls","rows","flops")},
            "kernel_shapes":[{"M":m,"K":kk,"N":n,"backend_source":source,"count":count}
                             for (m,kk,n,source),count in sorted(kernel.items())] if kernel else None,
            "reference_shape_status":reference_status,
            "coverage_scope":"sampled routed expert logical projections only; excludes shared expert, attention, dispatch and combine",
            "execution_identity_coverage":None,"measured_time_coverage":None,
            "application_representativeness_validated":False,
            "interpretation":"routing M is not padded kernel M or client batch; MKN match alone does not transfer uniform FP16 results to routed BF16"}
