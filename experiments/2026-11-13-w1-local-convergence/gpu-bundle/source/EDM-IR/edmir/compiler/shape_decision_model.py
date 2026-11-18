"""Experimental numerical decision head; never admitted by the compiler.

One score function for all candidates, whole-shape training, observed costs
only. Geometry is evidence about work, not an identified latency mechanism.
Softmin weights are NOT calibrated probabilities of being fastest.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

from .shared_candidate_selection import forward, parameter_count, canonical_choice
from .execution_policy import UNIFORM_PEER_POLICY

SCHEMA = "edmir.shape-decision.v1"
FEATURES = (
    "host", "merged_host", "geometry_unknown", "uniform_dependencies", "prepared",
    "log_P", "log_M", "log_K", "log_N", "log_SM", "log_q", "log_group_N",
    "log_gemms_per_destination", "log_logical_flops", "log_A_bytes", "log_B_bytes",
    "log_remote_input_bytes", "log_m_tiles", "log_n_tiles", "log_output_tiles",
    "log_ctas", "log_K_iterations", "valid_tile_fraction", "last_M_fraction",
    "last_N_fraction", "reference_waves", "last_wave_fraction",
    "log_epilogue_edges", "log_padded_work", "group_x_K_depth", "q_x_output_tiles",
)
VARIANTS = {"linear_latency": (0, 0., 0.), "linear_decision": (0, 1., 1.),
            "mlp_decision": (8, 1., 1.)}
SETTINGS = {"epochs": 100, "learning_rate": .003, "weight_decay": .001,
            "temperature": .05, "seeds": [0, 1], "gradient_clip": 10.,
            "batching": "all training shapes; equal shape mass", "max_seconds": 10.}


def features(structure):
    """Known CUTLASS geometry; explicit missingness for opaque library tiling."""
    ident, g = structure["identity"], structure["geometry"]
    p, m, k, n = ident["shape"]
    host = ident["baseline_id"] is not None
    merged = ident["family"].startswith("host_merged")
    log = math.log1p
    # The unique remote-input volume is logical, not measured interconnect traffic.
    out = [float(host), float(merged), float(g is None),
           float(ident["dependency_policy"] == UNIFORM_PEER_POLICY),
           float(ident["metadata_mode"] != "legacy"),
           log(p), log(m), log(k), log(n), log(structure["sm_count"])]
    q = g["q"] if g else 1
    group = g["tile_mnk"][1] / 128 if g else 1
    gm = p*m if merged else m
    out += [log(q) if g else 0., log(group) if g else 0.,
            log(structure["program"]["gemms_per_destination"]), log(2*gm*k*n),
            log(2*gm*k), log(2*k*n), log(2*p*(p-1)*m*k)]
    if g:
        ref = g["one_cta_per_sm_reference"]
        out += [log(g["m_tiles"]), log(g["n_tiles"]), log(g["output_tiles"]),
                log(g["ctas"]), log(g["partition_k_iterations"]),
                g["valid_output_fraction"], g["last_m_tile_rows"]/g["tile_mnk"][0],
                g["last_n_tile_columns"]/g["tile_mnk"][1], float(ref["rounds"]),
                ref["last_round_fraction"], log(g["epilogue_order"]["ordered_edges"]),
                log(g["padded_tile_work_flops"]),
                math.log2(group)*log(g["partition_k_iterations"]),
                math.log2(q)*log(g["output_tiles"])]
    else:
        out += [0.] * 14  # missingness flag above; not zero-cost service evidence
    if len(out) != len(FEATURES) or not all(math.isfinite(v) for v in out):
        raise ValueError("invalid execution features")
    return out


def _backward(theta, x, grad_score, width):
    import numpy as np
    d = x.shape[-1]
    grad = np.zeros_like(theta)
    grad[:d] = np.einsum("sjd,sj->d", x, grad_score)
    grad[d] = grad_score.sum()
    if width:
        start = d+1
        w = theta[start:start+d*width].reshape(d, width)
        h = np.tanh(x @ w + theta[start+d*width:start+d*width+width])
        dh = grad_score[..., None] * theta[-width:] * (1-h*h)
        grad[start:start+d*width] = np.einsum("sjd,sjh->dh", x, dh).ravel()
        grad[start+d*width:start+d*width+width] = dh.sum(axis=(0, 1))
        grad[-width:] = np.einsum("sjh,sj->h", h, grad_score)
    return grad


def objective(theta, x, target, mask, weights, width, cost_weight, gap_weight):
    """Log-time anchor + exact expected relative regret + observed-best gap.

    Gap uses log ratios, averages over all observed tied minima, and is zero
    for padding. Its purpose is hard-choice alignment, not uncertainty fitting.
    """
    import numpy as np
    score = forward(theta, x, width)
    count = len(x)
    error = np.where(mask, score-target, 0.)
    anchor = np.sum(weights*error**2)/count
    grad = 2*weights*error/count
    logits = np.where(mask, -score/SETTINGS["temperature"], -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(axis=1, keepdims=True)
    best = np.min(np.where(mask, target, np.inf), axis=1, keepdims=True)
    regret = np.expm1(np.where(mask, target-best, 0.))
    expected = np.sum(prob*regret, axis=1, keepdims=True)
    grad -= cost_weight*prob*(regret-expected)/(count*SETTINGS["temperature"])
    gap = 0.
    for i in range(count):
        minima = np.flatnonzero(mask[i] & (target[i] == best[i, 0]))
        scale = count * len(minima) * int(mask[i].sum())
        for b in minima:
            violation = np.where(mask[i], np.maximum(0., target[i]-target[i,b]-score[i]+score[i,b]), 0.)
            gap += float(violation @ violation)/scale
            dg = 2*gap_weight*violation/scale
            grad[i] -= dg
            grad[i,b] += dg.sum()
    grad *= mask
    value = float(anchor + cost_weight*expected.mean() + gap_weight*gap)
    gradient = _backward(theta, x, grad, width)
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite decision objective")
    return value, gradient


def validate_rows(rows, *, split):
    """One declared execution context; no mixed worlds, duplicate labels or blanks."""
    if not 2 <= len(rows) <= 4096:
        raise ValueError("2..4096 measured candidate rows required")
    grouped, contexts, seen = defaultdict(list), set(), set()
    for r in rows:
        shape = tuple(r["shape"])
        if len(shape) != 4 or shape[0] not in (4, 8) or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("invalid P,M,K,N")
        if r["split"] != split or r["measurement_kind"] != "observed" or not r["source_id"]:
            raise ValueError("explicit observed labels in the requested partition required")
        if not r["execution_context"] or not r["candidate_id"]:
            raise ValueError("execution and candidate identities required")
        contexts.add((r["execution_context"], shape[0]))
        if (shape, r["candidate_id"]) in seen:
            raise ValueError("duplicate shape/candidate; aggregate declared replicates first")
        seen.add((shape, r["candidate_id"]))
        if (type(r["is_host"]) is not bool or len(r["features"]) != len(FEATURES)
            or not all(math.isfinite(v) for v in r["features"])
            or not math.isfinite(r["latency_us"]) or r["latency_us"] <= 0):
            raise ValueError("finite features and positive measured latency required")
        grouped[shape].append(r)
    if len(contexts) != 1:
        raise ValueError("cannot mix execution contexts or GPU counts")
    if any(not any(r["is_host"] for r in rs) or all(r["is_host"] for r in rs) for rs in grouped.values()):
        raise ValueError("each measured surface needs host and fused observations")
    if any(len(rs) > 14 for rs in grouped.values()):
        raise ValueError("at most 14 candidates per surface")
    return [sorted(grouped[k], key=lambda r:r["candidate_id"]) for k in sorted(grouped)]


def fit(rows, variant="mlp_decision"):
    """Fixed two-seed AdamW experiment, train-only scaler, no test-based retry."""
    import numpy as np
    start = time.monotonic()
    def deadline():
        if time.monotonic()-start > SETTINGS["max_seconds"]:
            raise TimeoutError("decision fit exceeded fixed local budget")
    surfaces = validate_rows(rows, split="train")
    if len(surfaces) < 2:
        raise ValueError("at least two whole training shapes required")
    width, cw, gw = VARIANTS[variant]
    s, j, d = len(surfaces), max(map(len, surfaces)), len(FEATURES)
    x, y, weights = np.zeros((s,j,d)), np.zeros((s,j)), np.zeros((s,j))
    mask = np.zeros((s,j), bool)
    for i, surface in enumerate(surfaces):
        hosts = sum(r["is_host"] for r in surface)
        for t, r in enumerate(surface):
            x[i,t], y[i,t], mask[i,t] = r["features"], math.log(r["latency_us"]), True
            weights[i,t] = .5/(hosts if r["is_host"] else len(surface)-hosts)
    means = np.sum(x*weights[...,None], axis=(0,1))/s
    scales = np.maximum(np.sqrt(np.sum((x-means)**2*weights[...,None], axis=(0,1))/s), .1)
    x = np.where(mask[...,None], (x-means)/scales, 0.)
    a = np.concatenate((x,np.ones((s,j,1))), axis=-1).reshape(-1,d+1)
    w = weights.ravel()/s
    affine = np.linalg.solve(a.T@(a*w[:,None])+.001*np.eye(d+1), a.T@(w*y.ravel()))
    runs, params = [], []
    for seed in (SETTINGS["seeds"] if width else [0]):
        theta = np.zeros(parameter_count(d,width)); theta[:d+1] = affine
        if width:
            theta[d+1:d+1+d*width] = np.random.default_rng(seed).normal(0,.15/math.sqrt(d),d*width)
        first, second = np.zeros_like(theta), np.zeros_like(theta)
        initial = objective(theta,x,y,mask,weights,width,cw,gw)[0]
        for epoch in range(1, SETTINGS["epochs"]+1):
            deadline()
            _, grad = objective(theta,x,y,mask,weights,width,cw,gw)
            grad *= min(1., SETTINGS["gradient_clip"] / max(float(np.linalg.norm(grad)),1e-30))
            first = .9*first + .1*grad; second = .999*second + .001*grad*grad
            theta *= 1-SETTINGS["learning_rate"]*SETTINGS["weight_decay"]
            theta -= SETTINGS["learning_rate"]*(first/(1-.9**epoch))/(np.sqrt(second/(1-.999**epoch))+1e-8)
        value, grad = objective(theta,x,y,mask,weights,width,cw,gw)
        runs.append({"seed":seed,"initial_loss":initial,"final_loss":value,
                     "epochs":SETTINGS["epochs"],"gradient_norm":float(np.linalg.norm(grad)),
                     "initial_loss_worsened":bool(value>initial),"convergence_claim":False})
        params.append(theta.tolist())
    selected = min(range(len(runs)), key=lambda i:(runs[i]["final_loss"],runs[i]["seed"]))
    return {"schema":SCHEMA,"variant":variant,"features":list(FEATURES),"width":width,
            "means":means.tolist(),"scales":scales.tolist(),"theta":params[selected],
            "execution_context":rows[0]["execution_context"],"world_size":rows[0]["shape"][0],
            "training_shapes":[list(s[0]["shape"]) for s in surfaces],
            "training_sources":sorted({r["source_id"] for r in rows}),
            "settings":dict(SETTINGS),"runs":runs,"selected_start":selected,
            "seconds":time.monotonic()-start,"compiler_admission":False,
            "physical_service_parameters_identified":False}


def predict(model, rows):
    """Candidate order never enters the scorer. Refuse cross-context transfer."""
    import numpy as np
    if (model["schema"] != SCHEMA or model["features"] != list(FEATURES)
        or not 1 <= len(rows) <= 14 or len({r["candidate_id"] for r in rows}) != len(rows)
        or len({tuple(r["shape"]) for r in rows}) != 1):
        raise ValueError("one bounded candidate surface required")
    if any(r["execution_context"] != model["execution_context"] or r["shape"][0] != model["world_size"] for r in rows):
        raise ValueError("model is not calibrated for this execution context")
    rows = sorted(rows, key=lambda r:r["candidate_id"])
    x = (np.asarray([r["features"] for r in rows])-model["means"])/model["scales"]
    scores = forward(np.asarray(model["theta"]),x,model["width"])
    result = dict(zip((r["candidate_id"] for r in rows),map(float,scores)))
    canonical_choice(result)  # finite validation; names are not features
    return result


def evaluate(model, rows):
    surfaces = validate_rows(rows, split="evaluation")
    if any(list(s[0]["shape"]) in model["training_shapes"] for s in surfaces):
        raise ValueError("whole-shape train/evaluation leakage")
    records = []
    for surface in surfaces:
        scores = predict(model,surface); chosen = canonical_choice(scores)
        actual = {r["candidate_id"]:r["latency_us"] for r in surface}
        best = min(actual.values()); best_host = min(r["latency_us"] for r in surface if r["is_host"])
        best_fused = min(r["latency_us"] for r in surface if not r["is_host"])
        chosen_host = next(r["is_host"] for r in surface if r["candidate_id"]==chosen)
        records.append({"shape":surface[0]["shape"],"chosen":chosen,"scores_log_us":scores,
                        "measured_us":actual,"regret":actual[chosen]/best-1,
                        "host_wins":best_host<best_fused,"fused_wins":best_fused<best_host,
                        "wrong_host_to_fused":best_host<best_fused and not chosen_host,
                        "wrong_fused_to_host":best_fused<best_host and chosen_host,
                        "order_invariant":scores==predict(model,list(reversed(surface))),
                        "mape":sum(abs(math.expm1(scores[n]-math.log(t))) for n,t in actual.items())/len(actual)})
    return {"decisions":records,"mean_regret":sum(r["regret"] for r in records)/len(records),
            "max_regret":max(r["regret"] for r in records),
            "host_winning_shapes":sum(r["host_wins"] for r in records),
            "fused_winning_shapes":sum(r["fused_wins"] for r in records),
            "wrong_host_to_fused":sum(r["wrong_host_to_fused"] for r in records),
            "wrong_fused_to_host":sum(r["wrong_fused_to_host"] for r in records),
            "all_order_invariant":all(r["order_invariant"] for r in records)}
