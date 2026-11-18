"""Strict timing envelopes and paired block statistics for routed replay.

This module consumes records; it does not time a CUDA kernel or implement a
selector. Synthetic records can exercise statistics but cannot be paper data.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics


SCHEMA = "edmir.routed-timing.v1"
BOUNDARY = "post_topk_to_canonical_expert_output"
SYNTHETIC = "synthetic_pipeline_fixture"
MEASURED = "cuda_event_measurement"
FIXTURE_STATS = {"resamples":2000, "confidence":.95, "seed":1729,
                 "p95_definition":"nearest_rank", "ci_definition":"linear_percentile"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


def require_hash(name, value):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9a-f]{64}",value):
        raise ValueError(f"{name} must be SHA-256")


def request_split(request_id: str, salt: str) -> str:
    """Assign once per parent request, not per layer/phase/control/world size."""
    if not request_id or not salt:
        raise ValueError("request ID and split salt must be nonempty")
    bucket = int(digest([salt,request_id]),16)%100
    return "development" if bucket < 60 else "calibration" if bucket < 80 else "evaluation"


def validate_isolation(records, salt: str):
    seen = set()
    for r in records:
        if r["record_id"] in seen:
            raise ValueError("duplicate trace record ID")
        seen.add(r["record_id"])
        if r["split"] != request_split(r["parent_request_id"],salt):
            raise ValueError("parent-request split leakage")


def validate_batch(batch: dict) -> None:
    if batch.get("schema") != SCHEMA or batch.get("boundary") != BOUNDARY:
        raise ValueError("timing schema or boundary mismatch")
    kind = batch.get("evidence_kind")
    if kind not in {SYNTHETIC,MEASURED}:
        raise ValueError("timing provenance is required")
    if batch.get("timer") != ("synthetic" if kind == SYNTHETIC else "cuda_event"):
        raise ValueError("timer/provenance mismatch")
    if batch.get("warmups") != 10:
        raise ValueError("D7 normal points require 10 warmups")
    if batch.get("stage") != "normal":
        raise ValueError("this validator covers normal points; repeats use a separately pinned envelope")
    for name in ("trace_hash","protocol_hash","environment_hash","source_hash"):
        require_hash(name,batch.get(name))
    p = batch.get("world_size")
    if type(p) is not int or p not in (2,4,8):
        raise ValueError("primary world_size must be P2/P4/P8")
    if batch.get("dtype") != "bfloat16" or batch.get("phase") not in {"prefill","decode","synthetic"}:
        raise ValueError("primary BF16 phase mismatch")
    if not isinstance(batch.get("family"),str) or not batch["family"]:
        raise ValueError("missing candidate family")
    if not isinstance(batch.get("pairing_id"), str) or not batch["pairing_id"]:
        raise ValueError("missing paired execution cohort ID")
    if type(batch.get("numerical_check_passed")) is not bool:
        raise ValueError("numerical_check_passed must be a boolean")
    blocks = batch.get("blocks",[])
    if (len(blocks) != 10 or any(type(b.get("block_id")) is not int for b in blocks)
        or sorted(b["block_id"] for b in blocks) != list(range(10))):
        raise ValueError("normal point requires 10 unique complete blocks")
    for block in blocks:
        if kind == MEASURED:
            order = block.get("candidate_order", [])
            if (not order or any(not isinstance(f, str) for f in order)
                or len(set(order)) != len(order) or batch["family"] not in order):
                raise ValueError("measured block must log its randomized candidate order")
        if block.get("status") != "valid":
            raise ValueError("invalid block: retain the event and rerun the whole block; never trim samples")
        samples = block.get("rank_ms",[])
        if len(samples) != 20:
            raise ValueError("each block requires all 20 samples; outlier removal is forbidden")
        for ranks in samples:
            if len(ranks) != p or any(type(t) not in (int,float) or not math.isfinite(t) or t <= 0 for t in ranks):
                raise ValueError("sample must contain finite positive timings for every rank")


def block_durations(batch):
    validate_batch(batch)
    return [[max(ranks) for ranks in block["rank_ms"]]
            for block in sorted(batch["blocks"],key=lambda b:b["block_id"])]


def percentile(values, probability):
    values = sorted(values)
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile")
    index = (len(values)-1)*probability
    lo,hi = math.floor(index),math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def summarize(batch):
    values = [v for block in block_durations(batch) for v in block]
    return {"family":batch["family"], "samples":len(values), "blocks":10,
            "median_ms":statistics.median(values),
            "p95_ms":sorted(values)[math.ceil(.95*len(values))-1],
            "evidence_kind":batch["evidence_kind"],
            "aggregation":"max_rank_per_sample_then_sample_quantile"}


def compare_batches(host, candidate, *, resamples=2000, confidence=.95, seed=1729):
    """Paired bootstrap resamples whole blocks, keeping all samples within each.

    The estimand is ratio of pooled sample medians, not median of block ratios.
    Defaults are deterministic engineering fixtures, not campaign preregistration.
    """
    h,c = block_durations(host),block_durations(candidate)
    for name in ("trace_hash","protocol_hash","environment_hash","source_hash",
                 "world_size","phase","dtype","stage","evidence_kind","pairing_id"):
        if host[name] != candidate[name]:
            raise ValueError(f"paired evidence binding mismatch: {name}")
    if host["family"] == candidate["family"]:
        raise ValueError("comparison requires distinct families")
    if host["evidence_kind"] == MEASURED:
        hb = sorted(host['blocks'], key=lambda b:b['block_id'])
        cb = sorted(candidate['blocks'], key=lambda b:b['block_id'])
        if any(a['candidate_order'] != b['candidate_order'] for a,b in zip(hb,cb)):
            raise ValueError("paired blocks disagree on candidate execution order")
    if type(resamples) is not int or resamples < 100 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        indices = [rng.randrange(len(h)) for _ in h]
        hm = statistics.median([v for i in indices for v in h[i]])
        cm = statistics.median([v for i in indices for v in c[i]])
        draws.append(cm/hm-1)
    hs,cs = summarize(host),summarize(candidate)
    relative = cs["median_ms"]/hs["median_ms"]-1
    alpha = (1-confidence)/2
    low,high = percentile(draws,alpha),percentile(draws,1-alpha)
    return {"host":hs,"candidate":cs,"relative_latency":relative,
            "relative_latency_ci":[low,high], "speedup_over_comparator":1/(1+relative),
            "needs_repeat":abs(relative)<.01 or low<=0<=high,
            "statistically_supported_host_regression_over_1pct":low>.01,
            "bootstrap":{"resamples":resamples,"confidence":confidence,"seed":seed,
                         "resampling_unit":"paired_block"},
            "evidence_kind":host["evidence_kind"],"deployment_decision":None}


def require_paper_evidence(batches, *, campaign: dict) -> None:
    """Fail closed. Hash bindings are provenance checks, not authenticity proof."""
    if campaign.get("status") != "sealed" or not campaign.get("unresolved") == []:
        raise ValueError("paper export requires a sealed campaign with no unresolved choices")
    require_hash("protocol_hash",campaign.get("protocol_hash"))
    if not batches:
        raise ValueError("paper export cannot turn missing measurements into zeros")
    for batch in batches:
        validate_batch(batch)
        if batch["evidence_kind"] != MEASURED or batch["phase"] == "synthetic":
            raise ValueError("synthetic fixtures are forbidden in paper results")
        if batch["protocol_hash"] != campaign["protocol_hash"]:
            raise ValueError("campaign protocol hash mismatch")
        if not batch["numerical_check_passed"]:
            raise ValueError("numerical validation has not passed")
        if batch["trace_hash"] not in campaign.get("trace_hashes",[]):
            raise ValueError("trace absent from sealed campaign")
        if batch["environment_hash"] not in campaign.get("environment_hashes",[]):
            raise ValueError("environment absent from sealed campaign")
        if batch["source_hash"] not in campaign.get("source_hashes",[]):
            raise ValueError("backend build absent from sealed campaign")


def synthetic_batch(world_size: int, family: str, *, trace_hash: str, scale=1.0) -> dict:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("synthetic scale must be positive")
    return {"schema":SCHEMA,"boundary":BOUNDARY,"evidence_kind":SYNTHETIC,"timer":"synthetic",
            "warmups":10,"stage":"normal","world_size":world_size,"dtype":"bfloat16",
            "phase":"synthetic","family":family,"trace_hash":trace_hash,
            "pairing_id":"synthetic-cohort-"+trace_hash,
            "protocol_hash":digest("offline-fixture-not-d7-measurement"),
            "environment_hash":digest("no-gpu"),"source_hash":digest("synthetic-generator-v1"),
            "numerical_check_passed":True,
            "blocks":[{"block_id":b,"status":"valid",
                       "rank_ms":[[scale*(1+.02*b+.003*s+.001*r) for r in range(world_size)]
                                  for s in range(20)]} for b in range(10)]}
