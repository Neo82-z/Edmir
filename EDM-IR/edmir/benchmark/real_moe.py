"""Bounded real-MoE collection contracts. No model fitting or GPU imports."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import asyncio
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

SCHEMA = "edmir.real-moe.v1"
ENGINE = "0.20.1"
# One pinned model per process, chosen by EDMIR_MOE_MODEL; the default is the original GLM plan.
# gate_up and down are the (rows, columns) of one expert's two projections, as vLLM stores them.
# layers lists the MoE decoder layers when the full route analysis needs them; snapshot-only
# profiles leave it open and require only that the snapshot layer exists.
PROFILES = {
    "GLM": dict(model="zai-org/GLM-4.7-Flash", revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
                architecture="Glm4MoeLiteForCausalLM", layers=tuple(range(1, 47)), experts=64, top_k=4,
                gate_up=(3072, 2048), down=(2048, 1536), snapshot_layer=23, lengths=(512, 4096),
                engine=dict(attention_config={"backend": "TRITON_MLA", "flash_attn_version": 2,
                                              "disable_flashinfer_prefill": True})),
    "QWEN35": dict(model="Qwen/Qwen3.5-35B-A3B", revision="2508b2437e382c605eb0fc52b83f5545960f6b4a",
                   architecture="Qwen3_5MoeForConditionalGeneration", layers=None, experts=256, top_k=8,
                   gate_up=(1024, 2048), down=(2048, 512), snapshot_layer=20, lengths=(512,),
                   engine=dict(language_model_only=True)),
    "DSV2L": dict(model="deepseek-ai/DeepSeek-V2-Lite-Chat", revision="85864749cd611b4353ce1decdb286193298f64c7",
                  architecture="DeepseekV2ForCausalLM", layers=None, experts=64, top_k=6,
                  gate_up=(2816, 2048), down=(2048, 1408), snapshot_layer=13, lengths=(512,),
                  engine=dict(attention_config={"backend": "TRITON_MLA", "flash_attn_version": 2,
                                                "disable_flashinfer_prefill": True})),
}
PROFILE_ID = os.environ.get("EDMIR_MOE_MODEL", "GLM")
if PROFILE_ID not in PROFILES: raise ValueError("unknown EDMIR_MOE_MODEL: " + PROFILE_ID)
PROFILE = PROFILES[PROFILE_ID]
MODEL, REVISION = PROFILE["model"], PROFILE["revision"]
LAYERS = PROFILE["layers"]
LENGTHS = PROFILE["lengths"]
WORLD, EXPERTS, TOP_K = 8, PROFILE["experts"], PROFILE["top_k"]
MAX_TOKENS = 4096
MAX_LOG_BYTES = 256 * 1024**2
ORDERS = (("S1", "S32", "L1", "L32"), ("L32", "L1", "S32", "S1"),
          ("S32", "L32", "S1", "L1"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""): h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(data)
    temp.replace(path)


def integer(x, low, high, label):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f"invalid {label}: {x!r}")
    return x


def finite(x, label):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError(f"non-finite/non-numeric {label}")
    return x


def workload(name):
    if name not in ORDERS[0]: raise ValueError("unknown frozen workload")
    return dict(id=name, input_tokens=512 if name.startswith("S") else 4096,
                output_tokens=128, concurrency=1 if name.endswith("1") else 32,
                requests=32 if name.endswith("1") else 128)


def schedule():
    return [dict(block_id=f"r{rep}-{name}", repetition=rep, **workload(name))
            for rep, order in enumerate(ORDERS) for name in order]


def validate_corpus(corpus):
    if corpus.get("schema") != "edmir.real-moe-corpus.v1": raise ValueError("wrong corpus")
    rows = corpus["requests"]
    if len(rows) != 128 or len({r["id"] for r in rows}) != 128:
        raise ValueError("128 unique source requests required")
    if len({r["content"] for r in rows}) != 128: raise ValueError("duplicate source text")
    counts = Counter(r["category"] for r in rows)
    if counts != {"zh":43, "en":43, "code":42}: raise ValueError("category imbalance")
    for i, r in enumerate(rows):
        if r["id"] != f"case-{i:03d}" or r["category"] != ("zh", "en", "code")[i % 3]:
            raise ValueError("request order changed")
        if not 18000 <= len(r["content"]) <= 200000:
            raise ValueError("source must be sufficiently long and bounded")
    if corpus["request_content_sha256"] != digest(rows): raise ValueError("corpus changed")
    return dict(source_requests=128, categories=dict(counts), token_ids_frozen=False)


def tokenize_corpus(corpus, tokenizer):
    """Offline-only caller supplies the pinned tokenizer; never fetch from HF.

    Keep the native template prefix and assistant suffix; truncate the tokenized
    user context, whose task instruction is at the beginning. Canonical transport
    is token IDs, not decoding and then re-tokenizing a truncated string.
    """
    validate_corpus(corpus)
    sentinel = "EDMIR_UNIQUE_TEMPLATE_BOUNDARY_20261117"
    options = dict(add_generation_prompt=True, enable_thinking=False, return_dict=False)
    rendered = tokenizer.apply_chat_template([dict(role="user", content=sentinel)],
                                            tokenize=False, **options)
    if rendered.count(sentinel) != 1: raise ValueError("ambiguous template boundary")
    prefix, suffix = rendered.split(sentinel)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not prefix_ids or not suffix_ids or len(prefix_ids) + len(suffix_ids) >= 512:
        raise ValueError("unexpected template geometry")
    output = []
    for row in corpus["requests"]:
        messages = [dict(role="user", content=row["content"])]
        full = tokenizer.apply_chat_template(messages, tokenize=True, **options)
        if full[:len(prefix_ids)] != prefix_ids or full[-len(suffix_ids):] != suffix_ids:
            raise ValueError("template token boundary changed")
        if len(full) < max(LENGTHS): raise ValueError("source too short for the longest workload")
        for length in LENGTHS:
            ids = full[:length-len(suffix_ids)] + suffix_ids
            if len(ids) != length: raise ValueError("exact token length failed")
            output.append(dict(id=f"{row['id']}-{length}", source_id=row["id"],
                               category=row["category"], input_tokens=length,
                               prompt_token_ids=ids, token_ids_sha256=digest(ids)))
    packet = dict(schema="edmir.real-moe-tokenized.v1", model=MODEL, revision=REVISION,
                  engine=ENGINE, source_sha256=corpus["request_content_sha256"],
                  template_options=options, prefix_ids=prefix_ids, suffix_ids=suffix_ids,
                  truncation="native full prompt token prefix + native assistant suffix",
                  requests=output)
    return packet


def model_vocab_size(model_dir):
    """Token-id bound from config.json. Multimodal configs (Qwen3.5) keep it under text_config."""
    config=json.loads((Path(model_dir)/"config.json").read_text())
    for scope in (config,config.get("text_config") or {}):
        if type(scope.get("vocab_size")) is int and scope["vocab_size"]>0: return scope["vocab_size"]
    raise ValueError("config.json has no vocab_size")


def validate_token_packet(packet, corpus, vocab_size):
    validate_corpus(corpus)
    if (packet.get("schema") != "edmir.real-moe-tokenized.v1" or
        (packet.get("model"),packet.get("revision"),packet.get("engine")) != (MODEL,REVISION,ENGINE)
        or packet.get("source_sha256") != corpus["request_content_sha256"]):
        raise ValueError("token packet provenance mismatch")
    rows=packet["requests"]
    expected={f"case-{i:03d}-{n}" for i in range(128) for n in LENGTHS}
    if len(rows)!=len(expected) or {r["id"] for r in rows}!=expected: raise ValueError("missing/duplicate tokens")
    suffix=packet["suffix_ids"];prefix=packet["prefix_ids"]
    if not suffix or not prefix: raise ValueError("template boundaries missing")
    for r in rows:
        n=r["input_tokens"];ids=r["prompt_token_ids"]
        if n not in LENGTHS or r["id"]!=f"{r['source_id']}-{n}" or len(ids)!=n:
            raise ValueError("token length/identity mismatch")
        if any(type(t) is not int or not 0<=t<vocab_size for t in ids): raise ValueError("token out of range")
        if ids[:len(prefix)]!=prefix or ids[-len(suffix):]!=suffix or digest(ids)!=r["token_ids_sha256"]:
            raise ValueError("template/token hash mismatch")
    return True


def input_rows(requests):
    """One ordered CPU scheduler snapshot. Recomputed positions stay explicit."""
    rows=[]
    for r in requests:
        if not isinstance(r["request_id"], str) or not r["request_id"]: raise ValueError("request ID required")
        start=integer(r["computed_tokens"],0,4608,"computed tokens")
        n=integer(r["scheduled_tokens"],0,MAX_TOKENS,"scheduled tokens")
        prompt=integer(r["prompt_tokens"],1,4096,"prompt tokens")
        if start+n>4608: raise ValueError("context overflow")
        rows.extend(dict(request_id=r["request_id"], position=p,
                         phase="prefill" if p<prompt else "decode") for p in range(start,start+n))
    if len(rows)>MAX_TOKENS: raise ValueError("per-rank token bound exceeded")
    if len({(r["request_id"],r["position"]) for r in rows})!=len(rows): raise ValueError("duplicate token")
    return rows


def classify(prefill, decode):
    if prefill and decode:return "mixed"
    if prefill:return "pure_prefill"
    if decode:return "pure_decode"
    return "empty"


class StepSampler:
    def __init__(self):
        self.steps=Counter();self.tokens=Counter();self.last_step=-1

    def observe(self, step, rank_metadata):
        integer(step,0,10**8,"global step")
        if step<=self.last_step:raise ValueError("non-monotone global step")
        self.last_step=step
        if [r["rank"] for r in rank_metadata]!=list(range(WORLD)):
            raise ValueError("all ordered source ranks required")
        if len({r["block_id"] for r in rank_metadata})!=1:raise ValueError("rank block mismatch")
        pref=sum(integer(r["prefill"],0,MAX_TOKENS,"prefill") for r in rank_metadata)
        dec=sum(integer(r["decode"],0,MAX_TOKENS,"decode") for r in rank_metadata)
        category=classify(pref,dec);self.steps[category]+=1
        self.tokens["prefill"]+=pref;self.tokens["decode"]+=dec
        return category,category!="empty" and self.steps[category]<=3


def route_slice_bounds(observed_rows, *, rank, sizes, real_count, layout):
    integer(rank,0,WORLD-1,"rank")
    if len(sizes)!=WORLD or any(type(v) is not int or not 0<=v<=MAX_TOKENS for v in sizes):
        raise ValueError("invalid DP padded sizes")
    if layout not in ("local", "gathered_dp"):raise ValueError("routing layout not established")
    expected=sizes[rank] if layout=="local" else sum(sizes)
    integer(real_count,0,MAX_TOKENS,"real source rows")
    if observed_rows!=expected or real_count>sizes[rank]:
        raise ValueError("router rows differ from proven layout")
    offset=0 if layout=="local" else sum(sizes[:rank])
    return offset,offset+real_count


def slice_routes(ids, weights, *, rank, sizes, tokens, layout):
    """AgRs may route *after* all-gather. Select this rank's unique source slice."""
    if len(weights)!=len(ids):raise ValueError("router weight row count mismatch")
    offset,end=route_slice_bounds(len(ids),rank=rank,sizes=sizes,real_count=len(tokens),layout=layout)
    expected=len(ids)
    selected=ids[offset:offset+len(tokens)];selected_weights=weights[offset:offset+len(tokens)]
    counts={p:[0]*EXPERTS for p in ("prefill","decode")}
    for token,row,w in zip(tokens,selected,selected_weights):
        if token["phase"] not in counts:raise ValueError("unknown token phase")
        if len(row)!=TOP_K or len(set(row))!=TOP_K or any(type(e) is not int or not 0<=e<EXPERTS for e in row):
            raise ValueError("invalid/duplicate top-k expert")
        if len(w)!=TOP_K or any(finite(v,"route weight")<0 for v in w):raise ValueError("invalid routing weight")
        for e in row:counts[token["phase"]][e]+=1
    return dict(token_count=len(tokens), phase_counts=counts, topk_ids=selected,
                topk_weights=selected_weights, routing_layout=layout,
                source_slice_offset=offset, observed_router_rows=expected,
                padding_or_dummy_rows=sizes[rank]-len(tokens))


class JsonlWriter:
    """Per-worker gzip journal with bounded uncompressed size; partial runs survive."""
    def __init__(self,path,limit=MAX_LOG_BYTES):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.f=gzip.open(self.path,"xb");self.limit=limit;self.bytes=0
    def write(self,row):
        b=(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n").encode()
        if self.bytes+len(b)>self.limit:raise ValueError("journal byte budget exceeded")
        self.f.write(b);self.f.flush();self.bytes+=len(b)
    def close(self):self.f.close()


def read_journal(path,limit=MAX_LOG_BYTES):
    total=0
    with gzip.open(path,"rt") as f:
        while True:
            line=f.readline(8*1024**2)
            if not line:break
            total+=len(line.encode())
            if not line.endswith("\n") or total>limit:raise ValueError("oversized/incomplete journal")
            yield json.loads(line)


def summarize_routes(events):
    """All ranks/layers required; synthetic fixtures never become observed data."""
    steps={};routes={};kernels=defaultdict(list);origins=set();identities=set()
    for r in events:
        origins.add(r["origin"]);identities.add((r["run_id"],r["block_id"]))
        if r["origin"] not in ("synthetic_test","real_router"):raise ValueError("invalid origin")
        key=(r["step"],r["rank"])
        integer(r["rank"],0,7,"rank");integer(r["step"],0,10**8,"step")
        if r["kind"]=="step":
            if key in steps:raise ValueError("duplicate rank step")
            steps[key]=r
        elif r["kind"]=="route":
            integer(r["layer"],1,46,"layer");key+=(r["layer"],)
            if key in routes:raise ValueError("duplicate route")
            routes[key]=r
        elif r["kind"]=="kernel":kernels[key+(r["layer"],)].append(r)
        elif r["kind"]=="failure":raise ValueError("worker failure present")
    if len(origins)!=1 or len(identities)!=1 or not steps:raise ValueError("mixed/empty run provenance")
    allsteps=sorted({k[0] for k in steps});logical=Counter();hist=Counter();phase_hist=Counter()
    empty=0;routed=0;sampled=0;missing_geometry=[];unique_tokens=0;balance=[]
    for step in allsteps:
        if any((step,r) not in steps for r in range(WORLD)):raise ValueError("missing rank in step")
        ss=[steps[step,r] for r in range(WORLD)]
        if len({(s["category"],s["selected"]) for s in ss})!=1:raise ValueError("rank sampling mismatch")
        keys=[(t["request_id"],t["position"]) for s in ss for t in s["tokens"]]
        if len(set(keys))!=len(keys):raise ValueError("token replicated across source ranks")
        phases=Counter(t["phase"] for s in ss for t in s["tokens"])
        if not set(phases)<={"prefill","decode"}:raise ValueError("unknown phase")
        if classify(phases["prefill"],phases["decode"])!=ss[0]["category"]:raise ValueError("wrong mixed classification")
        unique_tokens+=len(keys)
        if not ss[0]["selected"]:
            if any(k[0]==step for k in routes):raise ValueError("routes outside selected steps")
            continue
        sampled+=1
        for layer in LAYERS:
            by_phase={p:[0]*EXPERTS for p in ("prefill","decode")}
            for rank,s in enumerate(ss):
                k=(step,rank,layer)
                if k not in routes:raise ValueError("missing route rank/layer")
                row=routes[k]
                rebuilt=slice_routes(row["topk_ids"],row["topk_weights"],rank=rank,
                    sizes=[len(steps[step,i]["tokens"]) for i in range(WORLD)],
                    tokens=s["tokens"],layout="local")
                if row["phase_counts"]!=rebuilt["phase_counts"] or row["token_count"]!=len(s["tokens"]):
                    raise ValueError("route histogram disagrees with raw IDs")
                for p in by_phase:
                    by_phase[p]=[a+b for a,b in zip(by_phase[p],row["phase_counts"][p])]
                if {v["projection"] for v in kernels[k]}!={"gate_up","down"}:
                    missing_geometry.append(k)
            total=[a+b for a,b in zip(by_phase["prefill"],by_phase["decode"])]
            if sum(total)!=len(keys)*TOP_K:raise ValueError("global routing conservation failed")
            routed+=sum(total);empty+=sum(v==0 for v in total)
            mean=sum(total)/EXPERTS
            balance.append(dict(step=step,layer=layer,max_over_mean=max(total)/mean if mean else None))
            for rank in range(WORLD):
                observed=kernels[(step,rank,layer)]
                if {v["projection"] for v in observed}!={"gate_up","down"}:continue
                if len(observed)!=2:raise ValueError("duplicate or unsupported chunked projection calls")
                for item in observed:
                    projection=item["projection"]
                    expected_dims={"gate_up":[8,3072,2048],"down":[8,2048,1536]}[projection]
                    if (item.get("weight_shape")!=expected_dims or item.get("physical_layout_verified") is not True
                        or item.get("input_dtype")!="torch.bfloat16" or item.get("weight_dtype")!="torch.bfloat16"):
                        raise ValueError("invalid observed kernel geometry/dtype")
                    emap=item.get("expert_map")
                    if not isinstance(emap,list) or len(emap)!=EXPERTS or sorted(v for v in emap if v>=0)!=list(range(8)):
                        raise ValueError("expert ownership map missing/invalid")
                    slots=item["local_expert_assignment"]
                    for expert,local in enumerate(emap):
                        if local<0:continue
                        entry=slots.get(str(local),slots.get(local,{"real_rows":0,"block_slots":0,"dummy_or_padding_slots":0}))
                        if entry["real_rows"]!=total[expert] or entry["block_slots"]!=entry["real_rows"]+entry["dummy_or_padding_slots"]:
                            raise ValueError("backend assignment disagrees with routed real rows")
            for m in total:
                hist[(ss[0]["category"],m)]+=1
                if m:
                    logical[(ss[0]["category"],"gate_up",m,2048,3072)]+=1
                    logical[(ss[0]["category"],"down",m,1536,2048)]+=1
            for phase,counts in by_phase.items():
                for m in counts:phase_hist[(ss[0]["category"],phase,m)]+=1
    if any(k[0:2] not in steps for k in routes):raise ValueError("orphan route")
    return dict(schema=SCHEMA,origin=origins.pop(),run_id=next(iter(identities))[0],
        sampled_steps=sampled,complete_steps=len(allsteps),sampled_routed_rows=routed,
        captured_token_observations=unique_tokens,empty_expert_step_layers=empty,
        M_histogram=[dict(category=c,M=m,count=n) for (c,m),n in sorted(hist.items())],
        phase_M_histogram=[dict(category=c,phase=p,M=m,count=n) for (c,p,m),n in sorted(phase_hist.items())],
        logical_shapes=[dict(category=c,projection=p,M=m,K=k,N=n,count=v) for (c,p,m,k,n),v in sorted(logical.items())],
        max_mean_load=balance,kernel_geometry_complete=not missing_geometry,
        missing_kernel_geometry=missing_geometry,execution_identity_coverage=None,
        measured_time_coverage=None,edmir_integrated=False,
        scope="sampled routed expert projections; mixed uses aggregate M; phase slices are accounting only")


async def collect_request(generate, request, request_id, *, timeout=180):
    start=time.perf_counter();last=0;updates=[];final=None;error=None
    try:
        async def consume():
            nonlocal last,final
            async for event in generate(request,request_id):
                now=time.perf_counter()-start;n=len(event["token_ids"])
                if n<last or n>128:raise ValueError("non-monotone/excess output tokens")
                if event.get("prompt_token_ids")!=request["prompt_token_ids"]:raise ValueError("engine changed prompt IDs")
                if n>last:updates.append(dict(seconds=now,new_tokens=n-last,cumulative_tokens=n))
                last=n;final=event
            if not final or not final.get("finished") or last!=128:raise ValueError("incomplete fixed-length generation")
        await asyncio.wait_for(consume(),timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:error=repr(exc)
    finish=time.perf_counter()-start
    first=updates[0]["seconds"] if updates else None
    one_per_event=bool(updates) and all(e["new_tokens"]==1 for e in updates)
    return dict(schema=SCHEMA,request_id=request_id,source_id=request["source_id"],
        input_tokens=len(request["prompt_token_ids"]),output_tokens=last,
        prompt_token_ids_sha256=digest(request["prompt_token_ids"]),
        output_token_ids=final["token_ids"] if final else [],success=error is None,error=error,
        output_updates=updates,wall_seconds=finish,ttft_seconds=first,
        tpot_seconds=(updates[-1]["seconds"]-first)/(last-1) if last>1 else None,
        inter_event_seconds=[b["seconds"]-a["seconds"] for a,b in zip(updates,updates[1:])],
        itl_seconds=[b["seconds"]-a["seconds"] for a,b in zip(updates,updates[1:])] if one_per_event else None,
        clock="client monotonic",timing_scope="AsyncLLM streaming including engine queue, excluding HTTP/tokenization")


async def collect_block(generate, requests, *, block_id, concurrency, timeout=180, checkpoint=None):
    if not requests or len(requests)>128:raise ValueError("bounded block required")
    integer(concurrency,1,32,"concurrency")
    semaphore=asyncio.Semaphore(concurrency)
    completed={}
    async def one(i,r):
        async with semaphore:
            result=await collect_request(generate,r,f"{block_id}-{i:03d}",timeout=timeout)
            completed[i]=result
            return result
    started=time.perf_counter()
    try:
        rows=await asyncio.gather(*(one(i,r) for i,r in enumerate(requests)))
    except BaseException:
        if checkpoint is not None:
            save(checkpoint,dict(schema=SCHEMA,block_id=block_id,status="interrupted",
                expected_requests=len(requests),completed_requests=[completed[k] for k in sorted(completed)],
                unfinished_request_ids=[f"{block_id}-{i:03d}" for i in range(len(requests)) if i not in completed],
                elapsed_seconds=time.perf_counter()-started))
        raise
    elapsed=time.perf_counter()-started
    return dict(schema=SCHEMA,block_id=block_id,seconds=elapsed,requests=rows,
                expected_requests=len(requests),successes=sum(r["success"] for r in rows),
                successful_output_tokens_per_second=sum(r["output_tokens"] for r in rows if r["success"])/elapsed,
                includes_queue_fill_and_drain=True)


def percentile(values,q):
    if not values:return None
    x=sorted(values);i=(len(x)-1)*q;lo=int(i);hi=math.ceil(i)
    return x[lo]+(x[hi]-x[lo])*(i-lo)


def summarize_blocks(blocks):
    expected={s["block_id"]:s for s in schedule()}
    if len({b["block_id"] for b in blocks})!=len(blocks):raise ValueError("duplicate block")
    report=[];seen=set()
    for b in blocks:
        if b["block_id"] not in expected:raise ValueError("unregistered block")
        spec=expected[b["block_id"]];rows=b["requests"];seen.add(b["block_id"])
        if len({r["request_id"] for r in rows})!=len(rows):raise ValueError("duplicate request")
        if len(rows)!=spec["requests"]:raise ValueError("missing request result")
        for r in rows:
            if r["input_tokens"]!=spec["input_tokens"]:raise ValueError("wrong workload length")
            if r["success"] and (r["output_tokens"]!=128 or r["error"] is not None):raise ValueError("invalid success")
            for k in ("wall_seconds","ttft_seconds","tpot_seconds"):
                if r[k] is not None and finite(r[k],k)<0:raise ValueError("negative time")
        good=[r for r in rows if r["success"]]
        duration=finite(b["seconds"],"block time")
        if duration<=0:raise ValueError("non-positive block time")
        report.append(dict(block_id=b["block_id"],workload=spec["id"],n=len(rows),successes=len(good),
            tokens_per_second=sum(r["output_tokens"] for r in good)/duration,
            ttft_p50=percentile([r["ttft_seconds"] for r in good],.5),
            ttft_p95=percentile([r["ttft_seconds"] for r in good],.95),
            tpot_p50=percentile([r["tpot_seconds"] for r in good],.5),
            latency_p95=percentile([r["wall_seconds"] for r in good],.95)))
    drift={}
    for name in ORDERS[0]:
        rr=[r for r in report if r["workload"]==name]
        for field in ("tokens_per_second","ttft_p50"):
            values=[r[field] for r in rr if r[field] is not None]
            ratio=(max(values)-min(values))/statistics.median(values) if len(values)==3 and min(values)>0 else None
            drift[f"{name}:{field}"]=dict(range_over_median=ratio,unstable=ratio>.1 if ratio is not None else None)
    return dict(schema=SCHEMA,blocks=report,missing_blocks=sorted(set(expected)-seen),drift=drift,
                complete=len(seen)==12 and all(r["successes"]==r["n"] for r in report),
                predictor_evaluated=False,edmir_integrated=False)


def union_intervals(intervals):
    out=[]
    for start,end in sorted(intervals):
        if finite(start,"start")<0 or finite(end,"end")<start:raise ValueError("invalid interval")
        if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
        else:out.append([start,end])
    return out


def timeline_overlap(records):
    """Same GPU/clock-domain only. Observed overlap is not causal speedup."""
    records=list(records)
    buckets=defaultdict(lambda:defaultdict(list))
    for r in records:
        if r.get("origin") not in ("nsys","synthetic_test"):raise ValueError("measured trace origin required")
        if r["kind"] not in ("communication","compute","unknown"):raise ValueError("explicit classification required")
        integer(r["rank"],0,7,"rank")
        buckets[(r["rank"],r["clock_domain"])][r["kind"]].append((r["start_ns"],r["end_ns"]))
    result=[]
    for (rank,clock),groups in sorted(buckets.items()):
        comm=union_intervals(groups["communication"]);compute=union_intervals(groups["compute"])
        intersection=[];i=j=0
        while i<len(comm) and j<len(compute):
            a,b=comm[i],compute[j];s=max(a[0],b[0]);e=min(a[1],b[1])
            if s<e:intersection.append((s,e))
            if a[1]<=b[1]:i+=1
            else:j+=1
        tc=sum(e-s for s,e in comm);tg=sum(e-s for s,e in compute);ov=sum(e-s for s,e in intersection)
        result.append(dict(rank=rank,clock_domain=clock,communication_union_ns=tc,compute_union_ns=tg,
            overlap_ns=ov,visible_overlap_ratio=ov/tc if tc else None,unknown_intervals=len(groups["unknown"])))
    return dict(ranks=result,all_eight_ranks_present={r["rank"] for r in records}==set(range(8)),
                interpretation="same-rank interval intersection; not hidden cost or end-to-end speedup")
