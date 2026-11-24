"""Per-model capture receipts, routing layout, and independent FP32 reference."""
import json
from pathlib import Path
import re

from make_design import read, sha

WORLD = 8
TILE_M = 64
SCHEMA = "edmir.gate-up-snapshot.v3"


def snapshot_spec(design, snapshot):
    """Geometry of one captured layer: E routed experts, E/8 per rank, top-k routing, K by N gate/up."""
    row = next(row for row in read(design / "snapshots.csv") if row["snapshot_id"] == snapshot)
    experts = int(row["routed_experts"])
    if experts % WORLD:
        raise ValueError("routed experts must divide evenly over eight ranks")
    return dict(snapshot=snapshot, width=row["width_id"], model=row["model"], revision=row["model_revision"],
                layer=int(row["layer_index"]), phase=row["category"], ordinal=int(row["step_ordinal"]),
                experts=experts, local=experts//WORLD, top_k=int(row["top_k"]), k=int(row["k"]), n=int(row["n"]))


def expert_map(spec, rank):
    """vLLM's linear placement: rank r owns experts [r*E/8, (r+1)*E/8)."""
    return [e % spec["local"] if e//spec["local"] == rank else -1 for e in range(spec["experts"])]


def locate(roots, snapshot):
    """The capture directory holding this snapshot; each model is captured into its own root."""
    found = [Path(root) for root in roots or () if (Path(root)/snapshot).is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"{snapshot}: expected one capture directory, found {len(found)}")
    return found[0]


def load_receipts(root, design, snapshot):
    spec = snapshot_spec(design, snapshot)
    directory = root / snapshot
    k, n, top_k = spec["k"], spec["n"], spec["top_k"]
    rows = []
    for rank in range(WORLD):
        path = directory / f"rank-{rank}.json"
        row = json.loads(path.read_text())
        expected = dict(schema=SCHEMA, origin="real_vllm", snapshot=snapshot,
                        rank=rank, layer=spec["layer"], phase=spec["phase"],
                        phase_ordinal=spec["ordinal"], model=spec["model"], model_revision=spec["revision"],
                        expert_map=expert_map(spec, rank),
                        gate_up_accumulator="float32")
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{snapshot}/rank {rank}: capture identity differs from design")
        rows.append(row)
    first = rows[0]
    for field in ("run_id", "step", "padded_sizes", "source_lock_sha256", "request_sha256", "native_config", "native_tune", "native_strides"):
        if not first.get(field) or any(r.get(field) != first[field] for r in rows):
            raise ValueError(f"{snapshot}: ranks disagree on {field}")
    tune = first["native_tune"]
    if tune.get("kind") not in {"default","tuned"} or (tune["kind"]=="tuned"
            and not re.fullmatch(r"[a-f0-9]{64}",tune.get("sha256",""))):
        raise ValueError(f"{snapshot}: invalid native tuning provenance")
    if first["native_strides"] != {"A":[k,1],"B":[n*k,k,1],"C":[top_k*n,n,1]}:
        raise ValueError(f"{snapshot}: native strides are not supported by replay")
    sizes = first["padded_sizes"]
    if len(sizes) != WORLD or any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError(f"{snapshot}: invalid rank sizes")
    if any(len(r["tokens"]) > sizes[i] for i, r in enumerate(rows)):
        raise ValueError(f"{snapshot}: inconsistent padding")
    if not any(r["tokens"] for r in rows):
        raise ValueError(f"{snapshot}: missing live tokens")
    for r in rows:
        for f in ("source_lock_sha256", "request_sha256", "snapshot_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}",r.get(f,"")):
                raise ValueError(f"{snapshot}: invalid {f}")
    phase = spec["phase"].removeprefix("pure_")
    if any(t.get("phase") != phase for r in rows for t in r["tokens"]):
        raise ValueError(f"{snapshot}: mixed phases")
    return rows


def load_rank(root, snapshot, receipt, spec):
    import torch

    path = Path(root) / snapshot / f"rank-{receipt['rank']}.pt"
    if sha(path) != receipt["snapshot_sha256"]:
        raise ValueError(f"{path}: tensor digest mismatch")
    rank = receipt["rank"]
    size = receipt["padded_sizes"][rank]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_tensors(payload, size, len(receipt["tokens"]), spec)
    validate_alignment(payload,receipt["native_config"],sum(receipt["padded_sizes"])*spec["top_k"],spec["local"])
    return payload


def validate_alignment(payload, config, slots, local):
    import torch

    ids,experts,post = (payload[k] for k in ("sorted_token_ids","expert_ids","num_tokens_post_padded"))
    bm = config["BLOCK_SIZE_M"]
    if (any(x.dtype!=torch.int32 or x.ndim!=1 or not x.is_contiguous() for x in (ids,experts,post))
            or post.numel()!=1 or type(bm) is not int or bm<=0):
        raise ValueError("invalid native launch metadata")
    end = post.item()
    if (not 0<=end<=ids.numel() or end%bm or end//bm>experts.numel()
            or ids[:end].lt(0).any() or ids[:end].gt(slots).any()
            or experts[:end//bm].lt(-1).any() or experts[:end//bm].ge(local).any()):
        raise ValueError("invalid captured native alignment extent")


def validate_tensors(payload, size, live, spec):
    import torch

    k, n, top_k = spec["k"], spec["n"], spec["top_k"]
    for key, shape, dtypes in (
        ("hidden", (size,k), (torch.bfloat16,)),
        ("expert_weights", (spec["local"],n,k), (torch.bfloat16,)),
        ("ids", (size,top_k), (torch.int32,torch.int64)),
        ("router_weights", (size,top_k), (torch.float32,)),
    ):
        t = payload[key]
        if (tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous()
                or not torch.isfinite(t).all().item()):
            raise ValueError(f"invalid {key}: shape, dtype, layout or nonfinite")
    ids = payload["ids"]
    if (ids.lt(0).any() or ids.ge(spec["experts"]).any()
            or ids[:live].sort(dim=1).values.diff(dim=1).eq(0).any()
            or payload["router_weights"].lt(0).any()):
        raise ValueError("invalid expert ids or router weights")


def layout(rank_ids, live_counts, local=8):
    """Source -> destination -> local expert; rows retain source/token/top-k slot."""
    result = []
    for ids, live in zip(rank_ids, live_counts, strict=True):
        by_expert = [[[] for _ in range(local)] for _ in range(WORLD)]
        for token, routes in enumerate(ids[:live]):
            for slot, expert in enumerate(routes):
                by_expert[expert//local][expert%local].append((token, slot))
        result.append(by_expert)
    return result


def tiles(counts):
    offset = 0
    work = []
    for expert,count in enumerate(counts):
        work.extend((i,offset+count,expert) for i in range(offset,offset+count,TILE_M))
        offset += count
    return work


def reference(hidden_all, weights, route_layout, rank):
    """Canonical GEMMs use the original source activations and FP32 arithmetic."""
    import torch

    outputs = []
    for hidden, routes in zip(hidden_all, route_layout, strict=True):
        for expert, assignments in enumerate(routes[rank]):
            if assignments:
                rows = torch.tensor([token for token, _ in assignments], device=hidden.device)
                outputs.append(hidden.index_select(0, rows).float() @ weights[expert].float().T)
    return torch.cat(outputs) if outputs else weights.new_empty((0,weights.shape[1]),dtype=torch.float32)


def correctness(result, expected, *, dtype=None):
    import torch

    target = torch.bfloat16 if dtype is None else dtype
    if result.dtype != target or result.shape != expected.shape:
        raise ValueError("projection output shape or dtype differs from the contract")
    got = result.float()
    nonfinite = (~torch.isfinite(got) | ~torch.isfinite(expected)).sum().item()
    if nonfinite or not got.numel():
        return None if nonfinite else 0.0, nonfinite
    return ((got-expected).abs()/expected.abs().clamp_min(1.)).max().item(), nonfinite
