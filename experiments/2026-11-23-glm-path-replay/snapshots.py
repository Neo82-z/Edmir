"""GLM layer-23 capture receipts, routing layout, and independent FP32 reference."""
import json
from pathlib import Path
import re

from make_design import read, sha

WORLD = 8
TOP_K = 4
TILE_M = 64


def load_receipts(root, design, snapshot):
    spec = next(row for row in read(design / "snapshots.csv") if row["snapshot_id"] == snapshot)
    directory = root / snapshot
    rows = []
    for rank in range(WORLD):
        path = directory / f"rank-{rank}.json"
        row = json.loads(path.read_text())
        expected = dict(schema="edmir.glm-gate-up-snapshot.v2", origin="real_vllm", snapshot=snapshot,
                        rank=rank, layer=int(spec["layer_index"]), phase=spec["category"],
                        phase_ordinal=int(spec["step_ordinal"]), model_revision=spec["model_revision"],
                        expert_map=[e%8 if e//8==rank else -1 for e in range(64)],
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
    if first["native_strides"] != {"A":[2048,1],"B":[3072*2048,2048,1],"C":[4*3072,3072,1]}:
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
    phase = spec["category"].removeprefix("pure_")
    if any(t.get("phase") != phase for r in rows for t in r["tokens"]):
        raise ValueError(f"{snapshot}: mixed phases")
    return rows


def load_rank(root, snapshot, receipt):
    import torch

    path = Path(root) / snapshot / f"rank-{receipt['rank']}.pt"
    if sha(path) != receipt["snapshot_sha256"]:
        raise ValueError(f"{path}: tensor digest mismatch")
    rank = receipt["rank"]
    size = receipt["padded_sizes"][rank]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_tensors(payload, size, len(receipt["tokens"]), 2048, 3072)
    validate_alignment(payload,receipt["native_config"],sum(receipt["padded_sizes"])*TOP_K)
    return payload


def validate_alignment(payload, config, slots):
    import torch

    ids,experts,post = (payload[k] for k in ("sorted_token_ids","expert_ids","num_tokens_post_padded"))
    bm = config["BLOCK_SIZE_M"]
    if (any(x.dtype!=torch.int32 or x.ndim!=1 or not x.is_contiguous() for x in (ids,experts,post))
            or post.numel()!=1 or type(bm) is not int or bm<=0):
        raise ValueError("invalid native launch metadata")
    end = post.item()
    if (not 0<=end<=ids.numel() or end%bm or end//bm>experts.numel()
            or ids[:end].lt(0).any() or ids[:end].gt(slots).any()
            or experts[:end//bm].lt(-1).any() or experts[:end//bm].ge(8).any()):
        raise ValueError("invalid captured native alignment extent")


def validate_tensors(payload, size, live, k, n):
    import torch

    for key, shape, dtypes in (
        ("hidden", (size,k), (torch.bfloat16,)),
        ("expert_weights", (8,n,k), (torch.bfloat16,)),
        ("ids", (size,TOP_K), (torch.int32,torch.int64)),
        ("router_weights", (size,TOP_K), (torch.float32,)),
    ):
        t = payload[key]
        if (tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous()
                or not torch.isfinite(t).all().item()):
            raise ValueError(f"invalid {key}: shape, dtype, layout or nonfinite")
    ids = payload["ids"]
    if (ids.lt(0).any() or ids.ge(64).any()
            or ids[:live].sort(dim=1).values.diff(dim=1).eq(0).any()
            or payload["router_weights"].lt(0).any()):
        raise ValueError("invalid expert ids or router weights")


def layout(rank_ids, live_counts):
    """Source -> destination -> local expert; rows retain source/token/top-k slot."""
    result = []
    for ids, live in zip(rank_ids, live_counts, strict=True):
        by_expert = [[[] for _ in range(8)] for _ in range(WORLD)]
        for token, routes in enumerate(ids[:live]):
            for slot, expert in enumerate(routes):
                by_expert[expert//8][expert%8].append((token, slot))
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
