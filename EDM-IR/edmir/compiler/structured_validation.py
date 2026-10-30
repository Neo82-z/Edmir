"""Label-free, deterministic whole-shape development folds.

This validates membership and candidate coverage, not measurement quality or
unseen-workload generalization. Execution-context IDs are caller declarations.
"""
from collections import defaultdict

SPLITS = ("train", "calibration", "evaluation")
BASELINES = ("host", "host_merged_cutlass", "host_merged_cublaslt")


def validate_partition_records(records, required_candidates):
    if not isinstance(records, (list, tuple)) or not 1 <= len(records) <= 16384:
        raise ValueError("bounded record sequence required")
    if not isinstance(required_candidates, dict) or not required_candidates:
        raise ValueError("declared candidate sets per execution context required")
    for names in required_candidates.values():
        if (not isinstance(names, (list, tuple)) or not 2 <= len(names) <= 14
                or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n for n in names)):
            raise ValueError("unique bounded candidate IDs required")
    groups, blocks, seen, worlds = {}, defaultdict(set), set(), defaultdict(set)
    for r in records:
        if set(r) != {"shape", "execution_context", "candidate_id", "replicate_id", "split"}:
            raise ValueError("partition metadata only; no latency labels or undeclared fields")
        shape = tuple(r["shape"])
        if (len(shape) != 4 or any(type(x) is not int or not 0 < x <= 2**20 for x in shape)
                or shape[0] not in (4, 8)):
            raise ValueError("bounded P4/P8 shape required")
        ctx, candidate, rep, split = (r[k] for k in ("execution_context", "candidate_id", "replicate_id", "split"))
        if (ctx not in required_candidates or candidate not in required_candidates[ctx]
                or not isinstance(rep, str) or not rep.strip() or split not in SPLITS):
            raise ValueError("undeclared context, candidate, repeat, or split")
        worlds[ctx].add(shape[0])
        if len(worlds[ctx]) > 1:
            raise ValueError("one execution context cannot describe different GPU counts")
        # Across contexts too: using the same shape for development and evaluation
        # needs an explicitly different transfer protocol, not this validator.
        if shape in groups and groups[shape] != split:
            raise ValueError("candidate/repeat leakage across whole-shape partitions")
        groups[shape] = split
        key = (ctx, shape, candidate, rep)
        if key in seen:
            raise ValueError("duplicate candidate repetition")
        seen.add(key); blocks[(ctx, shape, rep)].add(candidate)
    for (ctx, _, _), names in blocks.items():
        if names != set(required_candidates[ctx]):
            raise ValueError("incomplete candidate block; do not silently drop failures")
    return {"rows": len(records), "shapes": len(groups), "complete_blocks": len(blocks),
            "shapes_per_split": {s: sum(v == s for v in groups.values()) for s in SPLITS},
            "measurement_quality_verified": False, "independent_generalization_claim": False}


def development_folds(records, required_candidates):
    """Leave out one entire shape; ordering does not depend on labels or row order."""
    validate_partition_records(records, required_candidates)
    if any(r.get("split") != "train" for r in records):
        raise ValueError("fold input must be declared development/training metadata")
    shapes = sorted({tuple(r["shape"]) for r in records})
    if not 2 <= len(shapes) <= 64:
        raise ValueError("2..64 development shapes required")
    canonical = sorted(records, key=lambda r: (tuple(r["shape"]), r["execution_context"],
                                              r["replicate_id"], r["candidate_id"]))
    result = []
    for held_out in shapes:
        fold = [{**r, "split": "evaluation" if tuple(r["shape"]) == held_out else "train"}
                for r in canonical]
        result.append({"held_out_shape": list(held_out), "records": fold,
                       "audit": validate_partition_records(fold, required_candidates),
                       "purpose": "development_ablation_only"})
    return result
