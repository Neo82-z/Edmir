#!/usr/bin/env python3
"""Generate the frozen design tables for the final 8xA100 round. Offline only.

Run: python3 make_design.py --out ../design

Writes cells.csv, snapshots.csv, arms.csv, comparisons.csv, blocks.csv,
policies.json and queue.csv. Arm order inside a process is a stable keyed
permutation; the paired process uses the exact reverse order. Re-running the
script must reproduce byte-identical files (checked by tests/test_package.py).
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DESIGN_ID = "a100-final-round-crossover-v1"
RANKS = 8
K = 2048
M_BINS = tuple(2 ** i for i in range(11))  # 1 .. 1024 rows per source->destination
SEEDS = (20261120, 20261121, 20261122)
REPS = ("rep1", "rep2", "rep3")

# Projection widths of the three models (gate/up fused, N = 2 * moe_intermediate_size).
WIDTHS = (
    ("GLM", "zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67", 3072),
    ("QWEN36", "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", 1024),
    ("DSV2L", "deepseek-ai/DeepSeek-V2-Lite-Chat", "85864749cd611b4353ce1decdb286193298f64c7", 2816),
)
# Cells of K2048/N1024 already measured in earlier rounds; M64 was a calibration point.
HISTORICAL = {("QWEN36", 32): "historical_overlap", ("QWEN36", 64): "historical_calibration"}

# Real snapshots: same request set (S32 = 32 prompts x 512 input tokens) for every model.
MODELS = (
    # width_id, routed experts, top-k, num_hidden_layers, first MoE layer, capture layer
    ("GLM", 64, 4, 47, 1, 23),
    ("QWEN36", 256, 8, 40, 0, 20),
    ("DSV2L", 64, 6, 27, 1, 13),
)
PHASES = (
    ("PREFILL", "pure_prefill", 1, "first formal pure-prefill step"),
    ("DECODE", "pure_decode", 8, "eighth formal pure-decode step"),
)

ARMS = (
    # arm_id, stage, kind, implementation, sources_per_launch, split_k, n_tile_mult, gpu_status, block_type
    ("D_Q1_G2", "balanced", "direct", "per-source direct CUTLASS SM80 (J1)", 1, 1, 2, "gpu_run", "core"),
    ("D_Q4_G2", "balanced", "direct", "per-source direct CUTLASS SM80 (J1), P2 fixed policy", 1, 4, 2, "gpu_run", "core"),
    ("BNEW_Q1_G2", "balanced", "staged", "ncclAlltoAll to local + same J1 consumer as D_Q1_G2", 1, 1, 2, "new", "core"),
    ("BNEW_Q4_G2", "balanced", "staged", "ncclAlltoAll to local + same J1 consumer as D_Q4_G2", 1, 4, 2, "new", "core"),
    ("SLT", "balanced", "staged", "single ncclAlltoAll + tuned cuBLASLt (<=8 heuristics)", "", "", "", "new", "core"),
    ("HM_CUTLASS", "balanced", "staged", "host merged CUTLASS (historical baseline)", "", "", "", "gpu_run", "core"),
    ("HM_LT", "balanced", "staged", "host merged cuBLASLt (historical baseline)", "", "", "", "gpu_run", "core"),
    ("D_Q1_G1", "balanced", "direct", "per-source direct CUTLASS SM80 q1/g1 (J1)", 1, 1, 1, "gpu_run", "core"),
    ("BOLD_Q1_G1", "balanced", "staged", "pipeline: per-pair NCCL -> staging -> CUTLASS 64x128x32", 1, 1, 1, "gpu_run", "core"),
    ("J8_Q1_G2", "balanced", "direct", "source-parallel grouped GEMM (J8)", 8, 1, 2, "compiled_only", "ablation_j8"),
    ("FLUX_V2", "balanced", "staged_fused", "FLUX GemmGroupedV2AGScatterOp (COMET, SM80)", "", "", "", "not_built", "flux"),
    ("NATIVE", "snapshot", "staged", "vLLM 0.20.1 all-gather + Triton fused_moe gate/up", "", "", "", "gpu_run", "core"),
    ("D_Q1_G2_R", "snapshot", "direct", "ragged multi-expert direct, J1 q1/g2, BF16", 1, 1, 2, "new", "core"),
    ("D_Q4_G2_R", "snapshot", "direct", "ragged multi-expert direct, J1 q4/g2, BF16", 1, 4, 2, "new", "core"),
    ("BNEW_Q1_G2_R", "snapshot", "staged", "per-assignment AlltoAll + same consumer as D_Q1_G2_R", 1, 1, 2, "new", "core"),
    ("BNEW_Q4_G2_R", "snapshot", "staged", "per-assignment AlltoAll + same consumer as D_Q4_G2_R", 1, 4, 2, "new", "core"),
    ("FLUX_V2_R", "snapshot", "staged_fused", "FLUX GemmGroupedV2AGScatterOp on real routing", "", "", "", "not_built", "flux"),
    ("J8_Q1_G2_R", "snapshot", "direct", "ragged J8 grouped GEMM (only if implemented)", 8, 1, 2, "not_implemented", "ablation_j8"),
)

BLOCK_TYPES = {
    "balanced": {
        "core": ("D_Q1_G2", "D_Q4_G2", "BNEW_Q1_G2", "BNEW_Q4_G2", "SLT", "HM_CUTLASS", "HM_LT",
                 "D_Q1_G1", "BOLD_Q1_G1"),
        "flux": ("SLT", "FLUX_V2"),
        "ablation_j8": ("D_Q1_G2", "J8_Q1_G2", "SLT"),
    },
    "snapshot": {
        "core": ("NATIVE", "D_Q1_G2_R", "D_Q4_G2_R", "BNEW_Q1_G2_R", "BNEW_Q4_G2_R"),
        "flux": ("NATIVE", "FLUX_V2_R"),
        "ablation_j8": ("D_Q1_G2_R", "J8_Q1_G2_R", "NATIVE"),
    },
}

# ratio = T_baseline / T_candidate; > 1 means the candidate is faster.
COMPARISONS = (
    # comparison_id, family, stage, block_type, baseline, candidate, crossover
    ("Ba_q1", "B-a materialization", "balanced", "core", "BNEW_Q1_G2", "D_Q1_G2", 1),
    ("Ba_q4", "B-a materialization", "balanced", "core", "BNEW_Q4_G2", "D_Q4_G2", 1),
    ("Ba_bridge_g1", "B-a materialization (weak transport bridge)", "balanced", "core", "BOLD_Q1_G1", "D_Q1_G1", 1),
    ("Bb_q1", "B-b selection", "balanced", "core", "SLT", "D_Q1_G2", 1),
    ("Bb_q4", "B-b selection", "balanced", "core", "SLT", "D_Q4_G2", 1),
    ("splitk_J1", "split-K at J1", "balanced", "core", "D_Q4_G2", "D_Q1_G2", 0),
    ("hist_cutlass_vs_q4", "bridge to earlier claim", "balanced", "core", "HM_CUTLASS", "D_Q4_G2", 0),
    ("hist_lt_vs_q4", "bridge to earlier claim", "balanced", "core", "HM_LT", "D_Q4_G2", 0),
    ("baseline_upgrade_cutlass", "baseline upgrade", "balanced", "core", "HM_CUTLASS", "SLT", 0),
    ("baseline_upgrade_lt", "baseline upgrade", "balanced", "core", "HM_LT", "SLT", 0),
    ("transport_a2a_vs_lt", "staged transport and GEMM", "balanced", "core", "SLT", "BNEW_Q1_G2", 0),
    ("j8_vs_j1", "J8 ablation (q1/g2 only)", "balanced", "ablation_j8", "D_Q1_G2", "J8_Q1_G2", 0),
    ("Bb_j8", "B-b selection, J8 ablation", "balanced", "ablation_j8", "SLT", "J8_Q1_G2", 1),
    ("flux_vs_slt", "public fused baseline", "balanced", "flux", "SLT", "FLUX_V2", 0),
    ("R_Ba_q1", "B-a materialization", "snapshot", "core", "BNEW_Q1_G2_R", "D_Q1_G2_R", 0),
    ("R_Ba_q4", "B-a materialization", "snapshot", "core", "BNEW_Q4_G2_R", "D_Q4_G2_R", 0),
    ("R_Bb_q1", "B-b selection", "snapshot", "core", "NATIVE", "D_Q1_G2_R", 0),
    ("R_Bb_q4", "B-b selection", "snapshot", "core", "NATIVE", "D_Q4_G2_R", 0),
    ("R_native_vs_bnew", "staged transport", "snapshot", "core", "NATIVE", "BNEW_Q1_G2_R", 0),
    ("R_splitk_J1", "split-K at J1", "snapshot", "core", "D_Q4_G2_R", "D_Q1_G2_R", 0),
    ("R_flux_vs_native", "public fused baseline", "snapshot", "flux", "NATIVE", "FLUX_V2_R", 0),
    ("R_j8_vs_j1", "J8 ablation (q1/g2 only)", "snapshot", "ablation_j8", "D_Q1_G2_R", "J8_Q1_G2_R", 0),
    ("R_Bb_j8", "B-b selection, J8 ablation", "snapshot", "ablation_j8", "NATIVE", "J8_Q1_G2_R", 0),
)

POLICIES = {
    "tie_band_upper": 1.03,
    "pairs_required": 3,
    "r2a_mean_max": 0.0231,
    "r2a_max_max": 0.0538,
    "balanced": {
        "selection_set": ["D_Q1_G2", "D_Q4_G2", "SLT"],
        "staged_representative_priority": ["SLT", "BNEW_Q1_G2", "HM_LT", "HM_CUTLASS"],
        "policies": {
            "P0": "sealed_prediction",
            "ALWAYS_STAGED": "staged_representative",
            "FIXED_DIRECT_Q4": "D_Q4_G2",
            "FIXED_DIRECT_Q1": "D_Q1_G2",
        },
        "r2b_competitors": ["ALWAYS_STAGED", "FIXED_DIRECT_Q4"],
    },
    "snapshot": {
        "selection_set": ["D_Q1_G2_R", "D_Q4_G2_R", "BNEW_Q1_G2_R", "NATIVE"],
        "staged_representative_priority": ["NATIVE", "BNEW_Q1_G2_R"],
        "policies": {
            "P0": "sealed_prediction",
            "ALWAYS_STAGED": "staged_representative",
            "ALWAYS_BNEW": "BNEW_Q1_G2_R",
            "FIXED_DIRECT_Q4": "D_Q4_G2_R",
        },
        "r2b_competitors": ["ALWAYS_STAGED", "FIXED_DIRECT_Q4"],
    },
}


def fixed_order(items, *key):
    """Stable keyed permutation; independent of Python's RNG and hash seed."""
    def sort_key(item):
        fields = (DESIGN_ID, *key, item)
        return hashlib.sha256("|".join(map(str, fields)).encode("utf-8")).digest()
    return sorted(items, key=sort_key)


def ceil_div(a, b):
    return -(-a // b)


def cell_rows():
    rows = []
    for width_id, model, revision, n in WIDTHS:
        for m in M_BINS:
            rows.append(dict(
                cell_id=f"{width_id}_M{m:04d}", width_id=width_id, model=model, model_revision=revision,
                ranks=RANKS, m_per_source_destination=m, k=K, n=n,
                replay_g1=ceil_div(n, 128), replay_g2=ceil_div(n, 256),
                cohort=HISTORICAL.get((width_id, m), "unseen"),
            ))
    return rows


def snapshot_rows():
    rows = []
    for width_id, experts, topk, layers, first_moe, layer in MODELS:
        model = next(w for w in WIDTHS if w[0] == width_id)
        assert first_moe <= layer < layers
        for phase_id, category, ordinal, text in PHASES:
            rows.append(dict(
                snapshot_id=f"{width_id}_{phase_id}", width_id=width_id, model=model[1], model_revision=model[2],
                request_set="S32: 32 prompts x 512 input tokens, 128 output tokens, ignore_eos, temperature 0",
                parallel="TP1/DP8/EP8, static placement expert // (E/8), prefix cache off",
                category=category, step_ordinal=ordinal, step_rule=text, layer_index=layer,
                routed_experts=experts, top_k=topk, k=K, n=model[3],
            ))
    return rows


def arm_rows():
    keys = ("arm_id", "stage", "kind", "implementation", "sources_per_launch", "split_k",
            "n_tile_multiplier", "gpu_status", "block_type")
    return [dict(zip(keys, arm)) for arm in ARMS]


def comparison_rows():
    keys = ("comparison_id", "family", "stage", "block_type", "baseline_arm", "candidate_arm", "crossover")
    return [dict(zip(keys, c)) for c in COMPARISONS]


def block_rows():
    rows, index = [], 0
    plans = (
        ("balanced", [c["cell_id"] for c in cell_rows()], SEEDS),
        ("snapshot", [s["snapshot_id"] for s in snapshot_rows()], REPS),
    )
    for stage, units, reps in plans:
        for block_type, arms in BLOCK_TYPES[stage].items():
            for round_no, rep in enumerate(reps, start=1):
                for unit in fixed_order(units, "unit_order", stage, block_type, rep):
                    forward = fixed_order(arms, "arm_order", stage, block_type, unit, rep)
                    pair_id = f"{stage}:{block_type}:{unit}:{rep}"
                    for orientation, order in (("forward", forward), ("reverse", forward[::-1])):
                        index += 1
                        pid = f"{'B' if stage == 'balanced' else 'S'}{index:04d}"
                        for position, arm in enumerate(order, start=1):
                            rows.append(dict(
                                process_id=pid, stage=stage, block_type=block_type, round=round_no,
                                unit_id=unit, rep=rep, orientation=orientation, pair_id=pair_id,
                                arm_position=position, arm_id=arm,
                            ))
    return rows


def queue_rows(blocks):
    """One row per job. The order is the priority: when the lease runs out, drop jobs from the end."""
    processes = {}
    for b in blocks:
        processes.setdefault(b["process_id"], b)
    by_group = defaultdict(list)
    for pid, b in processes.items():
        by_group[(b["stage"], b["block_type"], b["round"])].append(pid)

    jobs = []

    def add(job_type, condition="always", process_id="", unit_id="", note=""):
        jobs.append(dict(queue_position=len(jobs) + 1, job_type=job_type, condition=condition,
                         process_id=process_id, unit_id=unit_id, note=note))

    def add_group(stage, block_type, round_no, condition):
        for pid in by_group[(stage, block_type, round_no)]:
            add("timing_block", condition, pid, processes[pid]["unit_id"],
                f"{stage}/{block_type}/round{round_no}")

    add("hw_gate", note="scripts/hw_gate.py; FAIL stops formal runs")
    add("build_identity", note="record hashes of all binaries, sources, inputs, weights")
    add("admission_smoke", note="every arm on GLM_M0001 and GLM_M1024 (balanced); one targeted fix per failing arm")
    add("freeze", note="scripts/manifest.py create over sources and binaries; no code changes after this")
    add("admission_full", note="every balanced arm x cell x seed numerics; writes admission.jsonl")
    add("microbench", note="F_direct, G_launch, F_staged, b_in, a2a, allgather, hbm, mma, lt_launch, sem_per_slice")
    add("calibrate_predict_seal", note="calibrate.py -> predict.py --cells -> seal and copy off-machine before any timing block")
    for r in (1, 2, 3):
        add_group("balanced", "core", r, "always")
    for width_id, *_ in MODELS:
        add("capture", unit_id=width_id, note="S32 capture: routes, activations, expert weights of the capture layer")
    add("route_stats_predict_seal", note="route_stats.py -> predict.py --snapshots -> seal and copy off-machine")
    add("admission_snapshot", note="every snapshot arm x snapshot numerics vs FP32 reference")
    for r in (1, 2, 3):
        add_group("snapshot", "core", r, "always")
    for width_id, *_ in MODELS:
        add("native_attribution", unit_id=width_id, note="3 instrumented + 3 uninstrumented S32 runs, alternating")
    for r in (1, 2, 3):
        add_group("balanced", "flux", r, "if_admitted:FLUX_V2")
        add_group("snapshot", "flux", r, "if_admitted:FLUX_V2_R")
    for r in (1, 2, 3):
        add_group("balanced", "ablation_j8", r, "if_admitted:J8_Q1_G2")
        add_group("snapshot", "ablation_j8", r, "if_admitted:J8_Q1_G2_R")
    add("udec_smoke", "if_exists:U-dec", note="real inputs, cross-process visibility, all routed layers, no fallback, output check")
    add("udec_ab", "if_passed:udec_smoke", note="paired whole-model decode A/B, 3 forward/reverse pairs")
    add("final_analysis", note="analyze.py, amdahl.py, manifest.py verify off-machine")
    return jobs


def validate(cells, snapshots, arms, comparisons, blocks, jobs):
    assert len(cells) == 33 and len({c["cell_id"] for c in cells}) == 33
    assert Counter(c["cohort"] for c in cells) == {"unseen": 31, "historical_overlap": 1, "historical_calibration": 1}
    assert len(snapshots) == 6
    arm_ids = {a["arm_id"]: a for a in arms}
    for stage, types in BLOCK_TYPES.items():
        for block_type, members in types.items():
            for arm in members:
                assert arm_ids[arm]["stage"] == stage, arm
    for c in comparisons:
        members = BLOCK_TYPES[c["stage"]][c["block_type"]]
        assert c["baseline_arm"] in members and c["candidate_arm"] in members, c
        if c["crossover"]:
            kinds = {arm_ids[c["baseline_arm"]]["kind"], arm_ids[c["candidate_arm"]]["kind"]}
            assert kinds == {"direct", "staged"}, c
    for stage, cfg in POLICIES.items():
        if stage in BLOCK_TYPES:
            assert set(cfg["selection_set"]) <= set(BLOCK_TYPES[stage]["core"])
    pairs = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        pairs[b["pair_id"]][b["orientation"]].append(b["arm_id"])
    expected_pairs = sum(len(t) * (33 if s == "balanced" else 6) * 3 for s, t in BLOCK_TYPES.items())
    assert len(pairs) == expected_pairs
    for pair in pairs.values():
        assert pair["forward"] == pair["reverse"][::-1] and len(set(pair["forward"])) == len(pair["forward"])
    queued = [j["process_id"] for j in jobs if j["job_type"] == "timing_block"]
    assert sorted(queued) == sorted({b["process_id"] for b in blocks}) and len(queued) == len(set(queued))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "design")
    args = parser.parse_args()
    cells, snapshots, arms = cell_rows(), snapshot_rows(), arm_rows()
    comparisons, blocks = comparison_rows(), block_rows()
    jobs = queue_rows(blocks)
    validate(cells, snapshots, arms, comparisons, blocks, jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cells", cells), ("snapshots", snapshots), ("arms", arms),
                       ("comparisons", comparisons), ("blocks", blocks), ("queue", jobs)):
        write_csv(args.out / f"{name}.csv", rows)
    (args.out / "policies.json").write_text(json.dumps(POLICIES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_proc = len({b["process_id"] for b in blocks})
    print(f"{len(cells)} cells, {len(snapshots)} snapshots, {len(arms)} arms, {len(comparisons)} comparisons, "
          f"{n_proc} timing processes, {len(blocks)} candidate timings, {len(jobs)} queue jobs")


if __name__ == "__main__":
    main()
