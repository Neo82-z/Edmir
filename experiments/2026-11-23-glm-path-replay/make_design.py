#!/usr/bin/env python3
"""Derive the smaller GLM design without rewriting the archived tables."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "2026-11-20-final-round" / "design"
DESIGN_ID = "glm-path-replay-2026-11-23.v2"
BRIDGE = ("D_Q1_G1", "BOLD_Q1_G1")
BALANCED = ("D_Q1_G2", "D_Q4_G2", "BNEW_Q1_G2", "BNEW_Q4_G2", "SLT")+BRIDGE
SNAPSHOT = ("D_Q1_G2_R", "D_Q4_G2_R", "BNEW_Q1_G2_R", "BNEW_Q4_G2_R", "NATIVE")
ARMS = {"balanced": BALANCED, "snapshot": SNAPSHOT}
# Consumer launches replay as one CUDA graph per arm; pack, AllToAll and barriers run eagerly.
GRAPH = "; consumer launches replayed as one CUDA graph, communication eager"
PRIMARY = ("Ba_q1", "Ba_q4", "Bb_q1", "Bb_q4", "R_Ba_q1", "R_Ba_q4", "R_Bb_q1", "R_Bb_q4", "Ba_bridge_g1")


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    cells = [r for r in read(OLD / "cells.csv") if r["width_id"] == "GLM"]
    snapshots = [r for r in read(OLD / "snapshots.csv") if r["width_id"] == "GLM"]
    units = {r["cell_id"] for r in cells} | {r["snapshot_id"] for r in snapshots}
    arms = [r for r in read(OLD / "arms.csv") if r["arm_id"] in ARMS[r["stage"]]]
    for arm in arms:
        if arm["stage"] == "snapshot":
            if arm["kind"] == "direct":
                arm["implementation"] = "ragged per-source expert Triton BF16 on symmetric peer memory; FP32 split merge"+GRAPH
            elif arm["arm_id"] == "NATIVE":
                arm["implementation"] = "pinned vLLM all-gatherv and captured Triton gate/up config on fixed routes"
            else:
                arm["implementation"] = "NCCL ragged AllToAll plus same Triton BF16 consumer as direct; FP32 split merge"+GRAPH
        elif arm["arm_id"] in {"D_Q1_G2", "D_Q4_G2", "D_Q1_G1"}:
            arm["implementation"] = "per-source Triton SM80 FP16 on symmetric peer memory; FP32 split merge"+GRAPH
            # gpu_status refers to the arm's compute kernel; this Triton consumer has not run on an A100.
            arm["gpu_status"] = "new"
        elif arm["arm_id"].startswith("BNEW_"):
            arm["implementation"] = "NCCL AllToAll plus identical per-source Triton consumer"+GRAPH
        elif arm["arm_id"] == "BOLD_Q1_G1":
            arm["implementation"] = "cyclic NCCL pairwise stage plus existing CUTLASS 64x128"
    comparisons = [r for r in read(OLD / "comparisons.csv") if r["comparison_id"] in PRIMARY]
    for c in comparisons:
        if c["comparison_id"] == "Ba_bridge_g1":
            c["crossover"] = "0"
    blocks = defaultdict(list)
    for row in read(OLD / "blocks.csv"):
        if row["block_type"] == "core" and row["unit_id"] in units and row["arm_id"] in ARMS[row["stage"]]:
            blocks[row["process_id"]].append(row)
    for rows in blocks.values():
        for i, row in enumerate(rows, 1):
            row["arm_position"] = i
    ordered = sorted(blocks, key=lambda p: (blocks[p][0]["stage"] != "snapshot", p))
    queue = []

    def add(job, unit="", process="", detail=""):
        queue.append(dict(position=len(queue)+1, job=job, unit_id=unit, process_id=process, detail=detail))

    add("hw_gate", detail="8x A100 SXM4 80GB, NV12, peer R/W, enforced power, load")
    add("build_balanced", detail="SM80 CUTLASS/CUDA and tuned cuBLASLt shared library")
    add("device_smoke", detail="synthetic 5 BF16 replay arms and 7 FP16 grid arms at M1/M64 before model capture")
    add("tune_native_glm", detail="one bounded E8 N1536 tuning job; record tuned or default")
    add("microbench", detail="1- and 7-peer reads by CTA count, HBM copy, launch, barrier, AllToAll per M")
    add("tune_slt", detail="up to 8 cuBLASLt heuristics per cell, 32 MiB; lock per-rank choices")
    add("capture_glm", detail="layer 23, pure prefill #1 and pure decode #8; save native launch tensors")
    add("prepare_both", detail="one RUN: 33 FP16 seed identities, 2 capture hashes, all builds and fixed baselines")
    add("admit_snapshot", detail="per-arm isolated workers, FP32 reference, finite outputs and source fingerprint")
    add("admit_balanced", detail="three seeds per cell; g1 bridge only if a balanced BNEW arm fails admission")
    add("lock_admission", detail="freeze bridge decision and all statuses before formal timing")
    for pid in ordered:
        row = blocks[pid][0]
        add("timing_block", row["unit_id"], pid, row["pair_id"])
    add("analyze_and_export", detail="Preserve paired coverage and verified figure/table input")

    tables = dict(cells=cells, snapshots=snapshots, arms=arms, comparisons=comparisons,
                  blocks=[r for p in ordered for r in blocks[p]], queue=queue)
    assert len(cells) == 11 and len(snapshots) == 2 and len(blocks) == 78
    assert len(comparisons) == len(PRIMARY)
    assert all(len(b)==len(ARMS[b[0]["stage"]]) for b in blocks.values())
    for name, rows in tables.items():
        write(out / f"{name}.csv", rows)
    policies = dict(
        design_id=DESIGN_ID, tie_band_upper=1.03, pairs_required=3, warmup=20, timed=100,
        balanced=dict(selection_set=["D_Q1_G2", "D_Q4_G2", "SLT"],
                      policies=dict(P0="sealed_prediction", ALWAYS_SLT="SLT",
                                    FIXED_DIRECT_Q1="D_Q1_G2", FIXED_DIRECT_Q4="D_Q4_G2")),
        snapshot=dict(selection_set=list(SNAPSHOT),
                      policies=dict(P0="sealed_prediction", ALWAYS_STAGED_Q1="BNEW_Q1_G2_R",
                                    ALWAYS_STAGED_Q4="BNEW_Q4_G2_R", ALWAYS_NATIVE="NATIVE",
                                    FIXED_DIRECT_Q1="D_Q1_G2_R", FIXED_DIRECT_Q4="D_Q4_G2_R")),
    )
    (out / "policies.json").write_text(json.dumps(policies, indent=2)+"\n")
    files = sorted([f"{name}.csv" for name in tables] + ["policies.json"])
    (out / "DESIGN.json").write_text(json.dumps({"design_id": DESIGN_ID,
        "files": {name: sha(out / name) for name in files}}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "design")
    generate(parser.parse_args().out)
