#!/usr/bin/env python3
"""Audit the archived cost basis without fitting, timing, or changing selection.

Wave counts below assume one CTA per SM and are geometry diagnostics only.
They are not measured occupancy, service concurrency, or a latency model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EDM-IR"))
from edmir.compiler.measurement_dataset import load_selector_dataset
from edmir.compiler.scale_aware_prediction import basis, geometry

SM_COUNT = 108
DATA = ROOT / "experiments/2026-10-20-offline-predictor/data"


def describe(shape, family):
    x = geometry(shape, family)
    if family == "host":
        raise ValueError("This audit concerns fused consumer geometry only")
    p, m, k, n = shape
    ctas = x["ctas_per_gemm"]
    output_tiles = x["m_tiles"] * x["n_tiles"]
    return {
        "P": p, "M": m, "K": k, "N": n, "family": family,
        "q": x["q"], "g": x["g"], "ctas_per_launch": ctas,
        "output_tiles": output_tiles,
        "valid_output_fraction": m * n / (output_tiles * 64 * 128 * x["g"]),
        "K_iterations_per_partition": x["K_iterations_per_partition"],
        "one_cta_per_sm_rounds": (ctas + SM_COUNT - 1) // SM_COUNT,
        "reference_round_tail_ctas": (ctas - 1) % SM_COUNT + 1,
        "grid_underfill_proxy": max(1 - ctas / SM_COUNT, 0),
    }


def run():
    p4, calibration, evaluation, _ = load_selector_dataset(DATA)
    rows = (*p4, *calibration, *evaluation)
    if len(rows) > 1024:
        raise ValueError("Audit bound exceeded; review the expanded dataset first")
    details, by_basis = [], defaultdict(list)
    for row in rows:
        if row.family == "host":
            continue
        d = describe(row.shape, row.family)
        details.append(d)
        key = (row.world_size, row.family, basis(row.shape, row.family)[1])
        by_basis[key].append(d)
    collisions = [group for group in by_basis.values() if len({
        (d["ctas_per_launch"], d["valid_output_fraction"]) for d in group
    }) > 1]

    # Constructed shapes are algebraic checks, not measurements or test labels.
    examples = []
    for a, b, family in (
        ((8, 32, 8192, 8192), (8, 64, 8192, 4096), "splitk8_groupedn2"),
        ((8, 32, 4096, 8192), (8, 64, 4096, 4096), "splitk8"),
    ):
        same = basis(a, family) == basis(b, family)
        if not same:
            raise ValueError("The cost basis changed; reassess the documented collision")
        examples.append({"a": describe(a, family), "b": describe(b, family),
                         "identical_complete_basis": same,
                         "measurement_labels_used": False})
    sources = [DATA / n for n in ("manifest.json", "selector_measurements.jsonl",
                                  "component_measurements.jsonl")]
    sources += [ROOT / "EDM-IR/edmir/compiler/scale_aware_prediction.py", Path(__file__)]
    result = {
        "schema": "edmir.prediction-geometry-audit.v1",
        "date": "2026-10-26", "input_rows": len(rows), "fused_rows": len(details),
        "grid_at_least_108": sum(d["ctas_per_launch"] >= SM_COUNT for d in details),
        "counts_by_P": {str(p): dict(Counter(
            "at_least_108" if d["ctas_per_launch"] >= SM_COUNT else "below_108"
            for d in details if d["P"] == p)) for p in (4, 8)},
        "historical_geometry_alias_groups": len(collisions),
        "historical_geometry_aliases": collisions,
        "constructed_equal_basis_examples": examples,
        "input_and_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sources},
        "model_fits": 0, "GPU_runs": 0, "latency_labels_used_in_analysis": False,
        "interpretation": [
            "Equal basis forces equal predictions for all coefficient choices within one fitted model.",
            "Different geometry does not prove different latency or explain existing M512 errors.",
            "One-CTA-per-SM rounds are a reference convention, not achieved concurrency.",
            "The archived full-program labels cannot identify isolated tile or semaphore times.",
        ],
    }
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; preserve prior evidence")
    start = time.monotonic()
    result, details = run()
    if time.monotonic() - start > 10:
        raise RuntimeError("Local audit exceeded its expected 10-second envelope")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.out / "geometry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    print(json.dumps({"output": str(args.out), "fused_rows": len(details),
                      "alias_groups": result["historical_geometry_alias_groups"],
                      "elapsed_seconds": round(time.monotonic() - start, 3)}))


if __name__ == "__main__":
    main()
