#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    ScheduleMeasurement,
    TwoStageSelector,
    evaluate_group_heldout,
    parse_schedule_family,
    read_schedule_measurements,
)


def sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def candidates_for_shape(
    *,
    group_id: str,
    world_size: int,
    rows_per_peer: int,
    k: int,
    n: int,
    families: list[str],
) -> dict[str, ScheduleMeasurement]:
    unique_remote = world_size * (world_size - 1) * rows_per_peer * k * 2
    n_tiles = math.ceil(n / 128)
    candidates = {}
    for family in families:
        q, g = parse_schedule_family(family)
        rho = 1 if family == "host" else math.ceil(n_tiles / g)
        ctas = 0 if family == "host" else math.ceil(rows_per_peer / 64) * rho * q
        candidates[family] = ScheduleMeasurement(
            group_id=group_id,
            family=family,
            world_size=world_size,
            rows_per_peer=rows_per_peer,
            k=k,
            n=n,
            measured_ms=1.0,
            remote_input_bytes=unique_remote,
            remote_a_replay=rho,
            split_k_slices=q,
            n_tiles_per_cta=g,
            estimated_ctas_per_op=ctas,
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze zero-shot eight-GPU predictions before observing their latency"
    )
    parser.add_argument("training_results", type=Path, nargs="+")
    parser.add_argument("--training-world-size", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-fused-advantage", type=float, default=0.0)
    args = parser.parse_args()

    measurements = read_schedule_measurements(
        tuple(args.training_results), world_size=args.training_world_size
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "edmir.frozen-fused-experiment.v1":
        raise ValueError("unsupported transfer manifest")
    selector = TwoStageSelector.fit(
        measurements,
        ridge=args.ridge,
        sm_count=int(manifest["hardware"]["sm_count_per_gpu"]),
        minimum_fused_advantage=args.minimum_fused_advantage,
    )
    base = list(manifest["candidate_space"]["base"])
    grouped = [
        f"{family}_groupedn2"
        for family in manifest["candidate_space"]["grouped_n_2"]
    ]
    families = [*base, *grouped]
    predictions = []
    for sweep in manifest["sweeps"]:
        for k in sweep["k_values"]:
            candidates = candidates_for_shape(
                group_id=sweep["id"],
                world_size=int(manifest["hardware"]["world_size"]),
                rows_per_peer=int(sweep["rows_per_peer"]),
                k=int(k),
                n=int(sweep["n"]),
                families=families,
            )
            selected, decision = selector.select(candidates)
            family_predictions = {
                "host": selector.host_model.predict_ms(candidates["host"]),
                **{
                    family: selector.fused_model.predict_ms(row)
                    for family, row in candidates.items()
                    if family != "host"
                },
            }
            predictions.append(
                {
                    "sweep_id": sweep["id"],
                    "role": sweep["role"],
                    "regime": sweep["regime"],
                    "world_size": manifest["hardware"]["world_size"],
                    "rows_per_peer": sweep["rows_per_peer"],
                    "k": k,
                    "n": sweep["n"],
                    "selected_family": selected.family,
                    "decision": decision,
                    "predicted_ms_by_family": family_predictions,
                }
            )
    heldout = evaluate_group_heldout(
        measurements,
        ridge=args.ridge,
        minimum_fused_advantage=args.minimum_fused_advantage,
    )
    report = {
        "schema": "edmir.frozen-transfer-predictions.v1",
        "status": "frozen-before-eight-gpu-measurement",
        "experiment_id": manifest["experiment_id"],
        "training_world_size": args.training_world_size,
        "target_world_size": manifest["hardware"]["world_size"],
        "training_results_sha256": sha256(list(args.training_results)),
        "manifest_sha256": sha256([args.manifest]),
        "ridge": args.ridge,
        "minimum_fused_advantage": args.minimum_fused_advantage,
        "four_gpu_group_heldout_metrics": heldout["metrics"],
        "eligible_for_default_admission": heldout["metrics"]["admitted"],
        "model": selector.to_dict(),
        "predictions": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote={args.out} predictions={len(predictions)} "
        f"eligible={str(report['eligible_for_default_admission']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
