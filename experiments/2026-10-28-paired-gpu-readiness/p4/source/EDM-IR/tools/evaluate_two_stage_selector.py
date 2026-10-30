#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    DEFAULT_SM_COUNT,
    SelectionGates,
    evaluate_two_stage_ablations,
    read_schedule_measurements,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate host admission and fused (g,q) selection with declared "
            "sweep groups held out"
        )
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sm-count", type=int, default=DEFAULT_SM_COUNT)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-fused-advantage", type=float, default=0.0)
    parser.add_argument("--minimum-problems", type=int, default=30)
    parser.add_argument("--minimum-direction-accuracy", type=float, default=0.99)
    parser.add_argument("--maximum-mean-regret", type=float, default=0.005)
    parser.add_argument("--maximum-worst-regret", type=float, default=0.01)
    parser.add_argument("--maximum-host-regressions", type=int, default=0)
    parser.add_argument(
        "--include-fold-models",
        action="store_true",
        help="retain every fold coefficient vector in addition to decisions",
    )
    parser.add_argument(
        "--include-ablation-decisions",
        action="store_true",
        help="retain per-shape decisions for every ablation, not only the primary model",
    )
    args = parser.parse_args()

    measurements = read_schedule_measurements(
        tuple(args.results), world_size=args.world_size
    )
    gates = SelectionGates(
        minimum_problems=args.minimum_problems,
        minimum_direction_accuracy=args.minimum_direction_accuracy,
        maximum_mean_regret=args.maximum_mean_regret,
        maximum_worst_regret=args.maximum_worst_regret,
        maximum_host_regressions=args.maximum_host_regressions,
    )
    report = evaluate_two_stage_ablations(
        measurements,
        ridge=args.ridge,
        sm_count=args.sm_count,
        minimum_fused_advantage=args.minimum_fused_advantage,
        gates=gates,
    )
    report["inputs"] = {
        "world_size": args.world_size,
        "results": [str(path) for path in args.results],
        "measurement_rows": len(measurements),
    }
    if not args.include_fold_models:
        for item in report["reports"].values():
            item.pop("fold_models", None)
    if not args.include_ablation_decisions:
        for label, item in report["reports"].items():
            if label != report["primary"]:
                item.pop("decisions", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for label, item in report["reports"].items():
        metrics = item["metrics"]
        print(
            f"{label:>12} admitted={str(metrics['admitted']).lower():5} "
            f"direction={metrics['direction_accuracy']:.3f} "
            f"mean_regret={metrics['mean_selection_regret']:.4f} "
            f"max_regret={metrics['max_selection_regret']:.4f} "
            f"host_regressions={metrics['host_regressions']} "
            f"coverage={metrics['fused_coverage']:.3f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
