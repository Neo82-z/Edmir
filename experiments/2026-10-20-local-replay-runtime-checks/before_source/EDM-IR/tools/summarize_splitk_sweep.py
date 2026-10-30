#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {path}")
    return rows


def by_k(rows: list[dict[str, str]]) -> dict[int, dict[str, dict[str, str]]]:
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        k = int(row["k"])
        family = row["family"]
        if family in grouped[k]:
            raise ValueError(f"duplicate family {family} for K={k}")
        grouped[k][family] = row
    return dict(grouped)


def winner(rows: dict[str, dict[str, str]]) -> str:
    return min(rows, key=lambda family: float(rows[family]["median_ms"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a split-K rule on one sweep and evaluate held-out regret"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train = by_k(read_rows(args.train))
    frozen_rule = {k: winner(rows) for k, rows in sorted(train.items())}
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    decisions: list[dict[str, str | int | float]] = []
    for path in args.heldout:
        heldout = by_k(read_rows(path))
        for k, selected_family in frozen_rule.items():
            if k not in heldout:
                raise ValueError(f"held-out {path} has no K={k}")
            rows = heldout[k]
            missing = set(train[k]) - set(rows)
            if missing:
                raise ValueError(f"held-out {path} K={k} misses {sorted(missing)}")
            oracle_family = winner(rows)
            selected_ms = float(rows[selected_family]["median_ms"])
            oracle_ms = float(rows[oracle_family]["median_ms"])
            host_ms = float(rows["host"]["median_ms"])
            regret_ms = selected_ms - oracle_ms
            decisions.append(
                {
                    "rows_per_peer": int(rows[selected_family]["rows_per_peer"]),
                    "n": int(rows[selected_family]["n"]),
                    "k": k,
                    "selected_family": selected_family,
                    "oracle_family": oracle_family,
                    "selected_ms": selected_ms,
                    "oracle_ms": oracle_ms,
                    "host_ms": host_ms,
                    "regret_ms": regret_ms,
                    "regret_pct": 100.0 * regret_ms / oracle_ms,
                    "speedup_vs_host": host_ms / selected_ms,
                    "exact_family": int(selected_family == oracle_family),
                    "selected_beats_host": int(selected_ms < host_ms),
                }
            )

    fields = tuple(decisions[0]) if decisions else ()
    if decisions:
        with (output / "heldout_decisions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(decisions)

    count = len(decisions)
    summary = {
        "train": str(args.train.resolve()),
        "heldout": [str(path.resolve()) for path in args.heldout],
        "frozen_rule": {str(k): family for k, family in frozen_rule.items()},
        "heldout_problems": count,
        "exact_family_accuracy": (
            sum(int(row["exact_family"]) for row in decisions) / count if count else None
        ),
        "mean_regret_pct": (
            sum(float(row["regret_pct"]) for row in decisions) / count if count else None
        ),
        "max_regret_pct": (
            max(float(row["regret_pct"]) for row in decisions) if count else None
        ),
        "mean_speedup_vs_host": (
            sum(float(row["speedup_vs_host"]) for row in decisions) / count
            if count
            else None
        ),
        "selected_beats_host_count": sum(
            int(row["selected_beats_host"]) for row in decisions
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
