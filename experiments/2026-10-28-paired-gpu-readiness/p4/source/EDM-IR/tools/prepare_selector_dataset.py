#!/usr/bin/env python3
"""Normalize existing measurements with provenance. No GPU or network access."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import (  # noqa: E402
    GEMM_ENV, P4_ENV, P8_ENV, load_selector_dataset, sha256,
)
from edmir.compiler.relative_selection import strict_surfaces  # noqa: E402
from edmir.compiler.two_stage_selection import read_schedule_measurements  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def positive_ms(value, *, unit):
    result = float(value) / {"ms": 1.0, "us": 1000.0}[unit]
    if not math.isfinite(result) or result <= 0:
        raise ValueError("missing, non-finite or non-positive latency")
    return result


def nccl_records(path, *, environment_id, world_size, source):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 13 or not fields[0].isdigit():
            continue
        size, count = int(fields[0]), int(fields[1])
        if fields[2:5] != ["float", "none", "-1"] or size != count * world_size * 4:
            raise ValueError("unexpected NCCL all-to-all layout/dtype")
        for mode, index in (("out_of_place", 5), ("in_place", 9)):
            rows.append({
                "id": f"{environment_id}:nccl-alltoall:{size}:{mode}",
                "environment_id": environment_id, "scope": "standalone_collective",
                "provider": "nccl-tests", "operation": "alltoall", "dtype": "float32",
                "world_size": world_size, "mode": mode,
                "size_bytes_per_rank_including_self": size,
                "elements_per_peer": count,
                "latency_ms": positive_ms(fields[index], unit="us"),
                "latency_statistic": "nccl-tests reported time",
                "algbw_GB_per_s": float(fields[index + 1]),
                "busbw_GB_per_s": float(fields[index + 2]),
                "wrong": None if fields[index + 3] == "N/A" else int(fields[index + 3]),
                "source": {**source, "line": line_number},
                "eligible_for_fused_training": False,
            })
    if not rows:
        raise ValueError("no NCCL measurement rows")
    return rows


def build_dataset(repo: Path, out: Path):
    if out.exists():
        raise ValueError("output exists; use a new directory")
    inputs = {}

    def record(path):
        path = path.resolve()
        name = str(path.relative_to(repo))
        inputs[name] = {"path": name, "sha256": sha256(path)}
        return inputs[name]

    # Pin the old file selection before applying any new normalization.
    audit_path = repo / "experiments/2026-10-19-selector-baseline-design/interaction_results.json"
    audit = json.loads(audit_path.read_text())
    record(audit_path)
    for expected in audit["inputs"]:
        actual = record(repo / expected["path"])
        if actual["sha256"] != expected["sha256"]:
            raise ValueError(f"historical input changed: {expected['path']}")
    p4_path = repo / "experiments/2026-08-29-a100-grouped-n-plan/two_stage_selector_report.json"
    p4_report = json.loads(p4_path.read_text())
    rows, normalized = [], []

    def add(path, p, env, role, group=None):
        source = record(path)
        with path.open(newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        measured_rows = read_schedule_measurements([path], world_size=p)
        for number, (row, raw) in enumerate(zip(measured_rows, raw_rows), 2):
            if group is not None:
                row = replace(row, group_id=group)
            rows.append(row)
            normalized.append({
                "id": f"{env}:{row.rows_per_peer}:{row.k}:{row.n}:{row.family}",
                "environment_id": env, "historical_role": role,
                "evidence_status": "retrospective_development",
                "scope": "full_operator", "latency_unit": "ms", "dtype": "float16",
                "timing_scope": "host steady_clock around backend.execute() plus backend.synchronize()",
                "kernel": raw["gemm_kernel"],
                "reported_timing": {
                    **{key: positive_ms(raw[key], unit="ms") for key in
                       ("min_ms", "median_ms", "p95_ms", "max_ms", "mean_ms")},
                    **{key: int(raw[key]) for key in ("warmup", "iterations")},
                },
                "m_semantics": "rows per source-destination pair",
                "rows_per_destination": p * row.rows_per_peer,
                "measurement": asdict(row), "source": {**source, "line": number},
                "repeat_observation": None, "scoring_override": None,
            })

    for name in p4_report["inputs"]["results"]:
        add(repo / name, 4, P4_ENV, "development")
    p8_root = repo / "experiments/2026-08-30-a100-8gpu/extracted"
    manifest_path = p8_root / "experiments/2026-08-29-a100-8gpu-transfer-plan/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_families = set(manifest["candidate_space"]["base"]) | {
        name + "_groupedn2" for name in manifest["candidate_space"]["grouped_n_2"]}
    for sweep in manifest["sweeps"]:
        path = p8_root / "results/p8-timing" / sweep["id"] / "repeat-01/results.csv"
        if not path.exists():
            path = p8_root / "results/p8-frozen" / sweep["id"] / "repeat-01/results.csv"
        add(path, 8, P8_ENV, sweep["role"], sweep["id"])
    groups = strict_surfaces(rows)
    by_key = {(r["environment_id"], r["measurement"]["rows_per_peer"],
               r["measurement"]["k"], r["measurement"]["n"],
               r["measurement"]["family"]): r for r in normalized}
    repeat_path = p8_root / "results/p8-analysis/boundary-repeats/replication_results.csv"
    repeat_source = record(repeat_path)
    with repeat_path.open(newline="") as stream:
        for number, raw in enumerate(csv.DictReader(stream), 2):
            key = (P8_ENV, int(raw["rows_per_peer"]), int(raw["k"]), int(raw["n"]), raw["family"])
            row = by_key[key]
            if (row["repeat_observation"] is not None
                    or row["measurement"]["group_id"] != raw["sweep_id"]
                    or not math.isclose(row["measurement"]["measured_ms"],
                                        float(raw["initial_ms"]), rel_tol=1e-10)):
                raise ValueError("repeat does not uniquely match an evaluation candidate")
            row["repeat_observation"] = {
                "median_ms": positive_ms(raw["median_ms"], unit="ms"),
                "source": {**repeat_source, "line": number},
                "usage": ("scoring override; not an independent additional sample"
                          if row["historical_role"] == "heldout"
                          else "archived only; original calibration label retained"),
            }
            if row["historical_role"] == "heldout":
                row["scoring_override"] = row["repeat_observation"]
    for row in normalized:
        m = row["measurement"]
        candidates = groups[(m["group_id"], m["world_size"], m["rows_per_peer"], m["k"], m["n"])]
        row["observed_candidate_count"] = len(candidates)
        row["complete_11_candidate_surface"] = set(candidates) == expected_families
        if row["environment_id"] == P8_ENV and not row["complete_11_candidate_surface"]:
            raise ValueError("incomplete P8 candidate surface")
    expected_shapes = {(8, s["rows_per_peer"], k, s["n"])
                       for s in manifest["sweeps"] for k in s["k_values"]}
    if {r.shape for r in rows if r.world_size == 8} != expected_shapes:
        raise ValueError("P8 shapes disagree with archived manifest")

    components = []
    gemm_root = repo / "experiments/2026-08-26-a100-cutlass-cost-model/results"
    for provider, filename in (("cublas", "cublas_gemm_measurements.jsonl"),
                               ("cutlass", "isolated_gemm_measurements.jsonl")):
        path = gemm_root / filename
        source = record(path)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            raw = json.loads(line)
            if raw["dtype"] != "float16" or raw.get("provider", provider) != provider:
                raise ValueError("unexpected GEMM precision/provider")
            components.append({
                "id": f"{GEMM_ENV}:{provider}:{raw['device']}:{raw['m']}:{raw['k']}:{raw['n']}",
                "environment_id": GEMM_ENV, "scope": "standalone_gemm",
                "provider": provider, "kernel": raw.get("kernel"), "dtype": raw["dtype"],
                "device": raw["device"], "m": raw["m"], "k": raw["k"], "n": raw["n"],
                "latency_ms": positive_ms(raw["median_us"], unit="us"),
                "latency_statistic": "median",
                "timing_scope": "CUDA events around standalone GEMM launch",
                "source": {**source, "line": number}, "eligible_for_fused_training": False,
            })
    for p, env, path in (
        (4, P4_ENV, repo / "experiments/2026-08-29-a100-grouped-n-plan/raw/extracted-v2/2026-08-29-g2/nccl/alltoall.txt"),
        (8, P8_ENV, p8_root / "logs/20_nccl_alltoall.txt"),
    ):
        components.extend(nccl_records(path, environment_id=env, world_size=p, source=record(path)))
    if len({r["id"] for r in components}) != len(components):
        raise ValueError("duplicate component measurement")
    environments = {}
    for env, p, evidence in (
        (GEMM_ENV, 4, "experiments/2026-08-26-runpod-4xa100-cost-model/README.md"),
        (P4_ENV, 4, "experiments/2026-08-29-a100-grouped-n-plan/README.md"),
        (P8_ENV, 8, "experiments/2026-08-30-a100-8gpu/README.md"),
    ):
        environments[env] = {
            "gpu": "NVIDIA A100-SXM4-80GB", "world_size": p, "sm_count": 108,
            "topology": "NV12 full peer access", "evidence": record(repo / evidence),
            "pooling_policy": "separate collection/stack; not interchangeable calibration",
        }
    out.mkdir(parents=True)
    for filename, records in (("selector_measurements.jsonl", normalized),
                              ("component_measurements.jsonl", components)):
        with (out / filename).open("x") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = {
        "schema": "edmir.offline-measurements.v1",
        "evidence_status": "retrospective development; no fresh held-out labels",
        "eligible_for_compiler_admission": False,
        "counts": {
            "selector_rows": len(normalized), "selector_shapes": len(groups),
            "historical_roles": dict(Counter(r["historical_role"] for r in normalized)),
            "scoring_overrides": sum(r["scoring_override"] is not None for r in normalized),
            "repeat_observations": sum(r["repeat_observation"] is not None for r in normalized),
            "component_rows": len(components),
            "component_scopes": dict(Counter(r["scope"] for r in components)),
        },
        "environments": environments, "inputs": sorted(inputs.values(), key=lambda x: x["path"]),
        "full_operator_collector": {
            "path": "EDM-IR/runtime/tools/benchmark_generated.cpp",
            "p4_commit": "f584d7d",
            "p8_commit": "72c01fd40d16210f667bd2c7f3697815f190c872",
            "sha256_at_both_commits": "4c9e705ae5bee2d5c03db6cf5c7c1cdb7a9264d665de24486a002b380ae7853c",
            "timed_region": "execute plus synchronize; preparation and input copies excluded",
            "percentile_rule": "sorted[floor(fraction * (n - 1))]",
        },
        "files": {name: sha256(out / name) for name in
                  ("selector_measurements.jsonl", "component_measurements.jsonl")},
        "normalization": {
            "latency": "ms; microsecond source values divided by 1000",
            "remote_input_bytes": "full-program sum across ranks; unchanged",
            "missing_candidates": "absent; never imputed",
            "repeat_medians": "replace only scoring labels; originals retained",
            "nccl_NA_wrong": "null, not zero",
            "external_component_data": "not imported into fused training",
            "device_gemm_rows": "four device observations per shape, not four new workloads",
            "full_operator_timer": "host wall time includes dispatch and synchronization; not isolated GPU kernel time",
            "reported_percentile": "collector uses sorted[floor(fraction*(iterations-1))], not interpolated quantiles",
        },
    }
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    load_selector_dataset(out)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(REPO, args.out)["counts"], indent=2))
