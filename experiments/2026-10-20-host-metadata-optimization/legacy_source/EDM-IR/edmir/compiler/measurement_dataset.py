"""Read the audited offline selector bundle, never component benchmark rows."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .relative_selection import strict_surfaces
from .two_stage_selection import ScheduleMeasurement

P4_ENV = "a100-sxm4-80gb-p4-2026-08-29"
P8_ENV = "a100-sxm4-80gb-p8-2026-08-30"
GEMM_ENV = "a100-sxm4-80gb-p4-2026-08-26"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selector_dataset(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "edmir.offline-measurements.v1":
        raise ValueError("unsupported offline dataset schema")
    required_files = {"selector_measurements.jsonl", "component_measurements.jsonl"}
    if set(manifest["files"]) != required_files:
        raise ValueError("manifest must hash both normalized data files")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"dataset hash mismatch: {name}")
    p4, calibration, evaluation = [], [], []
    ids = set()
    records = []
    for line in (directory / "selector_measurements.jsonl").read_text().splitlines():
        record = json.loads(line)
        records.append(record)
        if record["id"] in ids:
            raise ValueError(f"duplicate measurement id: {record['id']}")
        ids.add(record["id"])
        if (record["scope"], record["latency_unit"], record["dtype"]) != ("full_operator", "ms", "float16"):
            raise ValueError("selector accepts full-operator FP16 millisecond labels only")
        row = ScheduleMeasurement(**record["measurement"])
        row.validate()
        env, role = record["environment_id"], record["historical_role"]
        expected_p = {P4_ENV: 4, P8_ENV: 8}.get(env)
        if (row.world_size != expected_p or
                manifest["environments"][env]["world_size"] != expected_p):
            raise ValueError("measurement world size contradicts environment")
        environment = manifest["environments"][env]
        if (environment["gpu"], environment["sm_count"]) != ("NVIDIA A100-SXM4-80GB", 108):
            raise ValueError("dataset environment is outside the archived A100 contract")
        kernel = "sm80_64x128x32" + (f"_splitk{row.split_k_slices}" if row.split_k_slices > 1 else "")
        if record["kernel"] != kernel:
            raise ValueError("kernel identity does not match schedule family")
        if record["timing_scope"] != "host steady_clock around backend.execute() plus backend.synchronize()":
            raise ValueError("dataset timing scope differs from the full-program wall timer")
        if record["reported_timing"]["median_ms"] != row.measured_ms:
            raise ValueError("measurement label differs from reported median")
        if record["rows_per_destination"] != row.world_size * row.rows_per_peer:
            raise ValueError("rows_per_destination must equal P*M")
        override = record.get("scoring_override")
        if override is not None and role != "heldout":
            raise ValueError("repeat overrides are scoring-only")
        if override is not None and override != record.get("repeat_observation"):
            raise ValueError("scoring override lacks matching repeat provenance")
        if (env, role) == (P4_ENV, "development"):
            p4.append(row)
        elif (env, role) == (P8_ENV, "calibration"):
            calibration.append(row)
        elif (env, role) == (P8_ENV, "heldout"):
            evaluation.append(replace(row, measured_ms=override["median_ms"])
                              if override else row)
        else:
            raise ValueError("unknown environment/role combination")
    for subset in (p4, calibration, evaluation):
        strict_surfaces(subset)
    if {r.shape for r in calibration} & {r.shape for r in evaluation}:
        raise ValueError("calibration and scoring workloads overlap")
    all_rows = (*p4, *calibration, *evaluation)
    grouped = strict_surfaces(all_rows)
    for record in records:
        row = ScheduleMeasurement(**record["measurement"])
        count = len(grouped[row.grouped_shape])
        if record["observed_candidate_count"] != count:
            raise ValueError("reported candidate coverage does not match observations")
        if record["complete_11_candidate_surface"] != (count == 11):
            raise ValueError("reported complete-surface flag is inconsistent")
    counts = manifest["counts"]
    actual = {
        "selector_rows": len(all_rows), "selector_shapes": len(grouped),
        "historical_roles": dict(Counter(r["historical_role"] for r in records)),
        "scoring_overrides": sum(r.get("scoring_override") is not None for r in records),
        "repeat_observations": sum(r.get("repeat_observation") is not None for r in records),
    }
    if any(counts.get(key) != value for key, value in actual.items()):
        raise ValueError("manifest counts disagree with selector records")
    return tuple(p4), tuple(calibration), tuple(evaluation), manifest


def load_identified_selector_dataset(directory: Path):
    """Explicit migration of the audited v1 archive; never relabel its times.

    The loader above enforces the archive's kernel, environment and wall timer.
    Its retained runners use P sequential, per-source CUTLASS GEMMs. The old
    JSONL bytes remain unchanged; consumers get a mandatory baseline contract.
    """
    from .factorized_selection import IdentifiedMeasurements
    from .host_baseline import PER_SOURCE_CUTLASS

    p4, calibration, evaluation, manifest = load_selector_dataset(directory)
    datasets = tuple(IdentifiedMeasurements(rows, env, PER_SOURCE_CUTLASS)
                     for rows, env in ((p4, P4_ENV), (calibration, P8_ENV), (evaluation, P8_ENV)))
    for dataset in datasets:
        dataset.validate()
    return (*datasets, {**manifest, "identified_host_baseline": PER_SOURCE_CUTLASS.to_dict()})
