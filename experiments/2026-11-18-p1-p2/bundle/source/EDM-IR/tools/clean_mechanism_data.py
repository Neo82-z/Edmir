#!/usr/bin/env python3
"""Read-only validation and normalization of the frozen RunPod collection.

Raw samples remain intact. Quarantine applies to a whole configuration's three
processes, never to selected slow iterations. Warmup diagnostics stay separate.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_selector_validation import digest
from run_mechanism_gpu import check_row, summarize
from edmir.compiler.mechanism_prediction import HOSTS

STAGES = ("anchors", "mechanisms", "geometry")
FROZEN_PACKAGE = "54aff5ff71ff1dbab33a90348e84ec9d3bc1ea8b755e11d30c6ebd777bd6edc5"
FROZEN_BUILD = "95790bd5fe5316377d131af7d623077f91350d0a0d440123554359ad6ea42f93"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def safe_file(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or invalid sealed path: "+relative)
    return path


def validate_block_rows(rows, jobs, infos, package, context, stdout_rows, *, diagnostic_digest=None):
    if not (len(rows) == len(jobs) == len(stdout_rows)):
        raise ValueError("missing process/job/stdout")
    keys = set()
    for row, job, stdout in zip(rows, jobs, stdout_rows):
        key = (tuple(row["shape"]), row["candidate_id"], row["warmup"], row["repetition"])
        if key in keys:
            raise ValueError("duplicate process block")
        keys.add(key)
        trial = {**package, "warmup": job["diagnostic_warmup"]} if diagnostic_digest else package
        clean = {k: v for k, v in row.items() if k != "diagnostic_design_sha256"}
        validated = check_row(clean, job, infos[job["workload_id"]], trial)
        if clean != validated or validated != check_row(stdout, job, infos[job["workload_id"]], trial):
            raise ValueError("row, job, or raw stdout identity differs")
        if diagnostic_digest and row.get("diagnostic_design_sha256") != diagnostic_digest:
            raise ValueError("diagnostic design identity differs")
        if row["runtime"] != context["runtime"]:
            raise ValueError("runtime identity differs")
        if row["measurement_policy"] != {"profiler": False, "nvtx": False, "discarded_samples": 0}:
            raise ValueError("instrumented/filtered data cannot enter primary labels")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cleaning"]["config_id"]].append(row)
    configs = []
    for config_id, group in sorted(groups.items()):
        group.sort(key=lambda r: r["repetition"])
        if [r["repetition"] for r in group] != [0, 1, 2]:
            raise ValueError("configuration lacks three unique fresh-process blocks")
        r = group[0]
        values = [x["median_ms"]*1000 for x in group]
        windows = [statistics.median(x["samples_ms"][-20:])/statistics.median(x["samples_ms"][:20])
                   for x in group]
        flags = []
        if any(not .9 <= v <= 1.1 for v in windows):
            flags.append("within_process_drift")
        if max(values)/min(values) > 1.10:
            flags.append("between_process_spread")
        algorithms = sorted({x["gemm_algorithm_id"] for x in group})
        if len(algorithms) != 1:
            flags.append("algorithm_variation")
        primary = r["stage"] in ("mechanisms", "geometry") and r["metadata_mode"] in ("prepared", "merged")
        configs.append({"config_id": config_id, "context_id": r["cleaning"]["context_id"],
            "shape": r["shape"], "family": r["family"], "candidate_id": r["candidate_id"],
            "stage": r["stage"], "metadata_mode": r["metadata_mode"], "warmup": r["warmup"],
            "dependency_policy": None if r["family"] in HOSTS else "resource_queues_v1",
            "host_graph_note": "Native graph of the identified host implementation; fused policy is inapplicable." if r["family"] in HOSTS else None,
            "implementation_id": r["implementation_id"],
            "algorithm_ids": algorithms, "input_sha256": r["input_sha256"],
            "reference_sha256": r["reference_sha256"], "header_sha256": r["cleaning"]["header_sha256"],
            "process_medians_us": values, "observed_us": statistics.median(values),
            "last20_over_first20": windows, "max_process_over_min": max(values)/min(values),
            "quality_flags": flags, "primary_scope": primary,
            "eligible_for_calibration": primary and not flags,
            "process_ids": [x["cleaning"]["process_id"] for x in group],
            "iteration_samples_retained": sum(len(x["samples_ms"]) for x in group)})
    return configs


def clean_collection(root):
    root = Path(root).resolve()
    revision = root/"revision-01"
    bundle = root.parent/"bundle"  # revision-01 overlays sources on this sealed base bundle.
    envelope = read(revision/"package.json")
    package = envelope["package"]
    if digest(package) != envelope["package_sha256"] or digest(package) != FROZEN_PACKAGE:
        raise ValueError("wrong frozen execution package")
    files = {}
    for relative, expected in sorted(package["files"].items()):
        path = safe_file(bundle, relative)
        if sha(path) != expected:
            raise ValueError("sealed file changed: "+relative)
        files[relative] = expected
    for relative, expected in sorted(package["sources"].items()):
        source_root = revision/"source" if (revision/"source"/relative).exists() else bundle/"source"
        if sha(safe_file(source_root, relative)) != expected:
            raise ValueError("frozen source changed: "+relative)
    inputs = read(revision/"inputs.json")
    if inputs["package_sha256"] != FROZEN_PACKAGE:
        raise ValueError("input manifest belongs to another package")
    input_binary_hashes = {}
    for workload, info in sorted(inputs["workloads"].items()):
        for filename, field in (("inputs.bin", "input_sha256"), ("reference.bin", "reference_sha256")):
            relative = f"data/{workload}/{filename}"
            actual = sha(safe_file(bundle, relative))
            if actual != info[field]:
                raise ValueError("input/reference binary changed: "+relative)
            input_binary_hashes[relative] = actual
    build_dir = root/"server-backup/work/build"
    receipt_file = build_dir/"selector_build.json"
    receipt = read(receipt_file)
    if (sha(receipt_file) != FROZEN_BUILD or receipt["package_sha256"] != FROZEN_PACKAGE
            or set(receipt["binaries"]) != set(package["programs"])):
        raise ValueError("build receipt identity/coverage differs")
    for name, expected in receipt["binaries"].items():
        if sha(safe_file(build_dir, name)) != expected:
            raise ValueError("compiled binary changed: "+name)
    dependency_checks = []
    for original, expected in receipt["dependencies"].items():
        relative = Path(original).relative_to("/workspace/runpod-mechanism-collection")
        if relative.name == "libnccl.so":
            relative = Path("dependencies/libnccl.so.2.28.9")
        if not (root/"server-backup"/relative).is_file():
            dependency_checks.append({"original_path": original, "recorded_sha256": expected,
                                      "locally_verified": False, "reason": "Not included in the archived dependency subset."})
            continue
        if sha(safe_file(root/"server-backup", str(relative))) != expected:
            raise ValueError("archived build dependency changed: "+str(relative))
        dependency_checks.append({"original_path": original, "recorded_sha256": expected,
                                  "locally_verified": True, "archived_path": str(relative)})
    normalized, manifests, base_context = [], {}, None
    for stage in (*STAGES, "short_host_warmup_diagnostic"):
        diagnostic = stage not in STAGES
        directory = root/"server-backup/work/short-host-diagnostic-01" if diagnostic else root/"results"/(stage+"-01")
        context = read(directory/"context.json")
        common = {k: v for k, v in context.items() if k != "stage"}
        if base_context is None:
            base_context = common
        if common != base_context or common["build_sha256"] != FROZEN_BUILD:
            raise ValueError("stages have different hardware/build/timer context")
        if common["package_sha256"] != FROZEN_PACKAGE:
            raise ValueError("measurement package differs")
        raw = directory/"measurements.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        design = read(directory/"design.json") if diagnostic else None
        jobs = design["jobs"] if diagnostic else [j for j in package["jobs"] if j["stage"] == stage]
        stdout = [read(directory/f"stdout_{i:05d}.txt") for i in range(len(jobs))]
        validate_block_rows(rows, jobs, inputs["workloads"], package, context, stdout,
                            diagnostic_digest=digest(design) if diagnostic else None)
        summary = read(directory/"summary.json")
        if diagnostic:
            if (summary["raw_sha256"] != sha(raw) or summary["design_sha256"] != digest(design)
                    or summary["primary_dataset"] is not False):
                raise ValueError("diagnostic seal differs")
        else:
            run = read(directory/"run.json")
            if (context["stage"] != stage or run["context"] != context or run["raw_sha256"] != sha(raw)
                    or summarize(package, rows, stage) != summary):
                raise ValueError("run seal or regenerated summary differs")
        for row, job in zip(rows, jobs):
            # Timing/source identity, not a v2 current-runtime evidence identity.
            context_id = digest({"context": common, "warmup": row["warmup"],
                                 "iterations": row["iterations"], "timing_scope": row["timing_scope"],
                                 "measurement_policy": row["measurement_policy"]})
            config_id = digest({"context_id": context_id, "stage": stage, "shape": row["shape"],
                                "candidate_id": row["candidate_id"], "implementation_id": row["implementation_id"]})
            header = package["programs"][job["executable"]]["header"]
            normalized.append({**row, "cleaning": {"context_id": context_id, "config_id": config_id,
                "process_id": digest({"config": config_id, "block": row["repetition"]}),
                "source_file": str(raw.relative_to(root)), "source_ordinal": row["ordinal"],
                "header_sha256": package["files"][header], "raw_row_sha256": digest(row)}})
        manifests[stage] = {"raw_sha256": sha(raw), "context_sha256": sha(directory/"context.json"),
            "processes": len(rows), "samples": sum(len(r["samples_ms"]) for r in rows),
            "stdout_revalidated": len(stdout)}
    configs = aggregate(normalized)
    return normalized, configs, {"schema": "edmir.cleaned-mechanisms.v1",
        "package_sha256": FROZEN_PACKAGE, "build_sha256": FROZEN_BUILD,
        "verified_package_files": len(files), "verified_sources": len(package["sources"]),
        "verified_compiled_binaries": len(receipt["binaries"]),
        "archived_build_dependency_checks": dependency_checks,
        "input_manifest_sha256": sha(revision/"inputs.json"),
        "input_binary_hashes": input_binary_hashes,
        "source_layout": {"generated_files_and_inputs": "../bundle",
                          "runtime_sources": "revision-01/source overlay on ../bundle/source; every selected hash verified"},
        "execution_context": base_context, "stages": manifests,
        "all_processes": len(normalized), "all_samples": sum(len(r["samples_ms"]) for r in normalized),
        "all_configurations": len(configs), "primary_configurations": sum(r["primary_scope"] for r in configs),
        "calibration_configurations": sum(r["eligible_for_calibration"] for r in configs),
        "quarantined_primary": [r["config_id"] for r in configs if r["primary_scope"] and r["quality_flags"]],
        "quality_rule": "Whole 3-process configuration: last20/first20 outside [0.9,1.1], max/min >1.10, or algorithm change.",
        "raw_samples_discarded": 0, "independent_validation": False,
        "protocol_note": "20/200 warmups, legacy/prepared/merged, profiler captures and old/new sources are not interchangeable."}
