#!/usr/bin/env python3
"""Build or collect a prepared package on an existing GPU server; never rent one."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from prepare_selector_gpu import REPO, verify_package, implementation_id, LEGACY_CONTROLS
from independent_selector_validation import BASELINES, digest, score
from edmir.compiler.measurement_dataset import sha256


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)


def git_commit(path):
    return command(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def build(package, directory, cutlass, nccl_source, nccl_include, nccl_library, jobs):
    p = verify_package(package)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/"selector_build.json"
    if receipt.exists():
        raise ValueError("build receipt exists; use a new directory for a new build")
    # Record source identities plus the actual headers/library, not just names.
    versions = (cutlass/"include/cutlass/version.h").read_text()
    version = ".".join(re.search(rf"#define CUTLASS_{part}\s+(\d+)", versions).group(1)
                       for part in ("MAJOR", "MINOR", "PATCH"))
    identities = {"cutlass_commit": git_commit(cutlass), "cutlass_version": version,
                  "nccl_commit": git_commit(nccl_source)}
    for source in (cutlass, nccl_source):
        if command(["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
                   capture_output=True).stdout.strip():
            raise ValueError(f"dependency has tracked modifications; record a clean commit first: {source}")
    dependencies = {str(nccl_library): sha256(nccl_library),
                    str(nccl_include/"nccl.h"): sha256(nccl_include/"nccl.h"),
                    str(nccl_include/"nccl_device.h"): sha256(nccl_include/"nccl_device.h")}
    command(["cmake", "-S", REPO/"EDM-IR/runtime", "-B", directory, "-DCMAKE_BUILD_TYPE=Release",
             "-DEDMIR_ENABLE_CUDA=ON", "-DEDMIR_ENABLE_CUTLASS=ON", "-DEDMIR_ENABLE_NCCL_DEVICE=ON",
             f"-DEDMIR_CUTLASS_DIR={cutlass}", f"-DEDMIR_EXPERIMENT_DIR={package}",
             f"-DNCCL_INCLUDE_DIR={nccl_include}", f"-DNCCL_LIBRARY={nccl_library}",
             f"-DCMAKE_BUILD_RPATH={nccl_library.parent}"])
    command(["cmake", "--build", directory, "-j", jobs])
    command(["ctest", "--test-dir", directory, "--output-on-failure"])
    binaries = {name: sha256(directory/name) for name in p["programs"]}
    write_json(receipt, {"package_sha256": digest(p), "dependencies": dependencies,
                        "environment": identities, "binaries": binaries,
                        "nccl_library_sha256": sha256(nccl_library),
                        "nccl_source": str(nccl_source), "cutlass_source": str(cutlass),
                        "note": "Source commits and linked library hash recorded; verify the library was built from this NCCL source."})


def validate_measurement(row, job, inputs, package):
    for field in ("family", "shape"):
        if row.get(field) != job[field]:
            raise ValueError(f"runner returned a different {field}")
    p, m, k, n = job["shape"]
    if row.get("correctness_passed") is not True or row.get("nonfinite_outputs") != 0:
        raise ValueError("numerical check failed")
    if row.get("checked_elements") != p*p*m*n:
        raise ValueError("runner did not validate every output")
    for field in ("max_abs_error", "max_rel_error", "median_ms"):
        if not isinstance(row.get(field), (float, int)) or isinstance(row[field], bool) or not math.isfinite(row[field]):
            raise ValueError("nonfinite or missing runner result")
    if row["max_abs_error"] < 0 or row["max_rel_error"] < 0 or row["median_ms"] <= 0:
        raise ValueError("negative accuracy or nonpositive timing")
    if row.get("atol") != 0.02 or row.get("rtol") != 0.02 or (
        row["max_abs_error"] > 0.02 and row["max_rel_error"] > 0.02):
        raise ValueError("runner tolerance/check does not match protocol")
    if row.get("correctness_reference") != "CPU_FP32_on_identical_FP16_inputs" or row.get("input_seed") != 20261020:
        raise ValueError("input/reference contract differs")
    for field in ("warmup", "iterations"):
        if row.get(field) != package[field]:
            raise ValueError("runner timing protocol differs")
    samples = row.get("samples_ms", [])
    if len(samples) != package["iterations"] or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("raw iteration samples are missing or invalid")
    if not math.isclose(statistics.median(samples), row["median_ms"], rel_tol=1e-12):
        raise ValueError("reported median does not match raw iteration samples")
    if "metadata_mode" in job:
        if row.get("metadata_mode") != job["metadata_mode"]:
            raise ValueError("metadata mode mismatch")
        expected = implementation_id(job["family"], job["metadata_mode"])
        if expected != job["implementation_id"]:
            raise ValueError("package implementation contract mismatch")
        if row.get("timing_scope") != package["timing_scope"]:
            raise ValueError("timing boundary mismatch")
        if row.get("measurement_policy") != package["measurement_policy"]:
            raise ValueError("measurement filtering/instrumentation mismatch")
        summaries = {"mean_ms": statistics.mean(samples), "total_ms": math.fsum(samples),
                     "min_ms": min(samples), "max_ms": max(samples)}
        for field, calculated in summaries.items():
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (
                not math.isclose(value, calculated, rel_tol=1e-12)):
                raise ValueError(f"{field} differs from all raw samples")
    elif job["family"] in BASELINES:
        expected = BASELINES[job["family"]].baseline_id
    else:
        from edmir.compiler.two_stage_selection import parse_schedule_family
        q, g = parse_schedule_family(job["family"])
        expected = f"direct_peer_cutlass_sm80_q{q}_g{g}"
    if row.get("implementation_id") != expected or not row.get("gemm_algorithm_id"):
        raise ValueError("implementation/algorithm identity mismatch")
    return {**row, "input_sha256": inputs["input_sha256"], "reference_sha256": inputs["reference_sha256"],
            "repetition": job["repetition"], "trial_order": job["trial_order"],
            **({"candidate_id": job["candidate_id"]} if "candidate_id" in job else {})}


def measurement_environment(environ):
    """Reject known debug/injection modes; record settings without claiming to detect every attachment."""
    values = {k: v for k, v in environ.items()
              if k.startswith(("CUDA_", "NCCL_", "CUBLAS_", "NSYS_", "NVPROF_"))
              or k in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    if values.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise ValueError("CUDA_LAUNCH_BLOCKING changes the timing contract")
    if values.get("CUDA_INJECTION64_PATH") or values.get("CUDA_INJECTION32_PATH"):
        raise ValueError("CUDA profiler injection is enabled")
    if values.get("CUDA_PROFILE", "0") not in ("", "0"):
        raise ValueError("CUDA profiling is enabled")
    if values.get("NCCL_DEBUG", "WARN").upper() in ("INFO", "TRACE"):
        raise ValueError("NCCL INFO/TRACE logging must be disabled for timing")
    preload = values.get("LD_PRELOAD", "").lower()
    if any(name in preload for name in ("cupti", "nvperf", "nsight", "nvprof")):
        raise ValueError("profiling library preload is enabled")
    return values


def summarize_metadata(p, rows):
    """Keep independent-process observations and pair only matching repetition IDs."""
    if len(rows) != len(p["jobs"]):
        raise ValueError("metadata analysis needs every scheduled process")
    contracts = p["implementation_contracts"]
    expected_keys = {(tuple(j["shape"]), j["candidate_id"], j["repetition"]) for j in p["jobs"]}
    by_key = {}
    for row in rows:
        key = (tuple(row["shape"]), row["candidate_id"], row["repetition"])
        if key not in expected_keys or key in by_key:
            raise ValueError("unexpected or duplicate metadata observation")
        by_key[key] = row
    if by_key.keys() != expected_keys:
        raise ValueError("incomplete metadata observations")
    workloads = []
    for work in p["workloads"]:
        shape = tuple(work["shape"])
        candidates = {}
        for name in contracts:
            repetitions = [by_key[(shape, name, i)] for i in range(p["repetitions"])]
            medians = [r["median_ms"] for r in repetitions]
            means = [r["mean_ms"] for r in repetitions]
            candidates[name] = {
                "process_medians_ms": medians, "process_means_ms": means,
                "median_of_process_medians_ms": statistics.median(medians),
                "mean_of_process_means_ms": statistics.mean(means),
            }
        paired = {}
        for family in LEGACY_CONTROLS:
            before, after = candidates[family+"@legacy"], candidates[family+"@prepared"]
            paired[family] = {
                "legacy_over_prepared_process_median_ratios": [
                    b/a for b, a in zip(before["process_medians_ms"], after["process_medians_ms"])],
                "legacy_over_prepared_process_mean_ratios": [
                    b/a for b, a in zip(before["process_means_ms"], after["process_means_ms"])],
            }
        primary = [n for n, c in contracts.items() if c["primary"]]
        hosts = [n for n in primary if contracts[n]["family"] in BASELINES]
        fastest = lambda names: min(names, key=lambda n: candidates[n]["median_of_process_medians_ms"])
        workloads.append({"shape": list(shape), "candidates": candidates, "paired_legacy_controls": paired,
                          "fastest_primary_host": fastest(hosts), "observed_best_primary": fastest(primary)})
    return {"status": "development ablation; no automatic promotion", "workloads": workloads}


def environment(runtime, topo, inventory, world_size):
    devices = runtime["devices"]
    if len(devices) != world_size:
        raise ValueError("wrong device count")
    bus_to_index = {}
    for line in inventory.splitlines():
        index, bus = [x.strip() for x in line.split(",")]
        # nvidia-smi may use an 8-digit PCI domain while CUDA uses 4.
        bus_to_index[bus.lower().lstrip("0")] = index
    chosen = []
    for d in devices:
        if d["name"] != "NVIDIA A100-SXM4-80GB" or (d["major"], d["minor"], d["sm_count"]) != (8, 0, 108):
            raise ValueError("this frozen package requires A100-SXM4-80GB, sm_80, 108 SMs")
        if d["peers"] != [True]*world_size:
            raise ValueError("full peer access is required")
        chosen.append("GPU"+bus_to_index[d["pci_bus_id"].lower().lstrip("0")])
    if len(set(chosen)) != world_size:
        raise ValueError("duplicate physical devices")
    lines = [line.split() for line in topo.splitlines()]
    header = next(row for row in lines if row and row[0] == "GPU0")
    # The header begins GPU0 GPU1 ...; actual rows begin GPU0 X NV12 ...
    if len(header) < 2 or header[1] != "GPU1":
        raise ValueError("cannot parse nvidia-smi topology header")
    columns = {name: i for i, name in enumerate(header) if re.fullmatch(r"GPU\d+", name)}
    matrix = {row[0]: row[1:] for row in lines if row and re.fullmatch(r"GPU\d+", row[0])
              and len(row) > 1 and row[1] != "GPU1"}
    for a in chosen:
        for b in chosen:
            if a != b and matrix[a][columns[b]] != "NV12":
                raise ValueError(f"required NV12 topology missing between {a} and {b}")
    return {"gpu": "NVIDIA A100-SXM4-80GB", "arch": "sm_80", "sm_count": 108,
            "link_label": "NV12", "full_peer_access": True, "world_size": world_size,
            **{key: runtime[key] for key in ("driver_version", "cuda_version", "nccl_version", "cublas_version")}}


def collect(package, directory, out, *, resume=False):
    p = verify_package(package)
    timing_environment = measurement_environment(os.environ) if p["mode"] == "metadata_ablation_development" else None
    receipt = json.loads((directory/"selector_build.json").read_text())
    if receipt["package_sha256"] != digest(p):
        raise ValueError("binary build belongs to another package")
    for path, expected in receipt["dependencies"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("build dependency changed")
    for name, expected in receipt["binaries"].items():
        if sha256(directory/name) != expected:
            raise ValueError("compiled executable changed")
    # RUNPATH may be overridden by LD_LIBRARY_PATH. Check what the Linux
    # loader will resolve, rather than assuming the configured NCCL is loaded.
    libraries = {}
    for name in receipt["binaries"]:
        linked = command(["ldd", directory/name], capture_output=True).stdout
        if "not found" in linked:
            raise ValueError("a runtime shared library is missing")
        nccl_paths = []
        for library, path in re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)\s+\(", linked, re.MULTILINE):
            resolved = str(Path(path).resolve())
            if resolved not in libraries:
                libraries[resolved] = sha256(Path(resolved))
            if library.startswith("libnccl.so"):
                nccl_paths.append(resolved)
        if len(nccl_paths) != 1 or libraries[nccl_paths[0]] != receipt["nccl_library_sha256"]:
            raise ValueError("loader NCCL differs from the recorded build library")
    inputs = json.loads((package/"inputs.json").read_text())
    if inputs["package_sha256"] != digest(p):
        raise ValueError("inputs belong to another package")
    for work in p["workloads"]:
        for file, key in (("inputs.bin", "input_sha256"), ("reference.bin", "reference_sha256")):
            if sha256(package/"data"/work["id"]/file) != inputs["workloads"][work["id"]][key]:
                raise ValueError("input or reference content changed")
    if out.exists() and not resume:
        raise ValueError("output exists; explicitly use --resume for the same run")
    out.mkdir(parents=True, exist_ok=True)
    topo = command(["nvidia-smi", "topo", "-m"], capture_output=True).stdout
    inventory = command(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                        capture_output=True).stdout
    preflight = json.loads(command([directory/p["jobs"][0]["executable"], "--inspect"],
                                   capture_output=True, timeout=60).stdout)
    environment(preflight, topo, inventory, p["jobs"][0]["shape"][0])
    context = {"package_sha256": digest(p), "build_sha256": sha256(directory/"selector_build.json"),
               "topology": topo, "inventory": inventory,
               "runtime": preflight,
               "loaded_library_sha256": libraries,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    if timing_environment is not None:
        context["measurement_environment"] = timing_environment
    context_path = out/"context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("resume environment/build differs")
    write_json(context_path, context)
    raw_path = out/"measurements.jsonl"
    existing = [json.loads(s) for s in raw_path.read_text().splitlines()] if raw_path.exists() else []
    if len(existing) > len(p["jobs"]):
        raise ValueError("too many previous rows")
    runtimes, rows = [], []
    for i, job in enumerate(p["jobs"]):
        info = inputs["workloads"][job["workload_id"]]
        if i < len(existing):
            row = existing[i]
            if row != validate_measurement(row, job, info, p):
                raise ValueError("existing row provenance changed")
        else:
            data = package/"data"/job["workload_id"]
            args = [str(a) for a in (directory/job["executable"], job["family"], data/"inputs.bin",
                                    data/"reference.bin", p["warmup"], p["iterations"])]
            if "metadata_mode" in job:
                args.append(job["metadata_mode"])
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired as e:
                write_json(out/"failure.json", {"job": job, "reason": "process exceeded 180 seconds"})
                (out/f"stderr_{i:05d}.txt").write_bytes(e.stderr or b"")
                raise
            (out/f"stderr_{i:05d}.txt").write_text(result.stderr)
            (out/f"stdout_{i:05d}.txt").write_text(result.stdout)
            if result.returncode:
                write_json(out/"failure.json", {"job": job, "returncode": result.returncode})
                raise RuntimeError(f"GPU process failed; preserved output for job {i} in {out}")
            row = validate_measurement(json.loads(result.stdout), job, info, p)
        env = environment(row["runtime"], topo, inventory, job["shape"][0])
        if row["runtime"] != preflight:
            raise ValueError("runtime/devices differ from preflight")
        if runtimes and row["runtime"] != runtimes[0]:
            raise ValueError("runtime/devices changed during the run")
        runtimes.append(row["runtime"])
        if i >= len(existing):
            with raw_path.open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
                f.flush()
            print(f'{i+1}/{len(p["jobs"])} {job["workload_id"]} {job.get("candidate_id", job["family"])}', flush=True)
        rows.append(row)
    raw_hash = sha256(raw_path)
    rows = [{**row, "source": {"path": raw_path.name, "sha256": raw_hash, "line": i+1}}
            for i, row in enumerate(rows)]
    env.update(receipt["environment"])
    env["collector_sha256"] = p["collector_sha256"]
    env["loaded_library_sha256"] = libraries
    run = {"schema": ("edmir.independent-selector-run.v1" if p["mode"] == "independent" else
                      "edmir.metadata-ablation-run.v1" if p["mode"] == "metadata_ablation_development" else
                      "edmir.selector-pilot-run.v1"),
           "plan_sha256": p["plan_sha256"], "package_sha256": digest(p), "environment": env,
           "environment_id": (json.loads((package/"independent_plan.json").read_text())["plan"]["environment_id"]
                              if p["mode"] == "independent" else "pilot-a100-p4-development"),
           "timing_scope": p["timing_scope"], "dtype": "float16", "latency_unit": "ms",
           "baseline_contracts": p["baseline_contracts"], "fresh_process_per_repetition": True, "rows": rows}
    if p["mode"] == "metadata_ablation_development":
        run["environment_id"] = "metadata-ablation-a100-p4-development-v2"
        run["implementation_contracts"] = p["implementation_contracts"]
    write_json(out/"run.json", run)
    if p["mode"] == "independent":
        write_json(out/"score.json", score(json.loads((package/"independent_plan.json").read_text()), out/"run.json"))
    elif p["mode"] == "metadata_ablation_development":
        write_json(out/"summary.json", summarize_metadata(p, rows))
    else:
        summary = []
        for work in p["workloads"]:
            times = {name: statistics.median(r["median_ms"] for r in rows
                     if r["shape"] == work["shape"] and r["family"] == name)
                     for name in work["trial_orders"][0]}
            summary.append({"shape": work["shape"], "median_ms": times,
                            "fastest_host": min(BASELINES, key=lambda n: times[n]),
                            "observed_best": min(times, key=times.get)})
        write_json(out/"summary.json", {"status": "pilot development only", "workloads": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--package", type=Path, required=True)
    b.add_argument("--build-dir", type=Path, required=True)
    b.add_argument("--cutlass", type=Path, required=True)
    b.add_argument("--nccl-source", type=Path, required=True)
    b.add_argument("--nccl-include", type=Path, required=True)
    b.add_argument("--nccl-library", type=Path, required=True)
    b.add_argument("--jobs", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--package", type=Path, required=True)
    r.add_argument("--build-dir", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        if args.jobs < 1: parser.error("--jobs must be positive")
        build(args.package.resolve(), args.build_dir.resolve(), args.cutlass.resolve(),
              args.nccl_source.resolve(), args.nccl_include.resolve(), args.nccl_library.resolve(), args.jobs)
    else:
        collect(args.package.resolve(), args.build_dir.resolve(), args.out.resolve(), resume=args.resume)
