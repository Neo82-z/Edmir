#!/usr/bin/env python3
"""Read-only, bounded evidence verification. Does not build or launch GPU code."""
import csv
from hashlib import sha256 as hash_bytes
import io
import json
from pathlib import Path
import subprocess
import sys

import run_audit as audit
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from prepare_selector_gpu import verify_package, digest

CUTLASS_COMMIT = "dcf215af68a2d08d305076c152a06f201728cd53"
CUTLASS = audit.ROOT.parent/"cutlass"


def git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), *args], check=True,
                          capture_output=True, timeout=30).stdout


def verify(out):
    if out.exists():
        raise ValueError("choose a fresh output")
    directory = audit.ROOT/"experiments/2026-10-20-offline-predictor/data"
    data, _, _, manifest = load_identified_selector_dataset(directory)
    source_checks, rows = {}, []
    records = [json.loads(line) for line in (directory/"selector_measurements.jsonl").read_text().splitlines()]
    for record in records:
        source = record["source"]
        path = audit.ROOT/source["path"]
        if source["path"] not in source_checks:
            actual = sha256(path)
            assert actual == source["sha256"], str(path)
            source_checks[source["path"]] = actual
        lines = path.read_text().splitlines()
        parsed = next(csv.DictReader(io.StringIO(lines[0]+"\n"+lines[source["line"]-1]+"\n")))
        value = record["measurement"]
        assert parsed["family"] == value["family"]
        for key in ("rows_per_peer", "k", "n"):
            assert int(parsed[key]) == value[key]
        assert float(parsed["median_ms"]) == value["measured_ms"]
        rows.append({"id": record["id"], "source_line_verified": True})
    inputs = {}
    for item in manifest["inputs"]:
        actual = sha256(audit.ROOT/item["path"])
        assert actual == item["sha256"], item["path"]
        inputs[item["path"]] = actual
    collector = manifest["full_operator_collector"]
    collector_checks = {}
    for key in ("p4_commit", "p8_commit"):
        actual = hash_bytes(git(audit.ROOT, "show", f'{collector[key]}:{collector["path"]}')).hexdigest()
        assert actual == collector["sha256_at_both_commits"]
        collector_checks[key] = {"commit": collector[key], "sha256": actual}
    edmir_sources = {}
    for path in ("EDM-IR/runtime/src/nccl_device_fused_cutlass.cu",
                 "EDM-IR/runtime/src/cutlass_sm80_gemm.cuh"):
        historical = hash_bytes(git(audit.ROOT, "show", f"f584d7d:{path}")).hexdigest()
        current = sha256(audit.ROOT/path)
        assert historical == current, path
        edmir_sources[path] = {"historical_sha256": historical, "current_sha256": current,
                               "unchanged": True}
    headers = {}
    for path in ("include/cutlass/gemm/kernel/gemm.h",
                 "include/cutlass/gemm/threadblock/threadblock_swizzle.h",
                 "include/cutlass/gemm/threadblock/default_mma_core_sm80.h"):
        archived = hash_bytes(git(CUTLASS, "show", f"{CUTLASS_COMMIT}:{path}")).hexdigest()
        headers[path] = {
            "archived_sha256": archived, "working_tree_sha256": sha256(CUTLASS/path),
            "unchanged": archived == sha256(CUTLASS/path),
        }
    package_path = audit.ROOT/"experiments/2026-10-20-local-replay-runtime-checks/gpu_pilot"
    package = verify_package(package_path)
    input_manifest = json.loads((package_path/"inputs.json").read_text())
    assert input_manifest["package_sha256"] == digest(package)
    for work in input_manifest["workloads"].values():
        # The package's workload IDs are the manifest keys, verified below.
        assert len(work["shape"]) == 4
    for name, work in input_manifest["workloads"].items():
        for filename, key in (("inputs.bin", "input_sha256"), ("reference.bin", "reference_sha256")):
            assert sha256(package_path/"data"/name/filename) == work[key]
    geometry = []
    for shape in sorted(audit.PRIORITY):
        _, m, k, n = shape
        for q in (1, 2, 4, 8, 16):
            for g in (1, 2):
                tb_m, tb_n, tb_k = 64, 128*g, 32
                grid = ((m+tb_m-1)//tb_m, (n+tb_n-1)//tb_n, q)
                depth = (((k+tb_k-1)//tb_k+q-1)//q)*tb_k
                assert depth*q == k
                geometry.append({
                    "shape": list(shape), "q": q, "g": g,
                    "threadblock": [tb_m, tb_n, tb_k],
                    "warp": [32, 64, 32], "threads_per_cta_from_source": 128*g,
                    "grid_from_source": list(grid), "partition_k_from_source": depth,
                    "serial_split_k_reduction": q > 1,
                })
    result = {
        "schema": "edmir.selector-evidence-verification.v1",
        "historical_sources": source_checks, "verified_original_csv_rows": len(rows),
        "manifest_input_checks": inputs, "collector": collector_checks,
        "kernel_source": {
            "edmir_historical_commit": "f584d7d", "edmir_files": edmir_sources,
            "cutlass_historical_commit": CUTLASS_COMMIT,
            "cutlass_current_head": git(CUTLASS, "rev-parse", "HEAD").decode().strip(),
            "cutlass_headers": headers, "priority_geometry": geometry,
            "geometry_status": "Derived from archived source. Not a compiled-template or GPU execution check.",
            "host_template_build": {
                "passed": False, "log": "geometry_build.txt",
                "reason": "CUDA libcu++ headers conflict with macOS libc++; no template executable was produced.",
            },
        },
        "frozen_package": {
            "path": str(package_path.relative_to(audit.ROOT)),
            "package_sha256": digest(package), "source_files_verified": len(package["sources"]),
            "generated_files_verified": len(package["files"]), "input_workloads_verified": len(input_manifest["workloads"]),
            "gpu_built_or_executed": False,
        },
        "primary_rows": len(data.rows), "primary_shapes": len(data.validate()),
        "server_ready": False,
    }
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("verified_original_csv_rows", "primary_rows",
                                            "primary_shapes", "frozen_package")}, indent=2))


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
