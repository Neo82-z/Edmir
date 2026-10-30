#!/usr/bin/env python3
"""Inspect pinned public files as data; never import or execute downloaded code."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(catalog_path: Path, cache: Path):
    catalog = json.loads(catalog_path.read_text())
    files = {}
    for source in catalog["files"]:
        repo, revision, path = source["repo"], source["revision"], source["path"]
        # URL construction is limited to immutable GitHub raw revisions.
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("public source must be pinned to a commit")
        if any(part == ".." for part in Path(repo, path).parts):
            raise ValueError("invalid source path")
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        if source["url"] != url:
            raise ValueError("source URL disagrees with pinned revision")
        local = cache / repo / path
        if local.exists():
            data = local.read_bytes()
        else:
            data = urllib.request.urlopen(url, timeout=45).read()
            if digest(data) != source["sha256"]:
                raise ValueError(f"download checksum mismatch: {url}")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        if digest(data) != source["sha256"] or len(data) != source["bytes"]:
            raise ValueError(f"cached source checksum mismatch: {local}")
        files[repo, path] = local

    def csv_rows(repo, path):
        with files[repo, path].open(newline="") as stream:
            return list(csv.DictReader(stream))

    tilelang = []
    for path in ("ampere_benchmark/dense_matmul/data/data_float16_gemm.py",
                 "ampere_benchmark/contiguous_dequant_matmul/data/n8192_k8192_data.py"):
        # Literal tables only. Importing these files would execute log parsing.
        tree = ast.parse(files["tile-ai/tilelang-benchmark", path].read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "matmul_times_data"
                                  for t in n.targets))
        providers = ast.literal_eval(assignment.value)
        tilelang.append({
            "path": path,
            "providers": [{"name": name, "entries": len(values),
                           "positive_values": sum(v > 0 for v in values),
                           "missing_minus_one": sum(v == -1 for v in values)}
                          for name, values in providers],
            "decision": "not fused training; missing cuBLAS/Triton labels, incomplete timing provenance",
        })
    gp = csv_rows("pavlyhalim/GPPerf", "cutlass-analysis/data.csv")
    comm = csv_rows("bandlayash/multi-gpu-perf-project",
                    "results/a100_4gpu/nccl_sweep_results.csv")
    tb_repo = "Deep-Learning-Profiling-Tools/Tilebench"
    config = files[tb_repo, "tilebench/benchmarks/operators/matmul_fp32_fp16_fp8/config.yaml"].read_text()
    defaults = config.split("case_defaults:", 1)[1].split("case_grid:", 1)[0]
    fixed_dims = {name: int(re.search(rf"^\s+{name}:\s*(\d+)\s*$", defaults, re.M)[1])
                  for name in ("M", "N")}
    tilebench = {}
    for mode in ("default", "autotune"):
        table = csv_rows(tb_repo, f"results/B200/csv/matmul_fp32_fp16_fp8_{mode}.csv")
        fp16 = [r for r in table if r["dtype"] == "fp16"]
        tilebench[mode] = {
            "rows": len(table), "fp16_shapes": len(fp16), "fixed_dimensions": fixed_dims,
            "k_values": sorted({int(re.fullmatch(r"K=(\d+)", r["params"])[1]) for r in fp16}),
            "time_columns_ms": [name for name in table[0] if name.endswith("_ms")],
        }
    return {
        "schema": "edmir.public-source-audit.v1",
        "catalog_sha256": digest(catalog_path.read_bytes()),
        "checked_files": len(files), "admitted_public_fused_training_rows": 0,
        "tilelang_benchmark": tilelang,
        "gpperf_cutlass": {
            "rows": len(gp), "gpus": sorted({r["gpu_name"] for r in gp}),
            "kernels": sorted({r["kernel_name"] for r in gp}),
            "m_values": sorted({int(r["m"]) for r in gp}),
            "beta_counts": dict(Counter(r["beta"] for r in gp)),
            "decision": "external SIMT SGEMM, not A100 FP16 Tensor Core or fused labels",
            "unresolved": [
                "runtime unit not independently reconstructed from measurement command",
                "blocksize columns disagree with kernel-name tile in example rows",
                "TFlops field should not be used without unit verification",
                "dimension 128256 retained exactly as recorded",
            ],
        },
        "public_a100_nccl": {
            "rows": len(comm), "operation_counts": dict(Counter(r["op"] for r in comm)),
            "gpu_counts": sorted({int(r["gpus"]) for r in comm}),
            "columns": list(comm[0]),
            "nonzero_returncodes": sum(int(r["returncode"]) != 0 for r in comm),
            "decision": "bandwidth component reference only; no stored latency or per-row validation counts",
            "normalization": "do not reconstruct exact times from rounded bandwidth or treat one-GPU zero BW as latency",
        },
        "tilebench_b200_matmul": {
            "tables": tilebench,
            "decision": "component methodology/reference; hardware and M range differ from current A100 small-M target",
            "notes": [
                "default and autotune are different conditions, not independent repetitions",
                "recorded config: CUDA graph enabled, L2 flush enabled, warmup 1, repeat 3",
                "recorded tolerance atol 5.0, rtol 0.1 is not our FP16 correctness contract",
            ],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; use a new file")
    result = audit(args.catalog, args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"verified {result['checked_files']} pinned files; public fused training rows admitted: 0")
