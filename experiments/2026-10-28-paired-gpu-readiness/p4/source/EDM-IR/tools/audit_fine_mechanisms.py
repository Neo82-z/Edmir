#!/usr/bin/env python3
"""Bind finer execution expressions to the original binary and CUDA source."""
import json
from pathlib import Path
import re

from clean_mechanism_data import FROZEN_PACKAGE, read, sha
from edmir.compiler.mechanism_prediction import detailed_geometry, geometry, matrix_rank


def audit_resources(root, configs):
    root = Path(root)
    archive = root/"server-backup"
    directory = archive/"work/compiled-resources-02"
    file_manifest = read(archive/"result-file-manifest.json")
    audited_files = {}
    for name in ("resources-demangled.txt", "kernel_type_sizes.json", "receipt.json"):
        path = directory/name
        actual = sha(path)
        if actual != file_manifest[str(path.relative_to(archive))]["sha256"]:
            raise ValueError("static resource evidence changed: "+name)
        audited_files[str(path.relative_to(root))] = actual
    receipt = read(directory/"receipt.json")
    binary = archive/"work/build"/Path(receipt["benchmark_binary"]).name
    if sha(binary) != receipt["benchmark_sha256"]:
        raise ValueError("resource dump benchmark binary differs")
    package = read(root/"revision-01/package.json")
    if package["package_sha256"] != FROZEN_PACKAGE:
        raise ValueError("unknown measured source identity")
    source_name = "EDM-IR/runtime/src/nccl_device_fused_cutlass.cu"
    source = root.parent/"bundle/source"/source_name
    source_hash = sha(source)
    if source_hash != package["package"]["sources"][source_name]:
        raise ValueError("frozen CUDA source differs")
    launch = source.read_text()
    required = ("if (shared_bytes >= (48 << 10))", "cudaFuncSetAttribute(",
                "cudaFuncAttributeMaxDynamicSharedMemorySize", "return cudaGetLastError();")
    if any(text not in launch for text in required):
        raise ValueError("reviewed per-launch API path changed")
    lines = (directory/"resources-demangled.txt").read_text().splitlines()
    kernels = {}
    for i, line in enumerate(lines[:-1]):
        if " Function " not in line or "lsa_cutlass_gemm" not in line or "3, 8, 8, true" not in line:
            continue
        tile = tuple(map(int, re.search(r"GemmShape<(\d+), (\d+), (\d+)>", line).groups()))
        if tile not in ((64,128,32), (64,256,32)):
            continue
        fields = {k:int(v) for k,v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)",lines[i+1])}
        if tile in kernels:
            raise ValueError("ambiguous serial split-K resource symbol")
        kernels[tile] = {"dump_line": i+1, "compiled_resources": fields}
    sizes = {r["name"]:r for r in read(directory/"kernel_type_sizes.json")["types"]}
    records = []
    for g, registers, threads, shared in ((1,140,128,36864),(2,134,256,61440)):
        tile = (64,128*g,32)
        size = sizes[f"g{g}_serial_splitk"]
        if (kernels[tile]["compiled_resources"]["REG"] != registers
                or size["threads_per_cta"] != threads or size["shared_storage_bytes"] != shared):
            raise ValueError("encoded resource constants no longer match the compiled evidence")
        records.append({"g":g,"tile":tile,**kernels[tile],"type_sizes":size,
            "static_SHARED_zero_note":"cuobjdump SHARED:0 excludes dynamic launch allocation; type sizes are 36/60 KiB."})
    accounts = []
    for row in configs:
        if not row["primary_scope"] or row["family"].startswith("host"):
            continue
        accounts.append({"config_id":row["config_id"],"shape":row["shape"],"family":row["family"],
                         "counts":geometry(row["shape"],row["family"]),
                         "detail":detailed_geometry(row["shape"],row["family"])})
    # All of these would be aliases of a single g indicator plus an intercept.
    rows = [[1., a["detail"]["registers_per_thread"], a["counts"]["shared_storage_bytes"],
             a["detail"]["warps_per_CTA"], a["detail"]["dynamic_smem_optin_calls_per_execution"]]
            for a in accounts]
    return {"schema":"edmir.fine-mechanism-audit.v1", "package_sha256":FROZEN_PACKAGE,
        "audited_file_sha256":audited_files,"cuda_source_sha256":source_hash,
        "cuda_source_file":str(source.resolve()),"cuda_source_line":80,
        "benchmark_sha256":receipt["benchmark_sha256"],"kernels":records,"accounts":accounts,
        "g_only_resource_design": {"columns":["intercept","registers","shared","warps","optins"],
            "rank":matrix_rank(rows),"column_count":5,
            "interpretation":"Only two kernel families. Separate physical coefficients are unidentifiable; one opt-in-associated term is used, with confounding retained."},
        "resource_round_values":sorted({a["detail"]["resource_capacity_round_lower_bound"] for a in accounts}),
        "official_hardware_reference": "https://docs.nvidia.com/cuda/ampere-tuning-guide/#occupancy",
        "official_dynamic_shared_reference": "https://docs.nvidia.com/cuda/ampere-tuning-guide/#unified-shared-memory-l1-texture-cache",
        "limitations":["Static resource bounds are not achieved occupancy.",
            "No profiler exists for g2 here; API call counts are source-derived, not API durations.",
            "Largest 164 KiB carveout assumed only for an optimistic bound; actual carveout is unknown.",
            "No hardware performance counters, semaphore durations or physical traffic measurements added."]}
