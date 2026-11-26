#!/usr/bin/env python3
"""CPU-only check of everything the run needs before a GPU job starts. Changes nothing.

Run it in the locked image on the node, after the hardware gate:

  python3 preflight.py --library /work/libedmir_replay.so \\
      --cpu-receipt /cpu-records/<dir>/cpu-receipt.json --out /work/preflight.json

Each check is PASS, FAIL or SKIP with a reason. Any FAIL exits 2: fix it before the smoke.
`--local` skips the checks that need the locked image (runtime versions, library load).
"""
import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from make_design import DESIGN_ID, ROOT, sha

REPO = ROOT.parents[1]
ARCHIVE = ROOT.parent/"2026-11-20-final-round"
RUNTIME = {"torch":"2.11.0+cu130","vllm":"0.20.1"}
CAPTURE_INPUTS = ("experiments/2026-11-17-real-moe-local/corpus/requests-source.json",
                  "experiments/2026-11-16-next-window/model-manifest.json",
                  "experiments/2026-11-17-real-moe-local/vllm-source-lock.json")
ARCHIVED = ("scripts/predict.py","scripts/calibrate.py","scripts/common.py","scripts/hw_gate.py","params/scenario.json")
BRIDGE_SOURCES = ("EDM-IR/runtime/src/cutlass_gemm.cu","EDM-IR/runtime/src/cutlass_sm80_gemm.cuh")


def design():
    d = json.loads((ROOT/"design"/"DESIGN.json").read_text())
    changed = [name for name,h in d["files"].items() if sha(ROOT/"design"/name)!=h]
    if d["design_id"]!=DESIGN_ID or changed:
        raise ValueError(f"design tables differ from DESIGN.json: {changed or d['design_id']}")
    return d["design_id"]


def prediction():
    from analyze import load_prediction, predicted_classes
    rows,choices = load_prediction(ROOT/"preboot_prediction",ROOT/"design","balanced")
    return f"{len(rows)} crossover rows, {len(choices)} choices, {len(predicted_classes(ROOT/'preboot_prediction'))} cell classes"


def files():
    missing = [str(p) for p in [ARCHIVE/x for x in ARCHIVED]+[REPO/x for x in CAPTURE_INPUTS+BRIDGE_SOURCES]
               if not p.is_file()]
    if missing:
        raise ValueError("missing: "+", ".join(missing))
    return f"{len(ARCHIVED)+len(CAPTURE_INPUTS)+len(BRIDGE_SOURCES)} files present"


def collector(receipt_path):
    """The capture refuses a CPU receipt made from other collector sources; find that out now."""
    sys.path.insert(0,str(REPO/"EDM-IR"))
    from edmir.benchmark.real_moe_config import collection_files
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema")!="edmir.real-moe-cpu-receipt.v1" or receipt.get("status")!="passed":
        raise ValueError("CPU receipt is not a passed edmir.real-moe-cpu-receipt.v1")
    root = REPO/"EDM-IR"
    actual = {str(f.relative_to(root)):sha(f) for f in collection_files(root)}
    stale = sorted(k for k in actual if receipt["collection_source_hashes"].get(k)!=actual[k])
    if stale or set(actual)!=set(receipt["collection_source_hashes"]):
        raise ValueError("CPU receipt was made from different collector sources: "+", ".join(stale or ["file set"]))
    return f"{receipt.get('model','GLM (no model field)')}: {len(actual)} collector files match the CPU receipt"


def runtime():
    found = {p:importlib.metadata.version(p) for p in RUNTIME}
    found["triton"] = importlib.metadata.version("triton")
    wrong = {p:v for p,v in found.items() if p in RUNTIME and v!=RUNTIME[p]}
    if wrong:
        raise ValueError(f"locked image expected {RUNTIME}, found {wrong}")
    return found


def library(path):
    """Built with CUDA 12 it links libcublasLt.so.12, which the CUDA 13 image cannot load."""
    listing = subprocess.run(["ldd",str(path)],capture_output=True,text=True,timeout=30)
    lines = listing.stdout.splitlines()
    lt = [line.strip() for line in lines if "libcublasLt" in line]
    unresolved = [line.strip() for line in lines if "not found" in line]
    if listing.returncode or not lt or unresolved or not any("libcublasLt.so.13" in line for line in lt):
        raise ValueError(f"library must resolve libcublasLt.so.13: {lt or listing.stderr.strip()} {unresolved}")
    from balanced import load_library
    load_library(path)
    return dict(sha256=sha(path),cublasLt=lt[0])


def cache():
    folder = os.environ.get("TRITON_CACHE_DIR")
    if not folder:
        raise ValueError("TRITON_CACHE_DIR is unset; kernels would recompile in every process")
    Path(folder).mkdir(parents=True,exist_ok=True)
    probe = Path(folder)/".preflight"
    probe.write_text("ok")
    probe.unlink()
    return folder


def commit():
    stamp = REPO / "EDMIR_COMMIT"
    if not (REPO / ".git").exists() and stamp.is_file():
        revision = stamp.read_text().strip()
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("invalid commit stamp in the code archive")
        checked = set()
        manifest = REPO / "EDMIR_CODE.sha256"
        for line in manifest.read_text().splitlines():
            digest, separator, name = line.partition("  ")
            path = Path(name)
            if not separator or not name or path.is_absolute() or ".." in path.parts or path in checked:
                raise ValueError(f"invalid code archive entry: {name}")
            if sha(REPO / path) != digest:
                raise ValueError(f"code archive checksum mismatch: {name}")
            checked.add(path)
        if Path(stamp.name) not in checked:
            raise ValueError("code archive manifest does not cover the commit stamp")
        return revision
    if not shutil.which("git"):
        raise LookupError("git unavailable here; record the checked-out commit by hand")
    out = subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True,timeout=30)
    if out.returncode:
        raise LookupError("git unavailable here; record the checked-out commit by hand")
    dirty = subprocess.run(["git","-C",str(REPO),"status","--porcelain","--",str(ROOT),"EDM-IR/edmir","EDM-IR/tools"],
                           capture_output=True,text=True,timeout=30).stdout.strip()
    if dirty:
        raise ValueError("uncommitted changes in the package or the collector:\n"+dirty)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library",type=Path)
    ap.add_argument("--cpu-receipt",type=Path,action="append",help="one per model")
    ap.add_argument("--local",action="store_true")
    ap.add_argument("--out",type=Path)
    a = ap.parse_args()
    checks = [("design",design),("preboot_prediction",prediction),("files",files),("commit",commit)]
    for receipt in a.cpu_receipt or [None]:
        checks.append((f"cpu_receipt[{receipt}]",(lambda r=receipt:collector(r)) if receipt else None))
    checks.append(("runtime",None if a.local else runtime))
    checks.append(("library",(lambda:library(a.library)) if a.library and not a.local else None))
    checks.append(("triton_cache",None if a.local else cache))
    report = {}
    for name,fn in checks:
        if fn is None:
            report[name] = dict(status="SKIP",detail="not requested")
            continue
        try:
            report[name] = dict(status="PASS",detail=fn())
        except LookupError as exc:
            report[name] = dict(status="SKIP",detail=str(exc))
        except Exception as exc:
            report[name] = dict(status="FAIL",detail=repr(exc))
    report["package_sha256"] = {p.name:sha(p) for p in sorted(ROOT.glob("*.py"))}
    for name,r in report.items():
        if isinstance(r,dict) and "status" in r:
            print(f"{r['status']:4} {name}: {r['detail']}")
    if a.out:
        a.out.write_text(json.dumps(report,indent=2)+"\n")
    return 2 if any(isinstance(r,dict) and r.get("status")=="FAIL" for r in report.values()) else 0


if __name__=="__main__":
    sys.exit(main())
