#!/usr/bin/env python3
"""Compile the sealed Linux/SM80 package locally. Never requests a GPU."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BUNDLE = REPO / "experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2"
ASSETS = Path("/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01")
CONTEXT = "colima-edmir-build"
IMAGE = "edmir-cuda128-cpu-build:20261114"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True)
    args = p.parse_args()
    if not args.attempt.replace("-", "").replace("_", "").isalnum():
        p.error("attempt must be alphanumeric, hyphens or underscores")
    output = ROOT / args.attempt
    output.mkdir(exist_ok=False)
    command = ["docker", "--context", CONTEXT, "run", "--rm", "--platform", "linux/amd64",
               "--network=none", "--cpus=6", "--memory=10g",
               "--mount", f"type=bind,src={output},dst=/output",
               "--mount", f"type=bind,src={ROOT / 'build-native.sh'},dst=/build-native.sh,readonly",
               "--mount", f"type=bind,src={BUNDLE},dst=/workspace/precision-bundle,readonly",
               "--mount", f"type=bind,src={ASSETS / 'dependencies'},dst=/workspace/dependencies,readonly",
               "--mount", f"type=bind,src={ASSETS / 'inputs'},dst=/workspace/inputs,readonly",
               IMAGE, "bash", "/build-native.sh", "/workspace/precision-bundle/source/EDM-IR/tools/run_w1_gpu.py", "build",
               "--bundle", "/workspace/precision-bundle", "--inputs", "/workspace/inputs",
               "--dependencies", "/workspace/dependencies", "--build-dir", "/workspace/build-precision",
               "--jobs", "4", "--cutoff-epoch", str(time.time() + 3500)]
    receipt = {"kind": "local_cpu_cross_arch_build", "native_build_filesystem": True, "gpu_requested": False,
               "gpu_validation": "NOT_RUN", "started_epoch": time.time(), "command": command}
    (output / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "console.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(exit_code=result.returncode, finished_epoch=time.time())
    (output / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"attempt": str(output), "exit_code": result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
