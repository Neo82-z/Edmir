#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_name="resume-$stamp"
exec > >(tee "session/$run_name.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "session/$run_name.exit-code"; date -u +%FT%TZ > "session/$run_name.finished-utc"' EXIT
date -u +%FT%TZ > "session/$run_name.started-utc"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:/opt/edmir-nsys/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
python3 /workspace/resume-window.py --dry-run
python3 /workspace/resume-window.py
