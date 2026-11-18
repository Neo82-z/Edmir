#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
export PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R=/workspace/bundle/source/EDM-IR/tools/run_w1_gpu.py
CUTOFF=$(date -u -d '2026-11-11 08:17:00 UTC' +%s)
exec > >(tee -a session/resume-and-trace.log) 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > session/resume-exit-code; date -u +%FT%TZ > session/resume-finished-utc' EXIT
date -u +%FT%TZ > session/resume-started-utc
python3 "$R" collect --bundle /workspace/bundle --inputs /workspace/inputs --build-dir /workspace/build --out /workspace/results --cutoff-epoch "$CUTOFF"
export PATH=/opt/edmir-nsys/extracted/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
for trace_case in grouped-g1 grouped-g2 m512-serial m512-pipeline; do
  if (( $(date +%s) >= CUTOFF - 30 )); then
    printf '%s\t%s\n' "$trace_case" not_started_time_budget >> session/trace-status.tsv
    continue
  fi
  if python3 "$R" trace --trace-case "$trace_case" --bundle /workspace/bundle --inputs /workspace/inputs --build-dir /workspace/build --out /workspace/results --cutoff-epoch "$CUTOFF"; then
    printf '%s\t%s\n' "$trace_case" captured >> session/trace-status.tsv
  else
    printf '%s\t%s\n' "$trace_case" failed_see_preserved_logs >> session/trace-status.tsv
  fi
done
nvidia-smi -q > session/nvidia-smi-after.txt
printf '%s\n' finished > session/collection-status
