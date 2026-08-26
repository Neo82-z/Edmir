#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/llm-serving}
NCCL_HOME=${NCCL_HOME:-/root/edmir-deps/nccl-2.28.9/build}
CUTLASS_ROOT=${CUTLASS_ROOT:-/root/cutlass}
OUT_ROOT=${OUT_ROOT:-/root/edmir-fused-heldout}

cd "${REPO}/EDM-IR"
export PYTHONPATH="${REPO}/EDM-IR"

for spec in "32 64" "32 128" "32 256" "64 128" "64 256" "128 64" "128 128" "128 256"; do
  set -- ${spec}
  m=$1
  n=$2
  python tools/run_fused_kernel_sweep.py \
    --runtime runtime \
    --topology examples/full_p2p_topology.json \
    --cutlass "${CUTLASS_ROOT}" \
    --nccl-home "${NCCL_HOME}" \
    --out "${OUT_ROOT}/m${m}-n${n}" \
    --rows-per-peer "${m}" --n "${n}" \
    --k-values 256,512,1024,2048,4096 \
    --warmup 30 --iterations 300 --validate-first
done

python -m edmir evaluate-fused \
  --train "${REPO}/experiments/2026-08-26-a100-fused-k-boundary/results.csv" \
  --heldout "${OUT_ROOT}"/*/results.csv \
  --out "${OUT_ROOT}/evaluation.json"
