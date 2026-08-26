#!/usr/bin/env bash
set -euo pipefail

# Run inside the copied repository on the measured 4x A100 server. Keep NCCL
# isolated so the system 2.25 installation is never overwritten.
REPO=${REPO:-/root/llm-serving}
NCCL_HOME=${NCCL_HOME:-/root/edmir-deps/nccl-2.28.9/build}
CUTLASS_ROOT=${CUTLASS_ROOT:-/root/cutlass}
ARTIFACT=${ARTIFACT:-/root/edmir-fused-cutlass-artifact}
BUILD=${BUILD:-/root/edmir-fused-cutlass-build}
PROBLEM=${PROBLEM:-${REPO}/experiments/2026-08-26-a100-nccl-device-cutlass-fusion/problems/n0064.json}
KERNEL=${KERNEL:-sm80_64x128x32_splitk8}

cd "${REPO}/EDM-IR"
python -m edmir frontend "${PROBLEM}" --out /tmp/edmir-fused.dataflow.json
python -m edmir lower-fused /tmp/edmir-fused.dataflow.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 64 --arch sm_80 --gemm-provider cutlass \
  --gemm-kernel "${KERNEL}" --backend cpp --out-dir "${ARTIFACT}"

cmake -S runtime -B "${BUILD}" \
  -DEDMIR_ENABLE_CUDA=ON \
  -DEDMIR_ENABLE_CUTLASS=ON \
  -DEDMIR_ENABLE_NCCL_DEVICE=ON \
  -DEDMIR_GENERATED_DIR="${ARTIFACT}" \
  -DEDMIR_CUTLASS_DIR="${CUTLASS_ROOT}" \
  -DNCCL_INCLUDE_DIR="${NCCL_HOME}/include" \
  -DNCCL_LIBRARY="${NCCL_HOME}/lib/libnccl.so" \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build "${BUILD}" -j"$(nproc)"

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir "${BUILD}" --output-on-failure
"${BUILD}/edmir_validate_generated"
"${BUILD}/edmir_benchmark_generated" --warmup 30 --iterations 300

# Nsight is intentionally a short boundary check, not the timed result above.
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
  -o /tmp/edmir-fused-cutlass \
  "${BUILD}/edmir_benchmark_generated" --warmup 3 --iterations 1
