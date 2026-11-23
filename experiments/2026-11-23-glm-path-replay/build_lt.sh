#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "$0")" && pwd)"
runtime="$here/../../EDM-IR/runtime"
: "${CUTLASS_ROOT:?Point to an existing pinned CUTLASS checkout}"
out="${1:?Specify the output .so path}"
nvcc -std=c++20 -O3 -arch=sm_80 -shared -Xcompiler=-fPIC \
  -I"$CUTLASS_ROOT/include" -I"$runtime/include" \
  "$here/lt_bridge.cu" "$runtime/src/cutlass_gemm.cu" -lcublasLt -o "$out"
