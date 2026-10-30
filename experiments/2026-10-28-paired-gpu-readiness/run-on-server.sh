#!/usr/bin/env bash
set -euo pipefail

PAIR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR_PYTHON="${PAIR_PYTHON:-python3}"
profile="${1:?usage: bash run-on-server.sh p4|p8-grouped verify|inventory|build|preflight|run|resume|pack|trace [job-index]}"
action="${2:?missing action}"
case "$profile" in
  p4) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" ;;
  p8-grouped) export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" ;;
  *) exit 2 ;;
esac
package="$PAIR_ROOT/$profile"
work="$PAIR_ROOT/work/$profile"
runner="$package/source/EDM-IR/tools/run_pair_diagnosis_gpu.py"
preparer="$package/source/EDM-IR/tools/prepare_pair_diagnosis_gpu.py"
nccl="$PAIR_ROOT/dependencies/nccl"
mkdir -p "$work"

copy_build_evidence() {
  if [[ -d "$work/results" ]]; then
    for name in selector_build.json build.stdout.txt build.stderr.txt; do
      if [[ -f "$work/build/$name" ]]; then cp -- "$work/build/$name" "$work/results/$name"; fi
    done
  fi
}

pack_results() {
  copy_build_evidence
  "$PAIR_PYTHON" "$runner" pack --out "$work/results" --archive "$work/recovery-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
}

case "$action" in
  verify) "$PAIR_PYTHON" "$preparer" verify --package "$package" ;;
  inventory) "$PAIR_PYTHON" "$runner" inventory --out "$work/inventory.json" ;;
  build)
    "$PAIR_PYTHON" "$runner" build --package "$package" --build-dir "$work/build" \
      --cutlass "$PAIR_ROOT/dependencies/cutlass" --nccl-source "$nccl" \
      --nccl-include "$nccl/build/include" --nccl-library "$nccl/build/lib/libnccl.so" --jobs "${PAIR_BUILD_JOBS:-4}"
    ;;
  preflight) "$PAIR_PYTHON" "$runner" preflight --package "$package" --build-dir "$work/build" --out "$work/preflight.json" ;;
  run|resume)
    extra=()
    if [[ "$action" == resume ]]; then extra=(--resume); fi
    set +e
    "$PAIR_PYTHON" "$runner" run --package "$package" --build-dir "$work/build" --out "$work/results" "${extra[@]}"
    status=$?
    set -e
    if [[ -d "$work/results" ]]; then pack_results; fi
    exit "$status"
    ;;
  pack) pack_results ;;
  trace) "$PAIR_PYTHON" "$runner" trace --package "$package" --build-dir "$work/build" --out "$work/trace-${3:?missing job index}" --job-index "$3" ;;
  *) exit 2 ;;
esac
