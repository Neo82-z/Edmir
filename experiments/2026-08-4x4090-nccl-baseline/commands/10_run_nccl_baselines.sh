#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RESULT_ROOT" >&2
  exit 2
fi

RESULT_ROOT=$1
NCCL_DIR="$RESULT_ROOT/nccl"
TOPO_DIR="$RESULT_ROOT/environment/nccl-topology"
NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-/root/edmir-bench/nccl-tests}
BUILD_DIR="$NCCL_TESTS_DIR/build"
MIN_BYTES=${MIN_BYTES:-1K}
MAX_BYTES=${MAX_BYTES:-512M}
WARMUP_ITERS=${WARMUP_ITERS:-5}
TEST_ITERS=${TEST_ITERS:-20}

mkdir -p "$NCCL_DIR" "$TOPO_DIR"
MANIFEST="$RESULT_ROOT/manifest.tsv"
printf 'tag\tdevices\tngpus\tmode\tstatus\n' > "$MANIFEST"

for binary in sendrecv_perf alltoall_perf all_gather_perf all_reduce_perf; do
  if [[ ! -x "$BUILD_DIR/$binary" ]]; then
    echo "missing executable: $BUILD_DIR/$binary" >&2
    exit 2
  fi
done

run_perf() {
  local tag=$1
  local devices=$2
  local ngpus=$3
  local mode=$4
  local binary=$5
  local log="$NCCL_DIR/${tag}.txt"
  local topo="$TOPO_DIR/${tag}.xml"
  local env_args=(
    "CUDA_VISIBLE_DEVICES=$devices"
    "NCCL_DEBUG=INFO"
    "NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING"
    "NCCL_TOPO_DUMP_FILE=$topo"
  )
  if [[ "$mode" == "no_p2p" ]]; then
    env_args+=("NCCL_P2P_DISABLE=1")
  fi

  echo "running $tag devices=$devices mode=$mode"
  set +e
  env "${env_args[@]}" "$BUILD_DIR/$binary" \
    -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 \
    -g "$ngpus" -w "$WARMUP_ITERS" -n "$TEST_ITERS" \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$devices" "$ngpus" "$mode" "$status" >> "$MANIFEST"
}

pairs=("0,1" "0,2" "0,3" "1,2" "1,3" "2,3")
for mode in default no_p2p; do
  for devices in "${pairs[@]}"; do
    pair_name=${devices/,/_}
    run_perf "pair_${pair_name}_${mode}_sendrecv" "$devices" 2 "$mode" sendrecv_perf
  done
done

for mode in default no_p2p; do
  run_perf "all4_${mode}_alltoall" "0,1,2,3" 4 "$mode" alltoall_perf
  run_perf "all4_${mode}_all_gather" "0,1,2,3" 4 "$mode" all_gather_perf
  run_perf "all4_${mode}_all_reduce" "0,1,2,3" 4 "$mode" all_reduce_perf
done

echo "NCCL baselines complete; inspect $MANIFEST"
