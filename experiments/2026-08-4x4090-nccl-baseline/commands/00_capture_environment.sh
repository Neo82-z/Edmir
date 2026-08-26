#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RESULT_ROOT" >&2
  exit 2
fi

RESULT_ROOT=$1
ENV_DIR="$RESULT_ROOT/environment"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
mkdir -p "$ENV_DIR"

capture() {
  local output=$1
  shift
  "$@" > "$ENV_DIR/$output" 2>&1 || true
}

capture uname.txt uname -a
capture nvidia-smi.txt nvidia-smi
capture nvidia-smi-list.txt nvidia-smi -L
capture nvidia-smi-query.txt nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,power.limit,clocks.sm,clocks.mem \
  --format=csv
capture nvidia-smi-topo-m.txt nvidia-smi topo -m
capture nvidia-smi-topo-p2p-p.txt nvidia-smi topo -p2p p
capture nvidia-smi-nvlink-status.txt nvidia-smi nvlink --status
capture nvcc-version.txt nvcc --version
capture lspci-tree.txt lspci -tv
capture numa.txt numactl --hardware
capture git-revision.txt git -C "$REPO_ROOT" rev-parse HEAD

P2P_TEST_BIN=${P2P_TEST_BIN:-}
if [[ -n "$P2P_TEST_BIN" && -x "$P2P_TEST_BIN" ]]; then
  capture p2pBandwidthLatencyTest.txt "$P2P_TEST_BIN"
else
  printf '%s\n' \
    'not run: set P2P_TEST_BIN to the CUDA sample p2pBandwidthLatencyTest binary' \
    > "$ENV_DIR/p2pBandwidthLatencyTest.txt"
fi

if [[ -d "$REPO_ROOT/EDM-IR" ]]; then
  (
    cd "$REPO_ROOT/EDM-IR" || exit 1
    python3 -m edmir probe --out "$ENV_DIR/topology.json"
  ) > "$ENV_DIR/edmir-probe.txt" 2>&1 || true
fi

echo "environment captured under $ENV_DIR"

