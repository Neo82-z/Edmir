#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/sendrecv_perf /path/to/output-directory" >&2
  exit 2
fi

sendrecv_perf=$1
output=$2
if [[ ! -x "$sendrecv_perf" ]]; then
  echo "not executable: $sendrecv_perf" >&2
  exit 2
fi
mkdir -p "$output"

for pair in 0,1 0,2 0,3 1,2 1,3 2,3; do
  src=${pair%,*}
  dst=${pair#*,}
  log="$output/pair_${src}_${dst}_default_sendrecv.txt"
  echo "pair=$src,$dst log=$log"
  CUDA_VISIBLE_DEVICES="$pair" NCCL_DEBUG=WARN \
    "$sendrecv_perf" -b 1K -e 512M -f 2 -g 2 -w 5 -n 20 \
    2>&1 | tee "$log"
done
