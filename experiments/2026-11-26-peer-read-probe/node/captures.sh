#!/bin/bash
R=/srv/edmir-run
log() { echo "$(date -u +%H:%M:%S) $*"; }
run() { log "capture $1 start"; timeout 1500 $R/capture.sh "$1" "$2" "$3" > $R/logs/capture-$1.log 2>&1; log "capture $1 exit=$?"; }
run GLM    /code/experiments/2026-11-16-next-window/model-manifest.json "E=8,N=1536,device_name=NVIDIA_A100-SXM4-80GB.json"
run QWEN35 /cpu-records/manifests/QWEN35.json "E=32,N=512,device_name=NVIDIA_A100-SXM4-80GB.json"
run DSV2L  /cpu-records/manifests/DSV2L.json  "E=8,N=1408,device_name=NVIDIA_A100-SXM4-80GB.json"
log "CAPTURES DONE"
