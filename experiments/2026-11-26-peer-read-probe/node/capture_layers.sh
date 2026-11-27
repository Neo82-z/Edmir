#!/bin/bash
# capture_layers.sh  — the six exploratory captures of other layers, one after the other, then
# the comparisons on them. Needs bootstrap.sh models and the native tuning of setup.sh.
# The registered layers are GLM 23, Qwen3.5 20 and DSV2-Lite 13; these are an early and a late one.
#   ARMS=... ./capture_layers.sh
R=/srv/edmir-run
log() { echo "$(date -u +%H:%M:%S) $*"; }
run() {  # run <model> <manifest> <tuned config> <layer>
  [ -d $R/capture-layers/$1-L$4/workers/snapshots ] && { log "capture $1 layer $4 present"; return; }
  log "capture $1 layer $4 start"
  timeout 1500 $R/capture_layer.sh "$1" "$2" "$3" "$4" > $R/logs/capture-$1-L$4.log 2>&1; log "capture $1 layer $4 exit=$?"
}
for layer in 3 44; do run GLM /code/experiments/2026-11-16-next-window/model-manifest.json "E=8,N=1536,device_name=NVIDIA_A100-SXM4-80GB.json" $layer; done
for layer in 3 37; do run QWEN35 /cpu-records/manifests/QWEN35.json "E=32,N=512,device_name=NVIDIA_A100-SXM4-80GB.json" $layer; done
for layer in 3 24; do run DSV2L /cpu-records/manifests/DSV2L.json "E=8,N=1408,device_name=NVIDIA_A100-SXM4-80GB.json" $layer; done
log "CAPTURES DONE"
for spec in "GLM 3" "GLM 44" "QWEN35 3" "QWEN35 37" "DSV2L 3" "DSV2L 24"; do
  set -- $spec
  $R/run_exploratory.sh layers $1 $2 ${1}_DECODE ${1}_PREFILL
done
log "LAYERS DONE"
