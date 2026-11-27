#!/bin/bash
# setup.sh  — the machine-bound steps on a new node, in the order of START_HERE.md: host gate,
# runtime gate, read-only self-check, smoke, native tuning of the three models, microbenchmark,
# calibrated prediction and SLT tuning. Stops at the first gate that does not pass.
# Needs bootstrap.sh core and push_code.sh first. Logs go to /srv/edmir-run/logs.
source /srv/edmir-run/env.sh
E=/repo/experiments/2026-11-23-replay
log() { echo "$(date -u +%H:%M:%S) $*"; }
python3 $R/code/experiments/2026-11-23-replay/hw_gate.py host --out $R/receipts/host.json > $R/logs/gate-host.log 2>&1
log "host gate exit=$? $(grep -o '"verdict": "[A-Z]*"' $R/receipts/host.json)"
docker run --rm --pull never --network none --gpus all -v $R/code:/repo:ro -v $R/receipts:/receipts \
  --entrypoint python3 edmir/vllm:0.20.1-locked $E/hw_gate.py runtime \
  --host-receipt /receipts/host.json --out /receipts/gate.json > $R/logs/gate-runtime.log 2>&1
log "runtime gate exit=$? $(grep -o '"verdict": "[A-Z]*"' $R/receipts/gate.json | head -1)"
grep -q '"verdict": "PASS"' $R/receipts/gate.json || { log "gate did not pass; see logs/gate-runtime.log"; exit 1; }
$RUN preflight.py --library /work/libedmir_replay.so --cpu-receipt /cpu-records/GLM/cpu-receipt.json \
  --cpu-receipt /cpu-records/QWEN35/cpu-receipt.json --cpu-receipt /cpu-records/DSV2L/cpu-receipt.json \
  --out /work/preflight.json > $R/logs/preflight.log 2>&1
log "preflight exit=$? $(grep -c FAIL $R/logs/preflight.log) FAIL lines"
$R/smoke.sh > $R/logs/smoke-summary.log 2>&1; log "smoke: $(grep -c '^FAIL' $R/logs/smoke-summary.log) failed arms"
for w in GLM QWEN35 DSV2L; do timeout 900 $RUN native.py --out /work/moe_configs --width $w; done > $R/logs/native.log 2>&1
log "native tuning exit=$? $(ls $R/work/moe_configs | grep -c '^E=') configs"
$R/pre.sh
log "SETUP DONE"
