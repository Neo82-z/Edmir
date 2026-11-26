#!/bin/bash
source /srv/edmir-run/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "microbench"; timeout 1500 $RUN $T microbench --gate /receipts/gate.json --out /work/microbench.json > $R/logs/microbench.log 2>&1; log "microbench exit=$?"
log "calibrate"; $RUN calibrate_node.py --microbench /work/microbench.json --out /work/calibrated > $R/logs/calibrate.log 2>&1; log "calibrate exit=$?"
log "tune-slt"; timeout 2400 $RUN $T tune-slt --gate /receipts/gate.json --out /work/slt.json > $R/logs/slt.log 2>&1; log "tune-slt exit=$?"
log "PRE DONE"
