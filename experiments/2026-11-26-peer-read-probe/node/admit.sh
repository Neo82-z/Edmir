#!/bin/bash
source /srv/edmir-run/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "admit start"
$RUN replay.py --run /work/run --library /work/libedmir_replay.so $SN admit > $R/logs/admit.log 2>&1; log "admit exit=$?"
log "ADMIT DONE"
