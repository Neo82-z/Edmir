#!/bin/bash
source /srv/edmir-run/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "time-all start"
$RUN replay.py --run /work/run --library /work/libedmir_replay.so $SN time-all >> $R/logs/time-all.log 2>&1; log "time-all exit=$?"
log "TIMING DONE"
