#!/bin/bash
# Three seeds per cell and three processes per snapshot, with arm sets fixed before these runs.
cd /srv/edmir-run/exploratory
log() { echo "$(date -u +%H:%M:%S) $*"; }
cells=""; for w in GLM QWEN35 DSV2L; do for m in 0001 0002 0004 0008 0016 0032 0064 0128 0256 0512 1024; do cells="$cells ${w}_M$m"; done; done
snaps="GLM_DECODE QWEN35_DECODE DSV2L_DECODE GLM_PREFILL QWEN35_PREFILL DSV2L_PREFILL"
for i in 1 2 3; do
  log "snapshots repeat $i"
  TAG=-repeat$i ARMS=NATIVE,D_Q1_G2_R,D_Q4_G2_R,D_GATHER16_Q4_G2_R,D_GATHER16_Q8_G2_K64_R,D_GATHER32_Q1_G4_R,BNEW_Q4_G2_R ./run.sh snapshots $snaps 2>&1 | grep "wall ms\|FAILED\|not admitted"
done
for rep in 20261120 20261121 20261122; do
  log "grid seed $rep"
  REP=$rep TAG=-seed$rep ARMS=SLT,D_MRG16_Q4_G2,D_MRG16_Q8_G2,D_ALL32_Q1_G4,PULL_SLT,BNEW_Q4_G2,D_Q4_G2 ./run.sh variants $cells 2>&1 | grep "wall ms\|FAILED\|not admitted"
done
log "REPEATS DONE"
