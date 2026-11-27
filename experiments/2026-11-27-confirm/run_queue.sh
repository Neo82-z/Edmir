#!/bin/bash
# Node-side runner of the confirmatory queue. Resumable: a process with an output file is skipped.
#   run_queue.sh            run every process of queue.csv in order
# Expects /srv/edmir-run/env.sh, the code under /srv/edmir-run/code, receipts, work/slt.json,
# work/libedmir_replay.so, the formal RUN.json under work/run, and the captures under capture/.
source /srv/edmir-run/env.sh
C=$R/confirm; P=/repo/experiments/2026-11-26-peer-read-probe; Q=$R/code/experiments/2026-11-27-confirm
mkdir -p $C/out $C/cache/triton
arms() { python3 -c "import json,sys;print(','.join(json.load(open('$Q/arms.json'))[sys.argv[1]]['arms']))" $1; }
GRID=$(arms grid); SNAP=$(arms snapshot)
DOCKER="docker run --rm --pull never --network none --gpus all --ipc=host --hostname edmir --add-host edmir:127.0.0.1 \
  -v $R/code:/repo:ro -v $C:/confirm -v $R/receipts:/receipts:ro -v $R/work:/work:ro -v $R/capture:/capture:ro \
  -e TRITON_CACHE_DIR=/confirm/cache/triton -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/repo/experiments/2026-11-23-replay:$P \
  -w /repo/experiments/2026-11-23-replay --entrypoint python3 edmir/vllm:0.20.1-locked \
  -m torch.distributed.run --standalone --nproc-per-node=8"
log() { echo "$(date -u +%H:%M:%S) $*"; }
tail -n +2 $Q/queue.csv | while IFS=, read process stage unit pair rep orientation; do
  out=$C/out/$process-$unit-p$pair-$orientation.json
  [ -f $out ] && continue
  if [ $stage = grid ]; then
    timeout 600 $DOCKER $P/consumer_variants.py --gate /receipts/gate.json --slt /work/slt.json --library /work/libedmir_replay.so \
      --unit $unit --rep $rep --arms $GRID --orientation $orientation --out /confirm/out/$(basename $out) \
      > $C/out/$process.log 2>&1 < /dev/null
  else
    timeout 600 $DOCKER $P/snapshot_variants.py --gate /receipts/gate.json --run /work/run $SN \
      --unit $unit --arms $SNAP --orientation $orientation --out /confirm/out/$(basename $out) \
      > $C/out/$process.log 2>&1 < /dev/null
  fi
  log "$process $unit pair $pair $orientation exit=$? $(grep -c 'wall ms' $C/out/$process.log) lines"
done
log "CONFIRM DONE"
