#!/bin/bash
# Exploratory jobs after the formal queue.
#   run.sh probe
#   run.sh variants <cell>...       (ARMS=a,b,... to restrict, TAG=-x to name the output)
#   run.sh snapshots <snapshot>...  (same options)
source /srv/edmir-run/env.sh
X=$R/exploratory
PROBE="docker run --rm --pull never --network none --gpus all --ipc=host --hostname edmir --add-host edmir:127.0.0.1 \
  -v $R/code:/repo:ro -v $X:/probe -v $R/receipts:/receipts:ro -v $R/work:/work:ro -v $R/capture:/capture:ro \
  -e TRITON_CACHE_DIR=/probe/cache/triton -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/repo/experiments/2026-11-23-replay:/probe \
  -w /repo/experiments/2026-11-23-replay --entrypoint python3 edmir/vllm:0.20.1-locked \
  -m torch.distributed.run --standalone --nproc-per-node=8"
mkdir -p $X/cache/triton $X/variants $X/snapshots
mode=$1; shift
case "$mode" in
  probe) $PROBE /probe/peer_read_probe.py --gate /receipts/gate.json ${MODE:+--mode $MODE} --out /probe/peer-read-probe$TAG.json 2> $X/probe$TAG.err ;;
  variants) for u in "$@"; do
      $PROBE /probe/consumer_variants.py --gate /receipts/gate.json --slt /work/slt.json --library /work/libedmir_replay.so \
        --unit $u ${REP:+--rep $REP} ${ARMS:+--arms $ARMS} --out /probe/variants/$u$TAG.json 2> $X/variants/$u$TAG.err || echo "$u FAILED (variants/$u$TAG.err)"; done ;;
  snapshots) for u in "$@"; do
      $PROBE /probe/snapshot_variants.py --gate /receipts/gate.json --run /work/run $SN \
        --unit $u ${ARMS:+--arms $ARMS} --out /probe/snapshots/$u$TAG.json 2> $X/snapshots/$u$TAG.err || echo "$u FAILED (snapshots/$u$TAG.err)"; done ;;
esac
