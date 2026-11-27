#!/bin/bash
# Exploratory jobs on the GPU node. Scripts come from the deployed code; results go to
# /srv/edmir-run/exploratory.
#   run_exploratory.sh probe                       peer-read patterns (MODE=sizes for the size sweep)
#   run_exploratory.sh variants <cell>...          balanced grid cells, e.g. GLM_M0001
#   run_exploratory.sh snapshots <snapshot>...     registered snapshots, e.g. GLM_DECODE
#   run_exploratory.sh layers <model> <layer> <snapshot>...   exploratory captures of another layer
# Options by environment: ARMS=a,b,...  TAG=-name  REP=<seed> (grid)  ORIENT=forward|reverse|both
source /srv/edmir-run/env.sh
X=$R/exploratory; P=/repo/experiments/2026-11-26-peer-read-probe
mkdir -p $X/cache/triton $X/variants $X/snapshots $X/layers $R/capture-layers
PROBE="docker run --rm --pull never --network none --gpus all --ipc=host --hostname edmir --add-host edmir:127.0.0.1 \
  -v $R/code:/repo:ro -v $X:/probe -v $R/receipts:/receipts:ro -v $R/work:/work:ro -v $R/capture:/capture:ro \
  -v $R/capture-layers:/layers:ro \
  -e TRITON_CACHE_DIR=/probe/cache/triton -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/repo/experiments/2026-11-23-replay:$P \
  -w /repo/experiments/2026-11-23-replay --entrypoint python3 edmir/vllm:0.20.1-locked \
  -m torch.distributed.run --standalone --nproc-per-node=8"
COMMON="--gate /receipts/gate.json ${ARMS:+--arms $ARMS} ${ORIENT:+--orientation $ORIENT}"
mode=$1; shift
case "$mode" in
  probe)
    $PROBE $P/peer_read_probe.py --gate /receipts/gate.json ${MODE:+--mode $MODE} \
      --out /probe/peer-read-probe$TAG.json 2> $X/probe$TAG.err ;;
  variants)
    for u in "$@"; do
      $PROBE $P/consumer_variants.py $COMMON --slt /work/slt.json --library /work/libedmir_replay.so \
        --unit $u ${REP:+--rep $REP} --out /probe/variants/$u$TAG.json 2> $X/variants/$u$TAG.err \
        || echo "$u FAILED (variants/$u$TAG.err)"
    done ;;
  snapshots)
    for u in "$@"; do
      $PROBE $P/snapshot_variants.py $COMMON --run /work/run $SN --unit $u \
        --out /probe/snapshots/$u$TAG.json 2> $X/snapshots/$u$TAG.err || echo "$u FAILED (snapshots/$u$TAG.err)"
    done ;;
  layers)
    model=$1; layer=$2; shift 2
    for u in "$@"; do
      $PROBE $P/snapshot_variants.py $COMMON --layer $layer --snapshots /layers/$model-L$layer/workers/snapshots \
        --unit $u --out /probe/layers/$u-L$layer$TAG.json 2> $X/layers/$u-L$layer$TAG.err \
        || echo "$u layer $layer FAILED (layers/$u-L$layer$TAG.err)"
    done ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
