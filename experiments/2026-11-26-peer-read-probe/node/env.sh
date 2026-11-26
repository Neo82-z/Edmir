# Shared container templates for the 2026-11-26 run (see START_HERE.md, "容器内运行").
A=/srv/edmir-assets; R=/srv/edmir-run
RUN="docker run --rm --pull never --network none --gpus all --ipc=host --hostname edmir --add-host edmir:127.0.0.1 \
  -v $R/code:/repo:ro -v $R/work:/work -v $R/capture:/capture:ro \
  -v $A/cpu-records:/cpu-records:ro -v $R/receipts:/receipts \
  -e TRITON_CACHE_DIR=/work/cache/triton -e PYTHONUNBUFFERED=1 \
  -w /repo/experiments/2026-11-23-replay --entrypoint python3 edmir/vllm:0.20.1-locked"
T="-m torch.distributed.run --standalone --nproc-per-node=8 replay.py --library /work/libedmir_replay.so"
SN="--snapshots /capture/GLM/workers/snapshots --snapshots /capture/QWEN35/workers/snapshots --snapshots /capture/DSV2L/workers/snapshots"
