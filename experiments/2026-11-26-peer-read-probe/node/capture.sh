#!/bin/bash
# capture.sh <模型> <清单(容器内路径)> <调优配置文件名|none>   (START_HERE.md 第 7 步)
A=/srv/edmir-assets; R=/srv/edmir-run
m=$1; manifest=$2; tune=$3
extra=(); [ "$tune" != none ] && extra=(--native-tune "/moe_configs/$tune")
docker run --rm --pull never --network none --gpus all --ipc=host --hostname edmir --add-host edmir:127.0.0.1 \
  -v $R/code:/code:ro -v $A/models/$m:/model:ro \
  -v $A/cpu-records:/cpu-records:ro -v $R/work/moe_configs:/moe_configs:ro \
  -v $R/capture:/work \
  -e EDMIR_MOE_MODEL=$m \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e HF_HOME=/work/cache/hf -e XDG_CACHE_HOME=/work/cache -e VLLM_CACHE_ROOT=/work/cache/vllm \
  -e TRITON_CACHE_DIR=/work/cache/triton -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONPATH=/code/EDM-IR -e PYTHONUNBUFFERED=1 -e VLLM_HOST_IP=127.0.0.1 \
  --entrypoint python3 edmir/vllm:0.20.1-locked /code/EDM-IR/tools/run_real_moe.py --execute \
  --mode routes --snapshot-replay "${extra[@]}" \
  --corpus /code/experiments/2026-11-17-real-moe-local/corpus/requests-source.json \
  --tokens /cpu-records/$m/requests-tokenized.json --cpu-receipt /cpu-records/$m/cpu-receipt.json \
  --manifest "$manifest" --model-dir /model \
  --source-lock /code/experiments/2026-11-17-real-moe-local/vllm-source-lock.json --out /work/$m
