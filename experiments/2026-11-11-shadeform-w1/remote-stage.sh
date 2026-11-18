#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
COLLECTION_CUTOFF=$(python3 -c 'import datetime; print(int(datetime.datetime(2026,11,11,8,7,tzinfo=datetime.timezone.utc).timestamp()))')
R=/workspace/bundle/source/EDM-IR/tools/run_w1_gpu.py
mkdir -p session
exec > >(tee -a session/stages.log) 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > session/exit-code; date -u +%FT%TZ > session/finished-utc' EXIT
date -u +%FT%TZ > session/started-utc
nvidia-smi -q > session/nvidia-smi-before.txt
nvidia-smi topo -m > session/topology.txt
nvcc --version > session/nvcc.txt
python3 -m pip freeze > session/python-packages.txt
if [[ ! -d inputs ]]; then
  python3 rebuild_inputs.py --out inputs
fi
python3 "$R" verify --bundle /workspace/bundle --inputs /workspace/inputs
python3 "$R" build --bundle /workspace/bundle --inputs /workspace/inputs --dependencies /workspace/dependencies --build-dir /workspace/build --jobs 8 --cutoff-epoch "$COLLECTION_CUTOFF"
python3 "$R" pilot --bundle /workspace/bundle --inputs /workspace/inputs --build-dir /workspace/build --out /workspace/results --cutoff-epoch "$COLLECTION_CUTOFF"
python3 "$R" collect --bundle /workspace/bundle --inputs /workspace/inputs --build-dir /workspace/build --out /workspace/results --cutoff-epoch "$COLLECTION_CUTOFF"
nvidia-smi -q > session/nvidia-smi-after.txt
printf '%s\n' completed > session/collection-status
