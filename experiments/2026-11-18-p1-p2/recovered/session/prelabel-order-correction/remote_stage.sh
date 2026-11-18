#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/p1p2
mkdir -p session
stage=${1:?prepare, collect or trace}
exec > >(tee "session/$stage.log") 2>&1
trap 'rc=$?; echo "$rc" > "session/'"$stage"'.exit-code"; date -u +%FT%TZ > "session/'"$stage"'.finished-utc"' EXIT
date -u +%FT%TZ > "session/$stage.started-utc"
export PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
if [[ "$stage" == prepare ]]; then
  bash /workspace/preparation/install-offline.sh
  dpkg-deb -x /workspace/nsight-systems-2025.5.2.deb /opt/edmir-nsys
  /opt/edmir-nsys/opt/nvidia/nsight-systems/2025.5.2/bin/nsys --version > session/nsys-version.txt
  nvidia-smi -q > session/nvidia-before.txt
  nvidia-smi topo -m > session/topology.txt
  nvcc --version > session/nvcc.txt
  python3 rebuild_inputs.py
  python3 run.py --action verify
  date -u +%FT%TZ > session/prepare-complete
elif [[ "$stage" == collect || "$stage" == trace ]]; then
  test -s session/prepare-complete
  python3 run.py --action "$stage"
  nvidia-smi -q > "session/nvidia-after-$stage.txt"
else
  exit 2
fi

