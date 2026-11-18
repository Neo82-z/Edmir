#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
stage=${1:?prepare or collect required}
[[ "$stage" == prepare || "$stage" == collect ]] || exit 2
mkdir -p session
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_name="$stage-$stamp"
exec > >(tee "session/$run_name.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "session/$run_name.exit-code"; date -u +%FT%TZ > "session/$run_name.finished-utc"' EXIT
date -u +%FT%TZ > "session/$run_name.started-utc"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:/opt/edmir-nsys/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
if [[ "$stage" == prepare ]]; then
  test ! -e session/prepare-complete
  printf '%s  %s\n' 9e1e32f5205027990914dd5690dc9a3c7a17d8e7faf245a76b289f02a245fdcd edmir-w1-code-deps.tar.gz | sha256sum --check -
  tar -xzf edmir-w1-code-deps.tar.gz
  bash preparation/install-offline.sh
  printf '%s  %s\n' 77290633afb63a119d53a9f11e2c64cfcc2c3b79a2fb5b4b4e673e1de1dd6836 tools/nvidia/nsight-systems-2025.1.3_2025.1.3.140-1_amd64.deb | sha256sum --check -
  dpkg-deb -x tools/nvidia/nsight-systems-2025.1.3_2025.1.3.140-1_amd64.deb /opt/edmir-nsys
  nsys --version > session/nsys-version.txt
  nvcc --version > session/nvcc-version.txt
  python3 -m pip freeze > session/python-packages.txt
  nvidia-smi -q > session/nvidia-smi-before.txt
  nvidia-smi topo -m > session/topology.txt
  python3 rebuild_inputs.py --out inputs
  python3 precision-bundle/source/EDM-IR/tools/run_w1_gpu.py verify --bundle precision-bundle --inputs inputs
  python3 window/run_window.py --bundle precision-bundle --dry-run
  date -u +%FT%TZ > session/prepare-complete
else
  test -s session/prepare-complete
  python3 window/run_window.py --bundle precision-bundle --inputs inputs --build-dir build-precision \
    --out results-precision --guard window/platform-guard.json
  nvidia-smi -q > session/nvidia-smi-after.txt
  date -u +%FT%TZ > session/collector-exited-successfully
fi
