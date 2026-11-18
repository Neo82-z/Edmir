#!/usr/bin/env bash
# Run inside the pinned CUDA12.8.1/Ubuntu24.04 container, /workspace mounted.
set -Eeuo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p session
exec > >(tee -a session/bootstrap.log) 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > session/bootstrap-exit-code' EXIT
test -f extension/platform-guard.json
test ! -e bundle
printf '%s  %s\n' 9e1e32f5205027990914dd5690dc9a3c7a17d8e7faf245a76b289f02a245fdcd edmir-w1-code-deps.tar.gz | sha256sum -c -
tar -xzf edmir-w1-code-deps.tar.gz
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv git make g++ libnuma1 rsync ca-certificates
python3 -m venv /opt/edmir-python
python3 -m pip install --no-cache-dir cmake==3.30.5 numpy==2.2.6
mkdir -p /opt/edmir-nsys
(
  cd /opt/edmir-nsys
  apt-get download nsight-systems-2025.1.3=2025.1.3.140-251335620677v0
  printf '%s  %s\n' 77290633afb63a119d53a9f11e2c64cfcc2c3b79a2fb5b4b4e673e1de1dd6836 nsight-systems-2025.1.3_2025.1.3.140-251335620677v0_amd64.deb | sha256sum -c -
  dpkg-deb -x nsight-systems-2025.1.3_2025.1.3.140-251335620677v0_amd64.deb extracted
)
export PATH=/opt/edmir-nsys/extracted/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64:$PATH
nsys --version > session/nsys-version.txt
nvidia-smi -q > session/nvidia-smi-before.txt
nvidia-smi topo -m > session/topology.txt
nvcc --version > session/nvcc.txt
python3 -m pip freeze > session/python-packages.txt
python3 rebuild_inputs.py --out inputs
test ! -e build
cp -a extension/payload/build build
# These are the same hash-verified binaries recovered from the first run.
# run_extension preflight verifies all four binaries, dependencies and loaded libraries.
python3 extension/run_extension.py --bundle bundle --dry-run
python3 extension/run_extension.py --bundle bundle --inputs inputs --build-dir build \
  --out results --guard extension/platform-guard.json
nvidia-smi -q > session/nvidia-smi-after.txt
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > session/processes-after.txt
