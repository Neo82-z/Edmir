#!/usr/bin/env bash
set -Eeuo pipefail
# For the fixed CUDA 12.8.1 / Ubuntu 24.04 image only; verified with network=none.
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DEBIAN_FRONTEND=noninteractive
cd "$root_dir/offline"
sha256sum --check SHA256SUMS
# APT orders Python's pre-dependencies. --no-download also suppresses local
# file acquisition on this APT version, so isolation is verified by Docker.
apt-get install --no-install-recommends -y "$root_dir"/offline/debs/*.deb
python3 -m venv /opt/edmir-python
/opt/edmir-python/bin/pip install --no-index --find-links="$root_dir/offline/wheels" cmake==3.30.5 numpy==2.2.6
git config --system --add safe.directory /workspace/dependencies/cutlass
git config --system --add safe.directory /workspace/dependencies/nccl
nvcc --version
/opt/edmir-python/bin/cmake --version
/opt/edmir-python/bin/python3 -c 'import numpy; print("numpy", numpy.__version__)'
