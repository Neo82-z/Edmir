#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv git make g++ libnuma1 rsync ca-certificates
python3 -m venv /opt/edmir-python
/opt/edmir-python/bin/pip install --no-cache-dir cmake==3.30.5 numpy==2.2.6
touch /workspace/container-ready-ubuntu24
exec sleep infinity
