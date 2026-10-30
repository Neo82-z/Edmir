#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace
printf '%s  %s\n' d9739cb102401acf6d5950313bbadddf998cf92fe58f786b7146f7d407548007 2026-10-25-regime-gpu-validation.tar.gz | sha256sum -c -
test ! -e 2026-10-25-regime-gpu-validation
tar -xzf 2026-10-25-regime-gpu-validation.tar.gz
cd 2026-10-25-regime-gpu-validation
mkdir -p work
date -u --iso-8601=seconds > work/bootstrap-start.txt
python3 -S verify_delivery.py > work/delivery-verification.txt 2>&1
python3 -S server.py verify > work/package-verification.txt 2>&1
python3 -S server.py inventory > work/inventory-command.txt 2>&1
timeout --signal=TERM --kill-after=15s 600s python3 -S server.py build-deps --jobs 8 > work/build-deps-command.txt 2>&1
date -u --iso-8601=seconds > work/dependencies-finished.txt
timeout --signal=TERM --kill-after=15s 1200s python3 -S server.py build --jobs 8 > work/build-command.txt 2>&1
python3 -S server.py preflight > work/preflight-command.txt 2>&1
date -u --iso-8601=seconds > work/bootstrap-finished.txt
