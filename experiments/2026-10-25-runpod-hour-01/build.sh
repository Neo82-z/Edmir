#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace/2026-10-25-regime-gpu-validation
# The macOS tar retained uid 501. Restore ownership of our uploaded dependency
# copies; do not disable Git ownership checks or alter dependency contents.
chown -R root:root dependencies/cutlass dependencies/nccl
mv work/build-deps-command.txt work/build-deps-initial-ownership-failure.txt
python3 -S verify_delivery.py > work/delivery-verification-after-chown.txt 2>&1
timeout --signal=TERM --kill-after=15s 600s python3 -S server.py build-deps --jobs 8 > work/build-deps-command.txt 2>&1
date -u --iso-8601=seconds > work/dependencies-finished.txt
timeout --signal=TERM --kill-after=15s 1200s python3 -S server.py build --jobs 8 > work/build-command.txt 2>&1
python3 -S server.py preflight > work/preflight-command.txt 2>&1
date -u --iso-8601=seconds > work/bootstrap-finished.txt
