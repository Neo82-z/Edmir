#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace/2026-10-25-regime-gpu-validation
test -f work-recheck/finished.txt
python3 -S server.py run --work work-recheck --stage crossings --budget-seconds 900
python3 -S review.py --work work-recheck --out review-recheck-complete-02.json
date -u --iso-8601=seconds > work-recheck/all-primary-finished.txt
