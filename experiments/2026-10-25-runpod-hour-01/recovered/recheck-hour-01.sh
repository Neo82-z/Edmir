#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace/2026-10-25-regime-gpu-validation
test -f work/warmup-command-finished.txt
test ! -e work-recheck
mkdir work-recheck
ln -s ../work/build work-recheck/build
date -u --iso-8601=seconds > work-recheck/start.txt
python3 -S server.py run --work work-recheck --stage anchors --budget-seconds 900
python3 -S review.py --work work-recheck --out review-recheck-anchors-01.json
python3 -S server.py run --work work-recheck --stage library --budget-seconds 600
python3 -S review.py --work work-recheck --out review-recheck-01.json
date -u --iso-8601=seconds > work-recheck/finished.txt
