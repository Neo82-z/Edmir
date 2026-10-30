#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /workspace/2026-10-25-regime-gpu-validation
stage=$1
case "$stage" in anchors|crossings|library|warmup) ;; *) exit 2 ;; esac
test -f work/bootstrap-finished.txt
budget=$(python3 - "$2" <<'PY'
from datetime import datetime, timezone
import sys,time
deadline=datetime(2026,10,25,6,29,10,tzinfo=timezone.utc).timestamp()
print(min(int(sys.argv[1]),max(0,int(deadline-time.time()))))
PY
)
test "$budget" -ge 180
date -u --iso-8601=seconds > "work/${stage}-command-start.txt"
python3 -S server.py run --stage "$stage" --budget-seconds "$budget"
python3 -S review.py --work work --out "review-after-${stage}-01.json"
date -u --iso-8601=seconds > "work/${stage}-command-finished.txt"
