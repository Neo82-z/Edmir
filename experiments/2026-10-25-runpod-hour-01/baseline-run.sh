#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
task_root=/workspace/2026-10-25-regime-gpu-validation
test -f "$task_root/work-recheck/all-primary-finished.txt"
python3 -S /workspace/baseline_extension.py check --root "$task_root" --out /workspace/baseline-extension-01
python3 -S /workspace/baseline_extension.py collect --root "$task_root" --out /workspace/baseline-extension-01
python3 -S /workspace/baseline_extension.py review --root "$task_root" --out /workspace/baseline-extension-01 --report /workspace/baseline-extension-review.json
date -u --iso-8601=seconds > /workspace/baseline-extension-finished.txt
