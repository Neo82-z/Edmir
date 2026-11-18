#!/usr/bin/env bash
set -Eeuo pipefail
cd "${1:?workspace}"
test -f PAYLOAD_SHA256.json
python3 - <<'PY'
from pathlib import Path
import hashlib,json
for n,h in json.loads(Path('PAYLOAD_SHA256.json').read_text()).items():
    with Path(n).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=h:raise SystemExit('payload mismatch: '+n)
print('payload verified')
PY
sudo -n docker run -d --name edmir-p1p2-20261118 --gpus all --ipc=host \
  -v "$PWD:/workspace" -w /workspace/p1p2 \
  nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 sleep infinity
sudo -n docker exec edmir-p1p2-20261118 timeout --signal=TERM --kill-after=10s 600s bash /workspace/p1p2/remote_stage.sh prepare
