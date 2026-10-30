#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/edmir-assets
EXPECTED_VOLUME=zyw8svwb4a
case "$(findmnt -n -o SOURCE -T /workspace)" in
  *"/networkvolumes/$EXPECTED_VOLUME"*) ;;
  *) echo 'Unexpected persistent volume; refusing download' >&2; exit 2 ;;
esac
if compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
  echo 'GPU devices visible; refusing CPU preparation' >&2
  exit 2
fi
mkdir -p "$ROOT"/{tools,logs,manifests,software,uv-cache,python}
export UV_INSTALL_DIR="$ROOT/tools/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
export UV_CACHE_DIR="$ROOT/uv-cache"
export UV_LINK_MODE=copy
export HF_XET_HIGH_PERFORMANCE=0
export HF_XET_NUM_CONCURRENT_RANGE_GETS=8
export HF_HUB_DOWNLOAD_TIMEOUT=120
curl --fail --location --retry 3 https://astral.sh/uv/0.13.6/install.sh -o "$ROOT/software/install-uv-0.13.6.sh"
sh "$ROOT/software/install-uv-0.13.6.sh"
"$UV_INSTALL_DIR/uv" python install 3.12
"$UV_INSTALL_DIR/uv" venv --python 3.12 "$ROOT/download-venv"
"$UV_INSTALL_DIR/uv" pip install --python "$ROOT/download-venv/bin/python" huggingface-hub==2.3.0
"$UV_INSTALL_DIR/uv" pip freeze --python "$ROOT/download-venv/bin/python" > "$ROOT/manifests/download-requirements.lock"
"$ROOT/download-venv/bin/python" - <<'PY'
import inspect, json, platform, sys
import huggingface_hub
from pathlib import Path
assert 'max_workers' in inspect.signature(huggingface_hub.snapshot_download).parameters
Path('/workspace/edmir-assets/manifests/download-environment.json').write_text(json.dumps({
    'python': sys.version, 'platform': platform.platform(),
    'huggingface_hub': huggingface_hub.__version__, 'gpu_devices': [],
    'network_volume_id': 'zyw8svwb4a', 'pod_id': '66t5tzsexfmegv',
}, indent=2))
PY
set +e
timeout --signal=INT --kill-after=60s 12600 "$ROOT/download-venv/bin/python" -u \
  "$ROOT/prepare_model_assets.py" --download --root "$ROOT"
download_status=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/manifests/download-process-finished-at.txt"
printf '%s\n' "$download_status" > "$ROOT/manifests/download-process-exit-code.txt"
# This timeout bounds the job only. Platform Stop is separately required to end billing.
exit "$download_status"
