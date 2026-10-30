#!/usr/bin/env bash
# Run only on a separately authorized GPU Pod with the existing network volume.
set -euo pipefail
ASSET_ROOT=/workspace/edmir-assets
if [[ "${1:-}" != --install ]]; then
  echo 'Usage: bash install_cached_gpu_env.sh --install'
  echo 'Requires Linux x86_64, glibc >=2.35, a compatible NVIDIA driver and the asset volume.'
  echo 'Creates a separate Python 3.12 environment; never starts or rents a cloud instance.'
  exit 0
fi
"$ASSET_ROOT/download-venv/bin/python" - <<'PY'
import platform
assert platform.machine() == 'x86_64'
name, version = platform.libc_ver()
assert name == 'glibc' and tuple(map(int, version.split('.')[:2])) >= (2, 35), (name, version)
PY
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [[ -e "$ASSET_ROOT/gpu-venv" ]]; then
  echo 'gpu-venv already exists; inspect it before making changes' >&2
  exit 2
fi
export UV_CACHE_DIR="$ASSET_ROOT/uv-cache"
export UV_LINK_MODE=copy
"$ASSET_ROOT/tools/uv/uv" venv --python "$ASSET_ROOT/download-venv/bin/python" "$ASSET_ROOT/gpu-venv"
"$ASSET_ROOT/download-venv/bin/python" -m pip --python "$ASSET_ROOT/gpu-venv/bin/python" install \
  --no-index --find-links "$ASSET_ROOT/software/vllm-0.20.1-cp312-linux-x86_64" \
  --require-hashes -r "$ASSET_ROOT/manifests/gpu-requirements.lock"
"$ASSET_ROOT/gpu-venv/bin/python" - <<'PY'
import json, torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA driver/runtime check failed'
result = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
          'gpu_count': torch.cuda.device_count(),
          'model_execution_validated': False, 'multigpu_communication_validated': False}
Path('/workspace/edmir-assets/manifests/gpu-runtime-smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
PY
