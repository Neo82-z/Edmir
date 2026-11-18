#!/usr/bin/env bash
set -uo pipefail
# Native Linux output timestamps avoid VirtioFS/host clock skew during make.
if python3 "$@"; then
  build_status=0
else
  build_status=$?
fi
if [[ -d /workspace/build-precision ]]; then
  cp -a /workspace/build-precision /output/ || exit 2
fi
exit "$build_status"
