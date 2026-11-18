#!/usr/bin/env bash
set -Eeuo pipefail
# Use the exact package hashed during the previous successful W1 capture.
cd /workspace
mkdir -p tools/nvidia
package=nsight-systems-2025.1.3_2025.1.3.140-1_amd64.deb
url=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$package
if [[ ! -f "tools/nvidia/$package" ]]; then
  curl --fail --location --continue-at - --retry 3 --retry-all-errors --connect-timeout 20 --max-time 900 \
    --output "tools/nvidia/$package.partial" "$url"
  printf '%s  %s\n' 77290633afb63a119d53a9f11e2c64cfcc2c3b79a2fb5b4b4e673e1de1dd6836 \
    "tools/nvidia/$package.partial" | sha256sum --check -
  mv "tools/nvidia/$package.partial" "tools/nvidia/$package"
fi
printf '%s  %s\n' 77290633afb63a119d53a9f11e2c64cfcc2c3b79a2fb5b4b4e673e1de1dd6836 \
  "tools/nvidia/$package" | sha256sum --check -
# Extract on Linux first: package directory modes are read-only on macOS mounts.
staging=$(mktemp -d /tmp/edmir-nsys.XXXXXX)
trap 'rm -rf -- "$staging"' EXIT
dpkg-deb -x "tools/nvidia/$package" "$staging"
chmod -R u+w "$staging/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64"
test ! -e tools/nvidia/target-linux-x64
cp -a "$staging/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64" tools/nvidia/target-linux-x64
tools/nvidia/target-linux-x64/nsys \
  --version | tee tools/nvidia/version.txt
# --version validates the executable only. Real CUDA capture needs the GPU node.
