#!/usr/bin/env bash
set -Eeuo pipefail
cd "${1:?workspace required}"
mkdir -p session
exec > >(tee session/host-prepare.log) 2>&1
date -u +%FT%TZ
printf '%s  %s\n' ad0004740501eb74de4f9d5cb8494c59584875f8dafed3e20c54dea943c846ad payload.tar.gz | sha256sum --check -
tar -xzf payload.tar.gz
python3 verify-files.py . PAYLOAD_SHA256.json
sudo -n docker run -d --name edmir-final-20261117 --gpus all --ipc=host -v "$PWD:/workspace" -w /workspace nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 sleep infinity
sudo -n docker exec edmir-final-20261117 timeout --signal=TERM --kill-after=10s 600s bash /workspace/remote-stage.sh prepare
