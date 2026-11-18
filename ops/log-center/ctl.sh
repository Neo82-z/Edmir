#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export DOCKER_CONTEXT=colima-edmir-logs
compose() { docker-compose --env-file .local/stack.env -f compose.yaml "$@"; }
remotes() {
  for config in .local/remotes/*.json; do
    [[ -f "$config" ]] || continue
    name="${config##*/}"
    python3 remote.py "$1" "${name%.json}"
  done
}
case "${1:-status}" in
  up)
    python3 setup.py init
    if ! colima --profile edmir-logs status >/dev/null 2>&1; then
      colima start edmir-logs --cpus 6 --memory 10 --disk 60 --vm-type vz --mount-type virtiofs \
        --mount "$PWD:w" --ssh-config=false --activate=false
    fi
    colima --profile edmir-logs ssh -- sudo sysctl -w vm.max_map_count=1048576
    compose up -d --wait --wait-timeout 240 elasticsearch
    python3 setup.py es
    compose up -d kibana logstash
    python3 - <<'PY'
import time
from setup import request
end=time.monotonic()+240
while time.monotonic()<end:
    try:
        if request('/api/status',kibana=True).get('status',{}).get('overall',{}).get('level')=='available':break
    except Exception:pass
    time.sleep(2)
else:raise SystemExit('Kibana is not ready; preserve logs and inspect ctl.sh logs')
PY
    python3 setup.py kibana-ensure
    remotes start
    printf '%s\n' 'Kibana: http://127.0.0.1:15601' 'Credentials: .local/LOGIN.txt'
    ;;
  status) compose ps; python3 setup.py status; remotes status ;;
  logs) compose logs --tail 80 ;;
  stop) remotes stop; compose stop; colima --profile edmir-logs stop ;;
  remote-start) remotes start ;;
  remote-stop) remotes stop ;;
  backup)
    mkdir -p .local/backups
    archive=".local/backups/journal-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    tar -czf "$archive" -C .local events saved-objects.ndjson
    shasum -a 256 "$archive"
    ;;
  *) printf '%s\n' 'Usage: ./ctl.sh up|status|logs|stop|backup|remote-start|remote-stop'; exit 2 ;;
esac
