#!/bin/bash
# bootstrap.sh core|models  — run on a fresh GPU node, in an SSH session with agent forwarding (ssh -A).
#   core    everything the replay needs: code, bridge library, CPU records, the image and the
#           2026-11-26 captures (about 14 GB). Verifies them, loads the image, lays out /srv/edmir-run.
#   models  the three model directories and the tools (about 160 GB), only needed for new captures.
#           Runs at low priority so it can share the node with GPU work that is not being timed.
# Safe to rerun: transfers resume and finished steps are skipped.
#   CPU=shadeform@154.54.100.174 ./bootstrap.sh core
set -u
cpu=${CPU:-shadeform@154.54.100.174}
A=/srv/edmir-assets; R=/srv/edmir-run; B=$A/run-2026-11-26
IMG=vllm-0.20.1-amd64.docker.tar
SMALL="^(code-preflight|bridge|cpu-records|handoff|bootstrap)/"
log() { echo "$(date -u +%H:%M:%S) $*"; }
verify() {  # verify <pattern of paths in transfer.sha256>
  cd $A; rm -f /tmp/sh.*; grep -E "  ($1)" handoff/transfer.sha256 | split -n r/32 - /tmp/sh.
  ls /tmp/sh.* | xargs -P 32 -I{} sha256sum --quiet -c {} > /tmp/sha.out 2>&1; rc=$?
  log "checksums $1 exit=$rc bad=$(grep -c . /tmp/sha.out)"; [ $rc -eq 0 ] || head -5 /tmp/sha.out
  return $rc
}

case "${1:-}" in
core)
  ssh-keyscan -t ed25519 ${cpu#*@} 2>/dev/null > /tmp/cpu.key
  fp=$(ssh-keygen -lf /tmp/cpu.key | awk '{print $2}')
  [ "$fp" = "SHA256:D44ST9ZdqKJLOAfeb19RbwLI/OnD9eLDz8Ye3oRWeNM" ] || { log "CPU host key mismatch: $fp"; exit 9; }
  mkdir -p ~/.ssh; touch ~/.ssh/known_hosts
  grep -qF "$(awk '{print $3}' /tmp/cpu.key)" ~/.ssh/known_hosts || cat /tmp/cpu.key >> ~/.ssh/known_hosts
  sudo install -d -o "$(id -u)" -g "$(id -g)" $A $A/handoff $R $R/receipts $R/work $R/work/cache/triton $R/work/run $R/capture $R/logs
  rsync -a "$cpu:$A/handoff/transfer-files.txt" "$cpu:$A/handoff/transfer.sha256" $A/handoff/ </dev/null || { log "list fetch failed"; exit 8; }
  grep -E "$SMALL" $A/handoff/transfer-files.txt > /tmp/small.txt
  rsync -a --partial --files-from=/tmp/small.txt $cpu:$A/ $A/ </dev/null; log "small files exit=$?"
  rsync -a $cpu:$B/capture.sha256 $cpu:$B/work/run/RUN.json /tmp/ </dev/null && cp /tmp/RUN.json $R/work/run/RUN.json
  # The image is one 8 GB file: fetch it as eight byte ranges in parallel, then check its hash.
  if ! (cd $A && grep -E "  $IMG\$" handoff/transfer.sha256 | sha256sum --quiet -c - 2>/dev/null); then
    size=$(ssh -n $cpu stat -c %s $A/$IMG); part=$((size/8/1048576+1))
    for i in 0 1 2 3 4 5 6 7; do
      ssh -n $cpu "dd if=$A/$IMG bs=1M skip=$((i*part)) count=$part status=none" \
        | dd of=$A/$IMG bs=1M seek=$((i*part)) iflag=fullblock conv=notrunc status=none &
      sleep 0.5   # stay under the CPU node's limit on unauthenticated connections
    done
  fi
  # Captures in eight streams.
  ssh -n $cpu "cd $B && find capture -type f" > /tmp/cap.txt; rm -f /tmp/cf.*; split -n r/8 /tmp/cap.txt /tmp/cf.
  for f in /tmp/cf.*; do rsync -a --partial --files-from=$f $cpu:$B/ $R/ </dev/null & sleep 0.5; done
  wait; log "image and captures copied"
  core="code-preflight/|bridge/|cpu-records/|$IMG"
  if ! verify "$core"; then   # one plain pass repairs whatever a stream dropped
    rsync -a --files-from=/tmp/small.txt $cpu:$A/ $A/ </dev/null; rsync -a --inplace $cpu:$A/$IMG $A/ </dev/null
    rsync -a $cpu:$B/capture/ $R/capture/ </dev/null
    verify "$core" || exit 7
  fi
  (cd $R && sha256sum --quiet -c /tmp/capture.sha256) && log "capture checksums ok" || { log "capture checksum FAILED"; exit 6; }
  docker image inspect edmir/vllm:0.20.1-locked >/dev/null 2>&1 || docker load -i $A/$IMG | tail -1
  cp $A/bridge/libedmir_replay.so $R/work/
  [ -d $R/code ] || cp -a $A/code-preflight $R/code
  log "CORE DONE" ;;
models)
  rm -f /tmp/tf.*; grep -vE "$SMALL|^$IMG\$" $A/handoff/transfer-files.txt | split -n r/8 - /tmp/tf.
  ls /tmp/tf.* | xargs -P 8 -I{} nice -n 19 ionice -c 3 rsync -a --partial --files-from={} $cpu:$A/ $A/
  log "models pass exit=$?"
  nice -n 19 rsync -a --partial --files-from=$A/handoff/transfer-files.txt $cpu:$A/ $A/ </dev/null; log "final pass exit=$?"
  verify "models/|tools/" || exit 7
  log "MODELS DONE" ;;
*) echo "usage: bootstrap.sh core|models"; exit 2 ;;
esac
