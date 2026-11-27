#!/bin/bash
# bootstrap.sh  — run on a fresh GPU node, in an SSH session with agent forwarding (ssh -A).
# Pulls the assets and the 2026-11-26 captures from the CPU node, verifies them, loads the image
# and lays out /srv/edmir-run. Safe to rerun: rsync resumes and finished steps are skipped.
#   CPU=shadeform@154.54.100.174 ./bootstrap.sh
set -u
cpu=${CPU:-shadeform@154.54.100.174}
A=/srv/edmir-assets; R=/srv/edmir-run; B=$A/run-2026-11-26
log() { echo "$(date -u +%H:%M:%S) $*"; }
ssh-keyscan -t ed25519 ${cpu#*@} 2>/dev/null > /tmp/cpu.key
fp=$(ssh-keygen -lf /tmp/cpu.key | awk '{print $2}')
[ "$fp" = "SHA256:D44ST9ZdqKJLOAfeb19RbwLI/OnD9eLDz8Ye3oRWeNM" ] || { log "CPU host key mismatch: $fp"; exit 9; }
mkdir -p ~/.ssh; touch ~/.ssh/known_hosts
grep -qF "$(awk '{print $3}' /tmp/cpu.key)" ~/.ssh/known_hosts || cat /tmp/cpu.key >> ~/.ssh/known_hosts
sudo install -d -o "$(id -u)" -g "$(id -g)" $A $A/handoff $R $R/receipts $R/work $R/work/cache/triton $R/work/run $R/capture $R/logs
rsync -a "$cpu:$A/handoff/transfer-files.txt" "$cpu:$A/handoff/transfer.sha256" $A/handoff/ </dev/null || { log "list fetch failed"; exit 8; }
# Small files first (code, bridge library, CPU records), then the image and the model shards.
grep -E "^(code-preflight|bridge|cpu-records|handoff|bootstrap)/" $A/handoff/transfer-files.txt > /tmp/small.txt
rsync -a --partial --files-from=/tmp/small.txt $cpu:$A/ $A/ </dev/null; log "small files exit=$?"
rsync -a --partial $cpu:$B/capture/ $R/capture/ </dev/null &
rsync -a $cpu:$B/capture.sha256 $cpu:$B/work/run/RUN.json /tmp/ </dev/null
cp /tmp/RUN.json $R/work/run/RUN.json
rm -f /tmp/tf.*; grep -vE "^(code-preflight|bridge|cpu-records|handoff|bootstrap)/" $A/handoff/transfer-files.txt | split -n r/8 - /tmp/tf.
ls /tmp/tf.* | xargs -P 8 -I{} rsync -a --partial --files-from={} $cpu:$A/ $A/ </dev/null
log "assets pass exit=$?"
wait; log "captures copied"
rsync -a --partial --files-from=$A/handoff/transfer-files.txt $cpu:$A/ $A/ </dev/null; log "final pass exit=$?"
cd $A; rm -f /tmp/sh.*; split -n r/32 handoff/transfer.sha256 /tmp/sh.
ls /tmp/sh.* | xargs -P 32 -I{} sha256sum --quiet -c {} > /tmp/sha.out 2>&1; rc=$?
log "asset checksums exit=$rc bad=$(grep -c . /tmp/sha.out)"; [ $rc -eq 0 ] || { head -5 /tmp/sha.out; exit 7; }
(cd $R && sha256sum --quiet -c /tmp/capture.sha256) && log "capture checksums ok" || { log "capture checksum FAILED"; exit 6; }
docker image inspect edmir/vllm:0.20.1-locked >/dev/null 2>&1 || docker load -i $A/vllm-0.20.1-amd64.docker.tar | tail -1
cp $A/bridge/libedmir_replay.so $R/work/
[ -d $R/code ] || cp -a $A/code-preflight $R/code
log "BOOTSTRAP DONE"
