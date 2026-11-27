#!/bin/bash
# push_code.sh <user@gpu-node>  — run on the Mac, after bootstrap.sh created /srv/edmir-run/code.
# Brings the node's code folder to the current HEAD of main: the files of the deployed list, the
# exploratory folder and the confirmatory folder. Rewrites the commit stamp and the manifest and
# verifies every manifest entry against git.
set -e
H=$1; K=${KEY:-$HOME/Downloads/private_key.pem}
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
C=$(git rev-parse HEAD)
[ -z "$(git status --porcelain -- experiments/2026-11-23-replay experiments/2026-11-26-peer-read-probe experiments/2026-11-27-confirm EDM-IR/edmir EDM-IR/tools)" ] || { echo "uncommitted changes in deployed paths"; exit 2; }
{ cat experiments/2026-11-26-peer-read-probe/node/code-files.txt; git ls-files experiments/2026-11-26-peer-read-probe experiments/2026-11-27-confirm | grep -v "/results/"; } | sort -u > /tmp/edmir-deploy.txt
git archive HEAD $(cat /tmp/edmir-deploy.txt) | ssh -i $K $H "tar -x -C /srv/edmir-run/code"
ssh -i $K $H "cd /srv/edmir-run/code && echo $C > EDMIR_COMMIT && find . -type f ! -name EDMIR_CODE.sha256 ! -path './__pycache__/*' ! -name '*.pyc' | sort | xargs sha256sum > EDMIR_CODE.sha256 && cat EDMIR_CODE.sha256" > /tmp/edmir-node.sha
python3 - <<'PY'
import subprocess, hashlib
bad=n=0
for line in open("/tmp/edmir-node.sha").read().splitlines():
    digest,name = line.split("  ",1); name = name[2:]; n += 1
    if name=="EDMIR_COMMIT": continue
    blob = subprocess.run(["git","show","HEAD:"+name],capture_output=True)
    if blob.returncode or hashlib.sha256(blob.stdout).hexdigest()!=digest: bad += 1; print("DIFF",name)
print("deployed",subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),"|",n,"entries;",bad,"differ")
raise SystemExit(1 if bad else 0)
PY
# The node-side scripts run from /srv/edmir-run.
ssh -i $K $H "cd /srv/edmir-run && cp code/experiments/2026-11-26-peer-read-probe/node/*.sh . && cp code/experiments/2026-11-27-confirm/run_queue.sh . && chmod +x *.sh && ls *.sh | tr '\n' ' '"
