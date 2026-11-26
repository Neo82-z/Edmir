#!/bin/bash
# Rerun the device smoke and keep the log and receipt of every attempt.
source /srv/edmir-run/env.sh
n=$(ls $R/logs/smoke-attempt* 2>/dev/null | wc -l)
[ -f $R/logs/smoke.log ] && cp $R/logs/smoke.log $R/logs/smoke-attempt$((n+1)).log
[ -f $R/work/smoke.json ] && sudo mv $R/work/smoke.json $R/work/smoke-attempt$((n+1)).json
echo "smoke start $(date -u +%H:%M:%S)"
timeout 560 $RUN $T smoke --gate /receipts/gate.json --out /work/smoke.json > $R/logs/smoke.log 2>&1; echo "smoke exit=$? $(date -u +%H:%M:%S)"
python3 - <<"PY"
import json, collections
s=json.load(open("/srv/edmir-run/work/smoke.json"))
c=collections.Counter()
for group in ("arms","balanced"):
    for unit,arms in s[group].items():
        for arm,r in arms.items():
            err=str(r.get("rank_errors") or r.get("error") or "")[:170]
            c[(group, r.get("status"), r.get("launch_mode"), err, str(r.get("graph_errors"))[:120])]+=1
            if r.get("status")!="ok": print("FAIL", unit, arm, r.get("status"), r.get("scaled_max_err"), r.get("nonfinite"), r.get("fingerprint_ok"), err)
for k,v in sorted(c.items(), key=str): print(v, k)
PY
