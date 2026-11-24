#!/usr/bin/env python3
"""Seal a second prediction from the node's microbenchmark. Offline; no GPU commands.

The frozen calibrate.py and predict.py of the 2026-11-20 package are used unchanged.
Only parameters change: the pre-boot scenario values are replaced by values measured on
this node before any formal timing. Run it after the microbenchmark and pass the output
directory to `replay.py prepare --calibrated-prediction`.

How microbenchmark probes map to predictor parameters (fixed before boot):
  b_in       peer_copy (1 peer) and peer_read_7 (7 peers): bytes read per second by CTA count
  b_hbm      hbm_copy: bytes read plus written per second
  a2a        all_to_all: least-squares latency and bandwidth over the 11 grid payloads
  mma_peak   consumer_rate: FLOP/s of the shared Triton consumer on local rows, all SM slots filled
  G_launch   graph_launch: device time added by one more kernel launch in a replayed CUDA graph
  F_direct   two peer_barrier calls: the publish and release barriers of a direct iteration
  F_staged   defined as 1 us: a staged iteration has no fixed step besides the AllToAll, whose
             latency is fitted separately; the calibrator requires a positive value
Parameters without a probe (SM count, tile sizes, occupancy, L2 size, split-K partial bytes,
lt_launch, sem_per_slice, allgather) are copied from the scenario and listed in the receipt.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from make_design import ROOT, sha

ARCHIVE = ROOT.parent/"2026-11-20-final-round"
SCHEMA = "edmir.moe-path-calibration.v1"
F_STAGED_SECONDS = 1e-6
DEFINED = {"F_direct":"2 x peer_barrier","F_staged":"defined as 1e-6 s","G_launch":"graph_launch per launch"}


def records(micro):
    """One calibrator row per rank and probe; failed probes are skipped and reported."""
    rows,failed = [],set()
    for readings in micro["rank_measurements"]:
        rows.append(dict(kind="F_staged",seconds=F_STAGED_SECONDS))
        for r in readings:
            kind = r["kind"]
            if "error" in r:
                failed.add(kind)
            elif kind in ("peer_copy","peer_read_7"):
                rows.append(dict(kind="b_in",peers=r["peers"],ctas=r["ctas"],bytes=r["bytes"],seconds=r["seconds"]))
            elif kind=="hbm_copy":
                rows.append(dict(kind="hbm",bytes=r["bytes"],seconds=r["seconds"]))
            elif kind=="all_to_all":
                rows.append(dict(kind="a2a",bytes_in_per_rank=r["remote_bytes"],seconds=r["seconds"]))
            elif kind=="consumer_rate":
                rows.append(dict(kind="mma",flops=r["flops"],ctas=r["ctas"],g=r["g"],seconds=r["seconds"]))
            elif kind=="graph_launch":
                rows.append(dict(kind="G_launch",seconds=r["seconds"]))
            elif kind=="peer_barrier":
                rows.append(dict(kind="F_direct",seconds=2*r["seconds"]))
    return rows,sorted(failed)


def calibrated_parameters(micro, base):
    sys.path.insert(0,str(ARCHIVE/"scripts"))
    from calibrate import calibrate  # frozen

    rows,failed = records(micro)
    return calibrate(base,rows),rows,failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microbench",type=Path,required=True)
    ap.add_argument("--design",type=Path,default=ROOT/"design")
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    out = a.out.resolve()
    scenario = ARCHIVE/"params"/"scenario.json"
    receipt = dict(schema=SCHEMA,status="failed",microbench_sha256=sha(a.microbench),
                   scenario_sha256=sha(scenario),calibrate_sha256=sha(ARCHIVE/"scripts"/"calibrate.py"),
                   mapping_sha256=sha(Path(__file__)),defined=DEFINED)
    try:
        params,rows,failed = calibrated_parameters(json.loads(a.microbench.read_text()),
                                                   json.loads(scenario.read_text()))
        receipt.update(failed_probes=failed,copied_from_base=params["copied_from_base"])
        (out/"microbench.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
        (out/"params.json").write_text(json.dumps(params,indent=2,sort_keys=True)+"\n")
        subprocess.run([sys.executable,"-B",str(ARCHIVE/"scripts"/"predict.py"),"--params",str(out/"params.json"),
                        "--design",str(a.design.resolve()),"--out",str(out)],check=True,timeout=300)
        receipt.update(status="sealed",params_sha256=sha(out/"params.json"))
    except (ValueError,KeyError,subprocess.SubprocessError) as exc:
        receipt["reason"] = repr(exc)
    (out/"CALIBRATION.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(receipt["status"],receipt.get("reason",""))
    return 0 if receipt["status"]=="sealed" else 3


if __name__=="__main__":
    sys.exit(main())
