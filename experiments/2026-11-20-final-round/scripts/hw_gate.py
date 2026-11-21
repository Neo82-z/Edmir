#!/usr/bin/env python3
"""Read-only hardware gate for the 8xA100 node. Never changes power limits or clocks.

  python3 hw_gate.py --out gate.json [--load-test] [--p2p-bin /path/p2pBandwidthLatencyTest] \
      [--nccl-alltoall /path/alltoall_perf]

Checks: 8 GPUs, name contains A100 and 80GB, MIG off, enforced power limit equal to the default limit,
every GPU pair connected by NV12, peer read/write OK. With --load-test it runs BF16 matmuls on all GPUs
(PyTorch) while sampling SM clock, power draw and clock-event reasons.
Low instantaneous power draw is never a failure by itself; a lowered enforced limit is.
Exit 0 = PASS, 2 = FAIL (stop formal runs; the user contacts the provider), 3 = could not check.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ("index", "name", "memory.total", "mig.mode.current", "power.limit", "enforced.power.limit",
          "power.default_limit", "power.max_limit", "clocks.max.sm", "pci.bus_id")
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def smi(*args):
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def query(field):
    """One field at a time, so an unsupported field name does not hide the others."""
    try:
        out = smi(f"--query-gpu={field}", "--format=csv,noheader,nounits")
    except RuntimeError as e:
        return None, str(e)
    return [x.strip() for x in out.strip().splitlines()], None


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def data_rows(text, count):
    """Rows of an nvidia-smi topology matrix: 'GPUi <cell> <cell> ...'; header and legend lines are skipped."""
    rows = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        cells = line.split()
        if len(cells) > count and re.fullmatch(r"GPU\d+", cells[0]) and not cells[1].startswith("GPU"):
            rows.append(cells)
    return rows


def topology_ok(text, count):
    rows, bad = data_rows(text, count), []
    for cells in rows[:count]:
        for j, link in enumerate(cells[1:count + 1]):
            if f"GPU{j}" != cells[0] and link != "NV12":
                bad.append(f"{cells[0]}-GPU{j}:{link}")
    return not bad and len(rows) >= count, bad


def p2p_ok(text, count):
    rows = data_rows(text, count)
    return len(rows) >= count and all(c in ("OK", "X") for r in rows[:count] for c in r[1:count + 1])


def load_test(seconds, count):
    script = (
        "import sys, time, torch\n"
        "d = int(sys.argv[1]); t_end = time.time() + float(sys.argv[2])\n"
        "torch.cuda.set_device(d)\n"
        "a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16); b = torch.randn_like(a)\n"
        "while time.time() < t_end:\n"
        "    for _ in range(20): c = a @ b\n"
        "    torch.cuda.synchronize()\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, str(d), str(seconds)]) for d in range(count)]
    samples = []
    time.sleep(min(5.0, seconds / 3))
    reason_field = next((f for f in REASON_FIELDS if query(f)[0] is not None), None)
    while any(p.poll() is None for p in procs):
        fields = ["index", "clocks.sm", "power.draw"] + ([reason_field] if reason_field else [])
        try:
            out = smi(f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
            samples.append([[x.strip() for x in line.split(",")] for line in out.strip().splitlines()])
        except RuntimeError:
            pass
        time.sleep(1.0)
    codes = [p.wait() for p in procs]
    return dict(exit_codes=codes, reason_field=reason_field, samples=samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--load-test", action="store_true")
    ap.add_argument("--load-seconds", type=float, default=60.0)
    ap.add_argument("--min-load-sm-mhz", type=float, default=1275.0,
                    help="median SM clock under load below this is reported as a failure")
    ap.add_argument("--p2p-bin", type=Path)
    ap.add_argument("--nccl-alltoall", type=Path)
    args = ap.parse_args()

    report, failures, unchecked = dict(fields={}), [], []
    try:
        for f in FIELDS:
            values, err = query(f)
            report["fields"][f] = values if values is not None else {"error": err}
        report["topology"] = smi("topo", "-m")
        report["p2p_read"] = smi("topo", "-p2p", "r")
        report["p2p_write"] = smi("topo", "-p2p", "w")
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        report["error"] = str(e)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"UNCHECKED: {e}")
        return 3

    fields = report["fields"]
    names = fields["name"] if isinstance(fields["name"], list) else []
    if len(names) != args.gpus:
        failures.append(f"expected {args.gpus} GPUs, found {len(names)}")
    for i, name in enumerate(names):
        if "A100" not in name or "80GB" not in name:
            failures.append(f"GPU{i} is {name}")
    mig = fields["mig.mode.current"]
    if isinstance(mig, list) and any(m not in ("Disabled", "[N/A]", "N/A") for m in mig):
        failures.append(f"MIG mode {mig}")
    limit = fields["enforced.power.limit"] if isinstance(fields["enforced.power.limit"], list) else fields["power.limit"]
    default = fields["power.default_limit"]
    if isinstance(limit, list) and isinstance(default, list):
        for i, (l, d) in enumerate(zip(limit, default)):
            if number(l) is None or number(d) is None:
                unchecked.append(f"GPU{i} power limit unreadable ({l}/{d})")
            elif number(l) < number(d) - 1.0:
                failures.append(f"GPU{i} enforced power limit {l} W below default {d} W")
    else:
        unchecked.append("power limit fields unavailable")
    ok, bad = topology_ok(report["topology"], args.gpus)
    if not ok:
        failures.append("not all pairs NV12: " + ", ".join(bad[:8]))
    if not (p2p_ok(report["p2p_read"], args.gpus) and p2p_ok(report["p2p_write"], args.gpus)):
        failures.append("peer read/write not OK for every pair")

    if args.load_test:
        try:
            lt = load_test(args.load_seconds, args.gpus)
            report["load_test"] = lt
            if any(lt["exit_codes"]):
                failures.append(f"load test exit codes {lt['exit_codes']}")
            per_gpu = {}
            for sample in lt["samples"]:
                for row in sample:
                    per_gpu.setdefault(row[0], []).append(number(row[1]))
            for gpu, clocks in sorted(per_gpu.items()):
                clocks = sorted(c for c in clocks if c is not None)
                if clocks and clocks[len(clocks) // 2] < args.min_load_sm_mhz:
                    failures.append(f"GPU{gpu} median SM clock under load {clocks[len(clocks) // 2]:.0f} MHz")
        except Exception as e:  # report, never hide
            unchecked.append(f"load test could not run: {e}")
    for name, path, extra in (("p2p_bandwidth", args.p2p_bin, []), ("nccl_alltoall", args.nccl_alltoall,
                                                                     ["-b", "8", "-e", "256M", "-f", "2", "-g", str(args.gpus)])):
        if path:
            r = subprocess.run([str(path), *extra], capture_output=True, text=True)
            report[name] = dict(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
            if r.returncode:
                failures.append(f"{name} exited {r.returncode}")

    report["failures"], report["unchecked"] = failures, unchecked
    report["verdict"] = "FAIL" if failures else ("UNCHECKED" if unchecked else "PASS")
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(report["verdict"])
    for line in failures + unchecked:
        print(" -", line)
    return {"PASS": 0, "FAIL": 2, "UNCHECKED": 3}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
