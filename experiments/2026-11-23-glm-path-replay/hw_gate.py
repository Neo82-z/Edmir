#!/usr/bin/env python3
"""Read-only host/Fabric Manager and locked-image gate for the GLM experiment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent/"2026-11-20-final-round/scripts/hw_gate.py"
HOST_SCHEMA = "edmir.glm-host-gate.v1"
GATE_SCHEMA = "edmir.glm-hardware-gate.v1"
WORLD = 8
MIN_DRIVER = (580, 65, 6)
LOCKED_RUNTIME = {"torch": "2.11.0+cu130", "cuda": "13.0", "vllm": "0.20.1"}


def run(*cmd, timeout=20):
    try:
        proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                text=True,start_new_session=True)
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    with proc:
        try:
            stdout,stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout,stderr = proc.communicate()
            return {"returncode": None, "stdout": stdout.strip(), "stderr": f"timed out: {stderr.strip()}"}
    return {"returncode": proc.returncode, "stdout": stdout.strip(), "stderr": stderr.strip()}


def version(value):
    if isinstance(value,str) and re.fullmatch(r"\d+\.\d+\.\d+",value):
        return tuple(map(int,value.split(".")))
    return ()


def inventory():
    query = run("nvidia-smi","--query-gpu=uuid,driver_version","--format=csv,noheader,nounits")
    if query["returncode"]!=0:
        raise ValueError("GPU inventory unavailable: "+query["stderr"])
    parts = [tuple(s.strip() for s in line.split(",")) for line in query["stdout"].splitlines()]
    if (len(parts)!=WORLD or any(len(row)!=2 or not row[0].startswith("GPU-") for row in parts)
            or len({row[0] for row in parts})!=WORLD or len({row[1] for row in parts})!=1):
        raise ValueError("eight distinct GPU UUIDs and a common full driver version required")
    return [u for u,_ in parts],parts[0][1]


def check_fm(fm, uuids, guest_driver):
    if fm.get("source") not in {"local_service","provider_host"}:
        raise ValueError("local or provider host Fabric Manager evidence required")
    reported = fm.get("gpu_uuids",[])
    if len(reported)!=WORLD or set(reported)!=set(uuids):
        raise ValueError("Fabric Manager evidence refers to different GPU UUIDs")
    if fm.get("active")!="active" or fm.get("substate")!="running":
        raise ValueError("Fabric Manager service is not active/running on the host")
    host_driver = version(fm.get("host_driver"))
    manager = version(fm.get("fm_version"))
    if not host_driver or not manager:
        raise ValueError("full host driver and Fabric Manager versions required")
    if host_driver[0]>=620:
        if fm["source"]!="provider_host" or version(fm.get("fm_library_version"))!=manager:
            raise ValueError("R620+ requires host evidence that the fabric driver library matches standalone FM")
    elif host_driver!=manager:
        raise ValueError("Fabric Manager and host driver versions differ")
    if fm["source"]=="local_service" and fm["host_driver"]!=guest_driver:
        raise ValueError("local Fabric Manager evidence refers to another driver")
    if fm["source"]=="provider_host" and not fm.get("reference","").strip():
        raise ValueError("provider-host evidence requires an attributable support reference")


def check_host(host):
    if host.get("schema")!=HOST_SCHEMA or host.get("verdict")!="PASS":
        raise ValueError("passing host driver and Fabric Manager receipt required")
    uuids = host.get("gpu_uuids",[])
    if len(uuids)!=WORLD or len(set(uuids))!=WORLD or any(not u.startswith("GPU-") for u in uuids):
        raise ValueError("host receipt has an invalid GPU inventory")
    if version(host.get("driver"))<MIN_DRIVER:
        raise ValueError("Linux driver below 580.65.06 or full version unavailable")
    check_fm(host.get("fabric_manager",{}),uuids,host["driver"])


def host_receipt(provider_path):
    report = dict(schema=HOST_SCHEMA,verdict="UNCHECKED")
    try:
        uuids,driver = inventory()
        report.update(gpu_uuids=uuids,driver=driver)
        if provider_path:
            obj = json.loads(provider_path.read_text())
            if obj.get("source")!="provider_host":
                raise ValueError("expected a provider_host receipt")
        else:
            raw = run("systemctl","show","nvidia-fabricmanager.service","-p","ActiveState","-p","SubState")
            binary = run("nv-fabricmanager","-v")
            report.update(service_probe=raw,version_probe=binary)
            service = dict(line.split("=",1) for line in raw["stdout"].splitlines() if "=" in line)
            match = re.search(r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])",binary["stdout"]+" "+binary["stderr"])
            obj = dict(source="local_service",gpu_uuids=uuids,host_driver=driver,
                       fm_version=match[1] if match and binary["returncode"]==0 else None,
                       active=service.get("ActiveState") if raw["returncode"]==0 else None,
                       substate=service.get("SubState") if raw["returncode"]==0 else None)
        report["fabric_manager"] = obj
        report["verdict"] = "PASS"
        check_host(report)
    except (OSError,ValueError,TypeError) as exc:
        report.update(verdict="UNCHECKED",reason=str(exc))
    return report


def read_fabric(text):
    statuses = re.findall(r"^\s*Fabric\s*\n\s*State\s*:\s*([^\n]+)\n\s*Status\s*:\s*([^\n]+)",
                          text,re.MULTILINE)
    return [(state.strip(),status.strip()) for state,status in statuses]


def check_links(text, off_diagonal):
    rows = re.findall(r"^\s*GPU(\d+)\s+([^\n]+)",text,re.MULTILINE)
    data = [(int(i),row.split()[:WORLD]) for i,row in rows if not row.lstrip().startswith("GPU")]
    if sorted(i for i,_ in data)!=list(range(WORLD)):
        raise ValueError("topology matrix lacks eight distinct GPU rows")
    for rank,row in data:
        if len(row)!=WORLD or any(cell!=("X" if rank==peer else off_diagonal) for peer,cell in enumerate(row)):
            raise ValueError(f"GPU{rank} lacks a complete {off_diagonal} peer matrix")


def check_gate(gate):
    """Reject historical topology-only PASS receipts in the current experiment."""
    if gate.get("schema")!=GATE_SCHEMA or gate.get("verdict")!="PASS":
        raise ValueError("passing hardware gate required (host and locked-image receipts)")
    host = gate.get("host",{})
    check_host(host)
    if (gate.get("gpu_uuids")!=host["gpu_uuids"] or gate.get("driver")!=host["driver"]
            or gate.get("runtime")!=LOCKED_RUNTIME):
        raise ValueError("host and locked-image inventory or runtime do not match")
    legacy = gate.get("a100_checks",{})
    if legacy.get("verdict")!="PASS" or legacy.get("failures") or legacy.get("unchecked"):
        raise ValueError("A100 topology, power and load gate did not pass")
    for key,link in (("topology","NV12"),("p2p_read","OK"),("p2p_write","OK")):
        check_links(legacy.get(key,""),link)
    load = legacy.get("load_test",{})
    if load.get("exit_codes")!=[0]*WORLD or not load.get("samples"):
        raise ValueError("eight successful BF16 load checks with telemetry required")
    fabric = gate.get("fabric",[])
    if len(fabric) not in (0,WORLD):
        raise ValueError("incomplete fabric inventory")
    for state,status in fabric:
        if (state,status) not in {("N/A","N/A"),("Completed","Success")}:
            raise ValueError(f"unexpected fabric state: {state}/{status}")


def load_passing_gate(path):
    gate = json.loads(path.read_text())
    check_gate(gate)
    return gate


def runtime_receipt(a):
    report = dict(schema=GATE_SCHEMA,verdict="UNCHECKED")
    try:
        report["host"] = json.loads(a.host_receipt.read_text())
        check_host(report["host"])
        uuids,driver = inventory()
        report.update(gpu_uuids=uuids,driver=driver)
        import torch

        report["runtime"] = dict(torch=torch.__version__,cuda=torch.version.cuda,
                                 vllm=importlib.metadata.version("vllm"))
        if (report["runtime"]!=LOCKED_RUNTIME or driver!=report["host"]["driver"]
                or uuids!=report["host"]["gpu_uuids"]):
            raise ValueError("wrong package versions or container GPUs/driver differ from the host receipt")
        fabric = run("nvidia-smi","-q")
        report["fabric"] = read_fabric(fabric["stdout"])
        if fabric["returncode"]!=0 or len(report["fabric"]) not in (0,WORLD):
            raise ValueError("incomplete fabric inventory")
        for state,status in report["fabric"]:
            if (state,status) not in {("N/A","N/A"),("Completed","Success")}:
                raise ValueError(f"unexpected fabric state: {state}/{status}")
        if not torch.cuda.is_available() or torch.cuda.device_count()!=WORLD:
            raise ValueError("locked CUDA runtime could not initialize all eight GPUs")
        for rank in range(WORLD):
            p = torch.cuda.get_device_properties(rank)
            if ("A100" not in p.name or "SXM" not in p.name or (p.major,p.minor)!=(8,0)
                    or p.total_memory<75*1024**3):
                raise ValueError(f"rank {rank} is not an A100 SXM4 80GB SM80")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"a100.json"
            result = run(sys.executable,str(LEGACY),"--out",str(path),"--load-test",timeout=180)
            report["load_command"] = result
            if not path.is_file():
                raise ValueError("A100 topology and BF16 load report missing")
            report["a100_checks"] = json.loads(path.read_text())
            if result["returncode"]!=0:
                raise ValueError("A100 topology, power or BF16 load gate did not pass")
        report["verdict"] = "PASS"
        check_gate(report)
    except (OSError,ValueError,TypeError,RuntimeError,ImportError) as exc:
        report.update(verdict="UNCHECKED",reason=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action",required=True)
    host = sub.add_parser("host")
    host.add_argument("--provider-receipt",type=Path)
    runtime = sub.add_parser("runtime")
    runtime.add_argument("--host-receipt",type=Path,required=True)
    for stage in (host,runtime):
        stage.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = host_receipt(args.provider_receipt) if args.action=="host" else runtime_receipt(args)
    with args.out.open("x") as output:
        json.dump(result,output,indent=2)
        output.write("\n")
    print(result["verdict"],result.get("reason",""))
    return 0 if result["verdict"]=="PASS" else 3


if __name__=="__main__":
    sys.exit(main())
