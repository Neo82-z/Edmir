from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir import EDMBuilder, MovementDesc, detect_device, format_device_spec, known_device_specs  # noqa: E402
from ir.ops import COMM_ALL_GATHER, COMPUTE_QKV, DATA_DEPENDENCY, STREAM_ORDER  # noqa: E402
from ir.printing import format_summary, format_uops  # noqa: E402


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=5.0).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def main() -> None:
    print("== EDM-IR known device specs ==")
    for spec in known_device_specs():
        print(format_device_spec(spec))

    print("\n== local CUDA devices ==")
    smi = _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    print("nvidia-smi:", smi)

    count = 1
    if smi and not smi.startswith("<unavailable"):
        count = len([line for line in smi.splitlines() if line.strip()])

    for index in range(count):
        spec = detect_device(index)
        print(f"cuda:{index}", format_device_spec(spec))
        if spec is not None:
            print_smoke_uops(index, spec.cuda_arch)

    print("\n== topology ==")
    print(_run(["nvidia-smi", "topo", "-m"]))


def print_smoke_uops(index: int, cuda_arch: str | None) -> None:
    device = f"cuda:{index}"
    if cuda_arch:
        device = f"{device}/{cuda_arch}"

    builder = EDMBuilder(default_phase="device_probe")
    begin = builder.phase_begin("device_probe", 0.0)
    qkv = builder.compute(
        "probe_qkv",
        0.0,
        10.0,
        op=COMPUTE_QKV,
        device=device,
        stream="compute",
    )
    comm = builder.comm(
        "probe_all_gather",
        10.0,
        14.0,
        op=COMM_ALL_GATHER,
        device=device,
        stream="comm",
        movement=MovementDesc(kind="collective", backend="nccl", size_bytes=16 << 20, group="tp"),
    )
    end = builder.phase_end("device_probe", 14.0)
    builder.sequence(begin, qkv, kind=STREAM_ORDER)
    builder.edge(qkv, comm, DATA_DEPENDENCY, data_id="probe_qkv")
    builder.sequence(comm, end, kind=STREAM_ORDER)

    graph = builder.build()
    print("\nsmoke_uops:")
    print(format_uops(graph))
    print(format_summary(graph))


if __name__ == "__main__":
    main()
