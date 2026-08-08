from __future__ import annotations

import subprocess

from .compiler import TopologySnapshot


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=10.0).strip()


def _parse_p2p_matrix(text: str, world_size: int) -> tuple[tuple[bool, ...], ...]:
    rows: dict[int, tuple[bool, ...]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("GPU"):
            continue
        try:
            rank = int(fields[0][3:])
        except ValueError:
            continue
        values = fields[1 : 1 + world_size]
        if len(values) != world_size:
            continue
        rows[rank] = tuple(value.upper() in {"OK", "X"} for value in values)
    if len(rows) != world_size:
        raise ValueError("could not parse nvidia-smi P2P matrix")
    return tuple(rows[rank] for rank in range(world_size))


def probe_topology() -> TopologySnapshot:
    names_text = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    devices = tuple(line.strip() for line in names_text.splitlines() if line.strip())
    if not devices:
        raise RuntimeError("nvidia-smi reported no GPUs")
    p2p_text = _run(["nvidia-smi", "topo", "-p2p", "p"])
    raw_topology = _run(["nvidia-smi", "topo", "-m"])
    topology = TopologySnapshot(
        world_size=len(devices),
        devices=devices,
        p2p=_parse_p2p_matrix(p2p_text, len(devices)),
        raw_topology=raw_topology,
    )
    topology.validate()
    return topology


__all__ = ["probe_topology"]
