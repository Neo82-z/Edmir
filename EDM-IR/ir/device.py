from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Iterable


def _norm_name(name: str) -> str:
    """Normalize device names for registry lookup and profiler labels."""

    lowered = name.lower()
    for token in ("nvidia", "geforce", "tesla", "gpu"):
        lowered = lowered.replace(token, "")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def arch_from_compute_capability(cc: tuple[int, int] | None) -> str | None:
    if cc is None:
        return None
    major, _minor = cc
    if major == 9:
        return "Hopper"
    if major in (10, 12):
        return "Blackwell"
    return None


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Static and measured properties needed by EDM-IR analysis.

    The spec is intentionally not a full hardware database.  Static capability
    fields describe which mechanisms may exist; measured fields are filled by
    calibration or runtime detection when available.
    """

    name: str
    vendor: str = "NVIDIA"
    arch: str | None = None
    compute_capability: tuple[int, int] | None = None

    sm_count: int | None = None
    memory_bytes: int | None = None
    memory_kind: str | None = None
    memory_bandwidth_gbps: float | None = None

    supports_tensor_cores: bool = True
    supports_tma: bool = False
    supports_nvlink: bool = False
    supports_rdma: bool = False
    supports_wgmma: bool = False
    supports_fp8: bool = False

    nvlink_bandwidth_gbps: float | None = None
    pcie_bandwidth_gbps: float | None = None

    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return _norm_name(self.name)

    @property
    def cuda_arch(self) -> str | None:
        if self.compute_capability is None:
            return None
        major, minor = self.compute_capability
        return f"sm{major}{minor}"

    @property
    def hbm_bytes(self) -> int | None:
        if self.memory_kind is None or not self.memory_kind.upper().startswith("HBM"):
            return None
        return self.memory_bytes

    @property
    def hbm_bandwidth_gbps(self) -> float | None:
        if self.memory_kind is None or not self.memory_kind.upper().startswith("HBM"):
            return None
        return self.memory_bandwidth_gbps

    def with_runtime(
        self,
        *,
        name: str | None = None,
        sm_count: int | None = None,
        memory_bytes: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> DeviceSpec:
        merged_attrs = dict(self.attrs)
        if attrs:
            merged_attrs.update(attrs)
        return replace(
            self,
            name=name or self.name,
            sm_count=sm_count if sm_count is not None else self.sm_count,
            memory_bytes=memory_bytes if memory_bytes is not None else self.memory_bytes,
            attrs=merged_attrs,
        )


# Backward-compatible alias for earlier sketches.
DeviceDesc = DeviceSpec


class DeviceRegistry:
    """Registry for static device specs plus loose profiler-name aliases."""

    def __init__(self) -> None:
        self._specs: dict[str, DeviceSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        spec: DeviceSpec,
        *,
        aliases: Iterable[str] = (),
        replace_existing: bool = False,
    ) -> DeviceSpec:
        key = spec.key
        if not replace_existing and key in self._specs:
            raise ValueError(f"device spec already registered: {spec.name}")

        self._specs[key] = spec
        for alias in (spec.name, *aliases):
            alias_key = _norm_name(alias)
            if not replace_existing and alias_key in self._aliases and self._aliases[alias_key] != key:
                raise ValueError(f"device alias already registered: {alias}")
            self._aliases[alias_key] = key
        return spec

    def get(self, name: str) -> DeviceSpec:
        spec = self.match(name)
        if spec is None:
            raise KeyError(f"unknown device: {name}")
        return spec

    def match(self, raw_name: str) -> DeviceSpec | None:
        query = _norm_name(raw_name)
        if query in self._aliases:
            return self._specs[self._aliases[query]]

        # Profiler/runtime names often append memory size, PCIe/SXM suffixes,
        # or SKU-specific decorations.  Longest aliases win for stable matches.
        for alias, key in sorted(self._aliases.items(), key=lambda item: len(item[0]), reverse=True):
            if alias and (alias in query or query in alias):
                return self._specs[key]
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values())

    def specs(self) -> tuple[DeviceSpec, ...]:
        return tuple(self._specs.values())

    def supported(self) -> tuple[DeviceSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.arch in {"Hopper", "Blackwell"})


def _register_builtin_devices(registry: DeviceRegistry) -> None:
    hopper = dict(
        arch="Hopper",
        compute_capability=(9, 0),
        supports_tensor_cores=True,
        supports_tma=True,
        supports_nvlink=True,
        supports_rdma=True,
        supports_wgmma=True,
        supports_fp8=True,
    )
    blackwell = dict(
        arch="Blackwell",
        supports_tensor_cores=True,
        supports_tma=True,
        supports_wgmma=True,
        supports_fp8=True,
    )

    registry.register(
        DeviceSpec("H100", memory_kind="HBM", **hopper),
        aliases=("NVIDIA H100", "H100 PCIe", "H100 SXM", "H100 NVL", "sm90"),
    )
    registry.register(
        DeviceSpec("H200", memory_kind="HBM", **hopper),
        aliases=("NVIDIA H200", "H200 SXM", "sm90 h200"),
    )
    registry.register(
        DeviceSpec("GH200", memory_kind="HBM", **hopper),
        aliases=("NVIDIA GH200", "Grace Hopper", "GH200 Grace Hopper"),
    )
    registry.register(
        DeviceSpec(
            "B200",
            compute_capability=(10, 0),
            memory_kind="HBM",
            supports_nvlink=True,
            supports_rdma=True,
            **blackwell,
        ),
        aliases=("NVIDIA B200", "Blackwell B200", "sm100"),
    )
    registry.register(
        DeviceSpec(
            "GB200",
            compute_capability=(10, 0),
            memory_kind="HBM",
            supports_nvlink=True,
            supports_rdma=True,
            **blackwell,
        ),
        aliases=("NVIDIA GB200", "Grace Blackwell", "GB200 Grace Blackwell"),
    )
    registry.register(
        DeviceSpec(
            "RTX 5090",
            compute_capability=(12, 0),
            memory_kind="GDDR",
            supports_nvlink=False,
            supports_rdma=False,
            **blackwell,
        ),
        aliases=("NVIDIA GeForce RTX 5090", "GeForce RTX 5090", "RTX5090", "sm120"),
    )


DEFAULT_REGISTRY = DeviceRegistry()
_register_builtin_devices(DEFAULT_REGISTRY)


def register_device_spec(
    spec: DeviceSpec,
    *,
    aliases: Iterable[str] = (),
    replace_existing: bool = False,
    registry: DeviceRegistry = DEFAULT_REGISTRY,
) -> DeviceSpec:
    return registry.register(spec, aliases=aliases, replace_existing=replace_existing)


def get_device_spec(name: str, *, registry: DeviceRegistry = DEFAULT_REGISTRY) -> DeviceSpec:
    return registry.get(name)


def match_device_spec(raw_name: str, *, registry: DeviceRegistry = DEFAULT_REGISTRY) -> DeviceSpec | None:
    return registry.match(raw_name)


def known_device_specs(*, registry: DeviceRegistry = DEFAULT_REGISTRY) -> tuple[DeviceSpec, ...]:
    return registry.specs()


def supported_nvidia_specs(*, registry: DeviceRegistry = DEFAULT_REGISTRY) -> tuple[DeviceSpec, ...]:
    return registry.supported()


def detect_device(index: int = 0, *, registry: DeviceRegistry = DEFAULT_REGISTRY) -> DeviceSpec | None:
    """Detect the local CUDA device if PyTorch or nvidia-smi is available."""

    spec = _detect_with_torch(index, registry)
    if spec is not None:
        return spec
    return _detect_with_nvidia_smi(index, registry)


def _detect_with_torch(index: int, registry: DeviceRegistry) -> DeviceSpec | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        cc = torch.cuda.get_device_capability(index)
    except Exception:
        return None

    base = registry.match(name) or DeviceSpec(
        name=name,
        arch=arch_from_compute_capability(cc),
        compute_capability=cc,
        supports_tensor_cores=cc[0] >= 7,
        supports_tma=cc[0] >= 9,
        supports_wgmma=cc[0] >= 9,
        supports_fp8=cc[0] >= 9,
    )
    return base.with_runtime(
        name=name,
        sm_count=getattr(props, "multi_processor_count", None),
        memory_bytes=getattr(props, "total_memory", None),
        attrs={"runtime": "torch", "cuda_index": index},
    )


def _detect_with_nvidia_smi(index: int, registry: DeviceRegistry) -> DeviceSpec | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    name = parts[0]
    memory_bytes = None
    if len(parts) > 1 and parts[1].isdigit():
        memory_bytes = int(parts[1]) * 1024 * 1024

    base = registry.match(name) or DeviceSpec(name=name)
    return base.with_runtime(
        name=name,
        memory_bytes=memory_bytes,
        attrs={"runtime": "nvidia-smi", "cuda_index": index},
    )


__all__ = [
    "DEFAULT_REGISTRY",
    "DeviceDesc",
    "DeviceRegistry",
    "DeviceSpec",
    "arch_from_compute_capability",
    "detect_device",
    "get_device_spec",
    "known_device_specs",
    "match_device_spec",
    "register_device_spec",
    "supported_nvidia_specs",
]
