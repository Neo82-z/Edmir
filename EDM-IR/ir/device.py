from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Iterator, Generator, TYPE_CHECKING
import importlib, inspect, functools, pathlib, os, platform, contextlib, sys, re, atexit, pickle, decimal


@dataclass(frozen=True, slots=True)
class DeviceDesc:
    name: str
    vendor: str
    arch: str | None = None
    hbm_bytes: int | None = None
    
    compute_capability: tuple[int, int] | None = None
    sm_count: int | None = None
    hbm_bandwidth_gbps: float | None = None


    supports_tensor_cores: bool = False
    supports_tma: bool = False
    supports_nvlink: bool = False
    supports_rdma: bool = False
    supports_wgmma: bool = False
    supports_fp8: bool = False


    nvlink_bandwidth_gbps: float | None = None
    pcie_bandwidth_gbps: float | None = None


DEVICES = ["METAL", "NV", "CUDA"]
class Device:
    def __init__(self) -> None:
        self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
        self._opened_devices:set[str] = set()

        # tinygrad
        self._load_device_descs()

@proprty
def devices(self) -> list[str]:
    return self._devices

def DEFAULT(self) -> str: return DEV.device or self._select_device

@DEFAULT.setter
def DEFAULT(self, v): raise AttributeError(f'setting Device.DEFAULT is deprecated, use "with Context(DEV={v!r})" or "DEV.value = {v!r}"')



class Allocator(Generic[DeviceType]):
    def __init__(self, dev: DeviceType, supports_copy_from_disk: bool = True, supports_transfer: bool = True):
        
class LRUallocator(Allocator[DeviceType]):
    def __init__(self, dev: DeviceType, supports_copy_from_disk: bool = True, supports_transfer: bool = True):
        super().__init__(dev, supports_copy_from_disk, supports_transfer)
        self._cache: dict[str, Any] = {}
        self._lru_order: list[str] = []

    def allocate(self, key: str, size: int) -> Any:
        if key in self._cache:
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return self._cache[key]
        
        if len(self._cache) >= self.max_size:
            oldest_key = self._lru_order.pop(0)
            del self._cache[oldest_key]
        
        resource = self.dev.allocate_resource(size)
        self._cache[key] = resource
        self._lru_order.append(key)
        return resource

    def free(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._lru_order.remove(key)

    @property
    def max_size(self) -> int:
        return 10  # Example fixed size for the cache