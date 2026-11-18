from .base import Backend, BackendArtifact
from .cpp_backend import CppManifestBackend
from .json_backend import JsonBackend


def create_backend(name: str) -> Backend:
    if name == "json":
        return JsonBackend()
    if name == "cpp":
        return CppManifestBackend()
    raise ValueError(f"unknown backend {name!r}")


__all__ = [
    "Backend",
    "BackendArtifact",
    "CppManifestBackend",
    "JsonBackend",
    "create_backend",
]
