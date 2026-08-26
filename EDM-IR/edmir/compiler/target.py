from __future__ import annotations

from dataclasses import dataclass

from .model import TopologySnapshot


SCHEMA_TARGET = "edmir.target.v1"


@dataclass(frozen=True, slots=True)
class Target:
    """Backend-independent target contract, similar to an operation target in CUTLASS."""

    name: str
    topology: TopologySnapshot
    kind: str = "cuda"
    arch: str = "unknown"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("target name must not be empty")
        if self.kind != "cuda":
            raise ValueError("v0.1 supports only CUDA targets")
        self.topology.validate()

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_TARGET,
            "name": self.name,
            "kind": self.kind,
            "arch": self.arch,
            "topology": self.topology.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Target":
        schema = value.get("schema")
        if schema not in (None, SCHEMA_TARGET):
            raise ValueError(f"unsupported schema {schema!r}; expected {SCHEMA_TARGET!r}")
        target = cls(
            name=str(value["name"]),
            kind=str(value.get("kind", "cuda")),
            arch=str(value.get("arch", "unknown")),
            topology=TopologySnapshot.from_dict(value["topology"]),
        )
        target.validate()
        return target


__all__ = ["SCHEMA_TARGET", "Target"]
