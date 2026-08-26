from __future__ import annotations

from dataclasses import dataclass

from .model import A2AGemmProblem, TopologySnapshot


SCHEMA_TARGET = "edmir.target.v1"
GEMM_PROVIDERS = ("cublas", "cutlass")


@dataclass(frozen=True, slots=True)
class Target:
    """Backend-independent target contract, similar to an operation target in CUTLASS."""

    name: str
    topology: TopologySnapshot
    kind: str = "cuda"
    arch: str = "unknown"
    gemm_provider: str = "cublas"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("target name must not be empty")
        if self.kind != "cuda":
            raise ValueError("v0.1 supports only CUDA targets")
        if self.gemm_provider not in GEMM_PROVIDERS:
            raise ValueError(
                f"unsupported GEMM provider {self.gemm_provider!r}; "
                f"expected one of {GEMM_PROVIDERS}"
            )
        self.topology.validate()

    def validate_problem(self, problem: A2AGemmProblem) -> None:
        """Reject provider/shape combinations outside the implemented runtime contract."""

        self.validate()
        problem.validate()
        if self.gemm_provider != "cutlass":
            return
        if self.arch != "sm_80":
            raise ValueError("CUTLASS v0.1 supports only arch=sm_80")
        if problem.dtype != "float16":
            raise ValueError("CUTLASS v0.1 supports only float16 GEMM")
        if problem.layout != "row_major":
            raise ValueError("CUTLASS v0.1 supports only row_major GEMM")
        if problem.k % 8 != 0 or problem.n % 8 != 0:
            raise ValueError("CUTLASS v0.1 requires K and N to be divisible by 8")

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_TARGET,
            "name": self.name,
            "kind": self.kind,
            "arch": self.arch,
            "gemm_provider": self.gemm_provider,
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
            gemm_provider=str(value.get("gemm_provider", "cublas")),
            topology=TopologySnapshot.from_dict(value["topology"]),
        )
        target.validate()
        return target


__all__ = ["GEMM_PROVIDERS", "SCHEMA_TARGET", "Target"]
