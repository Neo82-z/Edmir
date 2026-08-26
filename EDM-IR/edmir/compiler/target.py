from __future__ import annotations

from dataclasses import dataclass, replace

from .model import A2AGemmProblem, CalibrationProfile, Program, TopologySnapshot


SCHEMA_TARGET = "edmir.target.v1"
GEMM_PROVIDERS = ("cublas", "cutlass", "cuda_reference")
COMMUNICATION_APIS = ("host_nccl", "nccl_device")
GEMM_KERNELS = (
    "auto",
    "library_default",
    "sm80_128x128x32",
    "sm80_64x128x32",
    "sm80_128x64x32",
    "lsa_naive_f16",
)


@dataclass(frozen=True, slots=True)
class Target:
    """Backend-independent target contract, similar to an operation target in CUTLASS."""

    name: str
    topology: TopologySnapshot
    kind: str = "cuda"
    arch: str = "unknown"
    gemm_provider: str = "cublas"
    gemm_kernel: str = "auto"
    communication_api: str = "host_nccl"

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
        if self.communication_api not in COMMUNICATION_APIS:
            raise ValueError(
                f"unsupported communication API {self.communication_api!r}; "
                f"expected one of {COMMUNICATION_APIS}"
            )
        if self.gemm_kernel not in GEMM_KERNELS:
            raise ValueError(
                f"unsupported GEMM kernel {self.gemm_kernel!r}; expected one of {GEMM_KERNELS}"
            )
        if self.gemm_provider == "cublas" and self.gemm_kernel not in (
            "auto",
            "library_default",
        ):
            raise ValueError("cuBLAS targets require gemm_kernel=auto or library_default")
        if self.gemm_provider == "cutlass" and self.gemm_kernel not in (
            "auto",
            "sm80_128x128x32",
            "sm80_64x128x32",
            "sm80_128x64x32",
        ):
            raise ValueError("CUTLASS targets require gemm_kernel=auto or an SM80 kernel name")
        if self.gemm_provider == "cuda_reference" and self.gemm_kernel not in (
            "auto",
            "lsa_naive_f16",
        ):
            raise ValueError(
                "CUDA reference targets require gemm_kernel=auto or lsa_naive_f16"
            )
        self.topology.validate()

    def validate_problem(self, problem: A2AGemmProblem) -> None:
        """Reject provider/shape combinations outside the implemented runtime contract."""

        self.validate()
        problem.validate()
        if self.gemm_provider == "cublas":
            return
        if self.gemm_provider == "cuda_reference":
            if self.arch != "sm_80":
                raise ValueError("CUDA reference fused runtime v0.1 supports only arch=sm_80")
            if problem.dtype != "float16":
                raise ValueError("CUDA reference fused runtime v0.1 supports only float16")
            if problem.layout != "row_major":
                raise ValueError("CUDA reference fused runtime v0.1 supports only row_major")
            return
        if self.arch != "sm_80":
            raise ValueError("CUTLASS v0.1 supports only arch=sm_80")
        if problem.dtype != "float16":
            raise ValueError("CUTLASS v0.1 supports only float16 GEMM")
        if problem.layout != "row_major":
            raise ValueError("CUTLASS v0.1 supports only row_major GEMM")
        if problem.k % 8 != 0 or problem.n % 8 != 0:
            raise ValueError("CUTLASS v0.1 requires K and N to be divisible by 8")

    def validate_profile(self, profile: CalibrationProfile) -> None:
        self.validate()
        profile.validate()
        if profile.gemm_provider != self.gemm_provider:
            raise ValueError(
                "calibration GEMM provider does not match target: "
                f"profile={profile.gemm_provider}, target={self.gemm_provider}"
            )
        if self.gemm_kernel != "auto" and profile.gemm_kernel != self.gemm_kernel:
            raise ValueError(
                "calibration GEMM kernel does not match explicit target kernel: "
                f"profile={profile.gemm_kernel}, target={self.gemm_kernel}"
            )
        if self.gemm_kernel == "auto" and profile.gemm_kernel not in (
            "auto",
            "library_default",
        ):
            raise ValueError(
                "auto target requires an auto-policy or library-default calibration profile"
            )

    @property
    def supports_fused_a2a_gemm(self) -> bool:
        return (
            self.communication_api == "nccl_device"
            and self.gemm_provider in {"cutlass", "cuda_reference"}
            and self.topology.has_full_p2p()
        )

    def validate_program(self, program: Program) -> None:
        """Check target capabilities required by an already-lowered Program."""

        self.validate_problem(program.problem)
        if self.world_size != program.problem.world_size:
            raise ValueError("target world size does not match Program")
        has_fused = any(op.kind == "fused_transfer_gemm" for op in program.ops)
        if has_fused and self.communication_api != "nccl_device":
            raise ValueError(
                "fused_transfer_gemm requires communication_api=nccl_device; "
                "use the unfused host NCCL schedule on this target"
            )
        if has_fused and self.gemm_provider not in {"cutlass", "cuda_reference"}:
            raise ValueError(
                "fused_transfer_gemm requires gemm_provider=cutlass or cuda_reference; "
                "cuBLAS is an external library launch, not an in-kernel GEMM"
            )
        if has_fused and not self.topology.has_full_p2p():
            raise ValueError("fused_transfer_gemm requires full peer-to-peer connectivity")

    def resolve(self, problem: A2AGemmProblem, *, gemm_m: int | None = None) -> "Target":
        """Materialize backend policy so emitted artifacts never contain `auto`."""

        self.validate_problem(problem)
        if self.gemm_kernel != "auto":
            return self
        if self.gemm_provider == "cublas":
            return replace(self, gemm_kernel="library_default")
        if self.gemm_provider == "cuda_reference":
            return replace(self, gemm_kernel="lsa_naive_f16")
        effective_m = problem.rows_per_peer if gemm_m is None else gemm_m
        if effective_m <= 96:
            return replace(self, gemm_kernel="sm80_64x128x32")
        return replace(self, gemm_kernel="sm80_128x128x32")

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
            "gemm_kernel": self.gemm_kernel,
            "communication_api": self.communication_api,
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
            gemm_kernel=str(value.get("gemm_kernel", "auto")),
            communication_api=str(value.get("communication_api", "host_nccl")),
            topology=TopologySnapshot.from_dict(value["topology"]),
        )
        target.validate()
        return target


__all__ = [
    "COMMUNICATION_APIS",
    "GEMM_KERNELS",
    "GEMM_PROVIDERS",
    "SCHEMA_TARGET",
    "Target",
]
