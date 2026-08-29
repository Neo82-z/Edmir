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
    "sm80_64x128x32_splitk2",
    "sm80_64x128x32_splitk4",
    "sm80_64x128x32_splitk8",
    "sm80_64x128x32_splitk16",
    "sm80_128x64x32",
    "lsa_naive_f16",
)
GEMM_KERNEL_THREADBLOCK_N = {
    "sm80_128x128x32": 128,
    "sm80_64x128x32": 128,
    "sm80_64x128x32_splitk2": 128,
    "sm80_64x128x32_splitk4": 128,
    "sm80_64x128x32_splitk8": 128,
    "sm80_64x128x32_splitk16": 128,
    "sm80_128x64x32": 64,
}
GEMM_KERNEL_THREADBLOCK_M = {
    "sm80_128x128x32": 128,
    "sm80_64x128x32": 64,
    "sm80_64x128x32_splitk2": 64,
    "sm80_64x128x32_splitk4": 64,
    "sm80_64x128x32_splitk8": 64,
    "sm80_64x128x32_splitk16": 64,
    "sm80_128x64x32": 128,
}
GEMM_KERNEL_SPLIT_K = {
    "sm80_64x128x32_splitk2": 2,
    "sm80_64x128x32_splitk4": 4,
    "sm80_64x128x32_splitk8": 8,
    "sm80_64x128x32_splitk16": 16,
}


@dataclass(frozen=True, slots=True)
class FusedTrafficEstimate:
    remote_ops: int
    remote_input_bytes: int
    remote_a_replay: int
    estimated_remote_a_load_bytes: int
    split_k_slices: int
    estimated_ctas_per_op: int

    def to_dict(self) -> dict[str, int]:
        return {
            "remote_ops": self.remote_ops,
            "remote_input_bytes": self.remote_input_bytes,
            "remote_a_replay": self.remote_a_replay,
            "estimated_remote_a_load_bytes": self.estimated_remote_a_load_bytes,
            "split_k_slices": self.split_k_slices,
            "estimated_ctas_per_op": self.estimated_ctas_per_op,
        }


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
            "sm80_64x128x32_splitk2",
            "sm80_64x128x32_splitk4",
            "sm80_64x128x32_splitk8",
            "sm80_64x128x32_splitk16",
            "sm80_128x64x32",
        ):
            raise ValueError("CUTLASS targets require gemm_kernel=auto or an SM80 kernel name")
        if (
            self.gemm_kernel in GEMM_KERNEL_SPLIT_K
            and self.communication_api != "nccl_device"
        ):
            raise ValueError("CUTLASS split-K is a fused NCCL Device API kernel")
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
        split_k_slices = GEMM_KERNEL_SPLIT_K.get(self.gemm_kernel, 1)
        split_k_alignment = 32 * split_k_slices
        if split_k_slices > 1 and problem.k % split_k_alignment != 0:
            raise ValueError(
                f"CUTLASS split-K={split_k_slices} kernel requires K divisible by "
                f"{split_k_alignment}"
            )

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

    def estimate_fused_traffic(self, program: Program) -> FusedTrafficEstimate:
        """Expose static remote-A traffic features without pretending to predict latency."""

        self.validate_program(program)
        fused_ops = tuple(op for op in program.ops if op.kind == "fused_transfer_gemm")
        if not fused_ops or len(fused_ops) != len(program.ops):
            raise ValueError("fused traffic estimation requires an all-fused Program")
        resolved = self.resolve(program.problem, gemm_m=program.chunk_rows)
        remote_ops = tuple(op for op in fused_ops if op.peer_rank != op.rank)
        remote_input_bytes = sum(
            op.rows * program.problem.bytes_per_row for op in remote_ops
        )
        if resolved.gemm_provider == "cuda_reference":
            replay = program.problem.n
            split_k_slices = 1
            ctas_per_op = (
                (program.chunk_rows + 15) // 16
                * ((program.problem.n + 15) // 16)
            )
        else:
            tile_n = GEMM_KERNEL_THREADBLOCK_N.get(resolved.gemm_kernel)
            tile_m = GEMM_KERNEL_THREADBLOCK_M.get(resolved.gemm_kernel)
            if tile_n is None or tile_m is None:
                raise ValueError(
                    f"CUTLASS kernel {resolved.gemm_kernel!r} has no traffic tile contract"
                )
            replay = (program.problem.n + tile_n - 1) // tile_n
            split_k_slices = GEMM_KERNEL_SPLIT_K.get(resolved.gemm_kernel, 1)
            ctas_per_op = (
                (program.chunk_rows + tile_m - 1) // tile_m
                * replay
                * split_k_slices
            )
        return FusedTrafficEstimate(
            remote_ops=len(remote_ops),
            remote_input_bytes=remote_input_bytes,
            remote_a_replay=replay,
            estimated_remote_a_load_bytes=remote_input_bytes * replay,
            split_k_slices=split_k_slices,
            estimated_ctas_per_op=ctas_per_op,
        )

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
    "FusedTrafficEstimate",
    "GEMM_KERNELS",
    "GEMM_KERNEL_SPLIT_K",
    "GEMM_KERNEL_THREADBLOCK_M",
    "GEMM_KERNEL_THREADBLOCK_N",
    "GEMM_PROVIDERS",
    "SCHEMA_TARGET",
    "Target",
]
