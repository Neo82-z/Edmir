from __future__ import annotations

import json

from ..model import Program
from ..target import Target
from ..verify import verify_program
from .base import BackendArtifact
from .json_backend import JsonBackend


def _cpp_string(value: str) -> str:
    return json.dumps(value)


DTYPE_ENUM = {
    "float16": "DType::kFloat16",
    "bfloat16": "DType::kBFloat16",
    "float32": "DType::kFloat32",
}
BUFFER_ROLE_ENUM = {
    "input": "BufferRole::kInput",
    "weight": "BufferRole::kWeight",
    "staging": "BufferRole::kStaging",
    "output": "BufferRole::kOutput",
    "token": "BufferRole::kToken",
}
OP_KIND_ENUM = {
    "transfer": "OpKind::kTransfer",
    "gemm": "OpKind::kGemm",
    "barrier": "OpKind::kBarrier",
    "fused_transfer_gemm": "OpKind::kFusedTransferGemm",
}
QUEUE_KIND_ENUM = {
    "comm": "QueueKind::kComm",
    "compute": "QueueKind::kCompute",
    "control": "QueueKind::kControl",
}
DEPENDENCY_KIND_ENUM = {
    "queue": "DependencyKind::kQueue",
    "read_after_write": "DependencyKind::kReadAfterWrite",
    "buffer_reuse": "DependencyKind::kBufferReuse",
    "write_after_write": "DependencyKind::kWriteAfterWrite",
}
GEMM_PROVIDER_ENUM = {
    "cublas": "GemmProvider::kCuBLAS",
    "cutlass": "GemmProvider::kCUTLASS",
    "cuda_reference": "GemmProvider::kCudaReference",
}
GEMM_KERNEL_ENUM = {
    "library_default": "GemmKernel::kLibraryDefault",
    "sm80_128x128x32": "GemmKernel::kSm80_128x128x32",
    "sm80_64x128x32": "GemmKernel::kSm80_64x128x32",
    "sm80_128x64x32": "GemmKernel::kSm80_128x64x32",
    "lsa_naive_f16": "GemmKernel::kLsaNaiveF16",
}
COMMUNICATION_API_ENUM = {
    "host_nccl": "CommunicationApi::kHostNCCL",
    "nccl_device": "CommunicationApi::kNCCLDevice",
}


class CppManifestBackend:
    """Lower a Program to a dependency-complete, header-only C++ descriptor.

    This is deliberately a runtime boundary, not fake CUDA code generation.
    A CUDA/NCCL/CUTLASS runtime can interpret these static descriptors without
    reimplementing compiler dependency analysis.
    """

    name = "cpp"

    def emit(self, program: Program, target: Target) -> BackendArtifact:
        target.validate_program(program)
        has_fused = any(op.kind == "fused_transfer_gemm" for op in program.ops)
        if has_fused and target.gemm_provider != "cuda_reference":
            raise ValueError(
                "C++ fused runtime currently requires gemm_provider=cuda_reference; "
                "CUTLASS device fusion is not implemented"
            )
        if not has_fused and target.communication_api != "host_nccl":
            raise ValueError("C++ runtime v0.1 implements only communication_api=host_nccl")
        target = target.resolve(program.problem, gemm_m=program.chunk_rows)
        if target.world_size != program.problem.world_size:
            raise ValueError("target world size does not match Program")
        report = verify_program(program.problem, program)
        report.require_valid()

        buffer_index = {buffer.name: index for index, buffer in enumerate(program.buffers)}
        op_index = {op.id: index for index, op in enumerate(program.ops)}
        accesses = [
            (op_index[op.id], buffer_index[name], write)
            for op in program.ops
            for write, names in ((False, op.reads), (True, op.writes))
            for name in names
        ]
        dependencies = [
            (
                op_index[dependency.src],
                op_index[dependency.dst],
                dependency.kind,
                -1 if dependency.buffer is None else buffer_index[dependency.buffer],
            )
            for dependency in report.dependence.dependencies
        ]

        lines = [
            "#pragma once",
            "",
            "#include <array>",
            "#include <string_view>",
            "",
            '#include "edmir/runtime/program.hpp"',
            "",
            "namespace edmir_generated {",
            "",
            "using namespace edmir::runtime;",
            "",
            f"inline constexpr std::string_view kTarget = {_cpp_string(target.name)};",
            f"inline constexpr std::string_view kArch = {_cpp_string(target.arch)};",
            "inline constexpr GemmProvider kGemmProvider = "
            f"{GEMM_PROVIDER_ENUM[target.gemm_provider]};",
            "inline constexpr GemmKernel kGemmKernel = "
            f"{GEMM_KERNEL_ENUM[target.gemm_kernel]};",
            "inline constexpr CommunicationApi kCommunicationApi = "
            f"{COMMUNICATION_API_ENUM[target.communication_api]};",
            f"inline constexpr std::string_view kStrategy = {_cpp_string(program.strategy)};",
            "inline constexpr ProblemDesc kProblem{"
            f"{program.problem.world_size}, {program.problem.rows_per_peer}, "
            f"{program.problem.k}, {program.problem.n}, {DTYPE_ENUM[program.problem.dtype]}"
            "};",
            f"inline constexpr std::int32_t kChunkRows = {program.chunk_rows};",
            f"inline constexpr std::int32_t kBufferCount = {program.buffer_count};",
            "",
            f"inline constexpr std::array<BufferDesc, {len(program.buffers)}> kBuffers{{{{",
        ]
        lines.extend(
            "  {"
            f"{_cpp_string(buffer.name)}, {buffer.rank}, {BUFFER_ROLE_ENUM[buffer.role]}, "
            f"{buffer.size_bytes}ull"
            "},"
            for buffer in program.buffers
        )
        lines.extend(
            [
                "}};",
                "",
                f"inline constexpr std::array<OpDesc, {len(program.ops)}> kOps{{{{",
            ]
        )
        lines.extend(
            "  {"
            f"{_cpp_string(op.id)}, {OP_KIND_ENUM[op.kind]}, {op.rank}, "
            f"{-1 if op.peer_rank is None else op.peer_rank}, {QUEUE_KIND_ENUM[op.queue]}, "
            f"{_cpp_string(op.tile or '')}, {op.rows}"
            "},"
            for op in program.ops
        )
        lines.extend(
            [
                "}};",
                "",
                f"inline constexpr std::array<AccessDesc, {len(accesses)}> kAccesses{{{{",
            ]
        )
        lines.extend(
            f"  {{{op}, {buffer}, {'true' if write else 'false'}}},"
            for op, buffer, write in accesses
        )
        lines.extend(
            [
                "}};",
                "",
                f"inline constexpr std::array<DependencyDesc, {len(dependencies)}> "
                "kDependencies{{",
            ]
        )
        lines.extend(
            f"  {{{src}, {dst}, {DEPENDENCY_KIND_ENUM[kind]}, {buffer}}},"
            for src, dst, kind, buffer in dependencies
        )
        lines.extend(
            [
                "}};",
                "",
                "inline constexpr ProgramView program() {",
                "  return {",
                "      kTarget, kArch, kGemmProvider, kGemmKernel, kCommunicationApi, "
                "kStrategy, kProblem, "
                "kChunkRows, kBufferCount,",
                "      std::span<const BufferDesc>{kBuffers},",
                "      std::span<const OpDesc>{kOps},",
                "      std::span<const AccessDesc>{kAccesses},",
                "      std::span<const DependencyDesc>{kDependencies},",
                "  };",
                "}",
                "",
                "}  // namespace edmir_generated",
                "",
            ]
        )

        files = dict(JsonBackend().emit(program, target).files)
        files["edmir_program.hpp"] = "\n".join(lines)
        return BackendArtifact(
            backend=self.name,
            files=files,
            metadata={
                "target": target.name,
                "arch": target.arch,
                "gemm_provider": target.gemm_provider,
                "gemm_kernel": target.gemm_kernel,
                "communication_api": target.communication_api,
                "buffers": len(program.buffers),
                "ops": len(program.ops),
                "dependencies": len(dependencies),
            },
        )


__all__ = ["CppManifestBackend"]
