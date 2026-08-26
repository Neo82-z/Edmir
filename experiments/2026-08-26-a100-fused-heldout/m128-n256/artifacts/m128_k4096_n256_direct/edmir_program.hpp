#pragma once

#include <array>
#include <string_view>

#include "edmir/runtime/program.hpp"

namespace edmir_generated {

using namespace edmir::runtime;

inline constexpr std::string_view kTarget = "a100-direct";
inline constexpr std::string_view kArch = "sm_80";
inline constexpr GemmProvider kGemmProvider = GemmProvider::kCUTLASS;
inline constexpr GemmKernel kGemmKernel = GemmKernel::kSm80_64x128x32;
inline constexpr CommunicationApi kCommunicationApi = CommunicationApi::kNCCLDevice;
inline constexpr std::string_view kStrategy = "fused";
inline constexpr ProblemDesc kProblem{4, 128, 4096, 256, DType::kFloat16};
inline constexpr std::int32_t kChunkRows = 128;
inline constexpr std::int32_t kBufferCount = 1;

inline constexpr std::array<BufferDesc, 36> kBuffers{{
  {"input.s0.d0.c0", 0, BufferRole::kInput, 1048576ull},
  {"weight.r0", 0, BufferRole::kWeight, 2097152ull},
  {"output.s0.d0.c0", 0, BufferRole::kOutput, 65536ull},
  {"input.s1.d1.c0", 1, BufferRole::kInput, 1048576ull},
  {"weight.r1", 1, BufferRole::kWeight, 2097152ull},
  {"output.s1.d1.c0", 1, BufferRole::kOutput, 65536ull},
  {"input.s2.d2.c0", 2, BufferRole::kInput, 1048576ull},
  {"weight.r2", 2, BufferRole::kWeight, 2097152ull},
  {"output.s2.d2.c0", 2, BufferRole::kOutput, 65536ull},
  {"input.s3.d3.c0", 3, BufferRole::kInput, 1048576ull},
  {"weight.r3", 3, BufferRole::kWeight, 2097152ull},
  {"output.s3.d3.c0", 3, BufferRole::kOutput, 65536ull},
  {"input.s1.d0.c0", 1, BufferRole::kInput, 1048576ull},
  {"output.s1.d0.c0", 0, BufferRole::kOutput, 65536ull},
  {"input.s2.d1.c0", 2, BufferRole::kInput, 1048576ull},
  {"output.s2.d1.c0", 1, BufferRole::kOutput, 65536ull},
  {"input.s3.d2.c0", 3, BufferRole::kInput, 1048576ull},
  {"output.s3.d2.c0", 2, BufferRole::kOutput, 65536ull},
  {"input.s0.d3.c0", 0, BufferRole::kInput, 1048576ull},
  {"output.s0.d3.c0", 3, BufferRole::kOutput, 65536ull},
  {"input.s2.d0.c0", 2, BufferRole::kInput, 1048576ull},
  {"output.s2.d0.c0", 0, BufferRole::kOutput, 65536ull},
  {"input.s3.d1.c0", 3, BufferRole::kInput, 1048576ull},
  {"output.s3.d1.c0", 1, BufferRole::kOutput, 65536ull},
  {"input.s0.d2.c0", 0, BufferRole::kInput, 1048576ull},
  {"output.s0.d2.c0", 2, BufferRole::kOutput, 65536ull},
  {"input.s1.d3.c0", 1, BufferRole::kInput, 1048576ull},
  {"output.s1.d3.c0", 3, BufferRole::kOutput, 65536ull},
  {"input.s3.d0.c0", 3, BufferRole::kInput, 1048576ull},
  {"output.s3.d0.c0", 0, BufferRole::kOutput, 65536ull},
  {"input.s0.d1.c0", 0, BufferRole::kInput, 1048576ull},
  {"output.s0.d1.c0", 1, BufferRole::kOutput, 65536ull},
  {"input.s1.d2.c0", 1, BufferRole::kInput, 1048576ull},
  {"output.s1.d2.c0", 2, BufferRole::kOutput, 65536ull},
  {"input.s2.d3.c0", 2, BufferRole::kInput, 1048576ull},
  {"output.s2.d3.c0", 3, BufferRole::kOutput, 65536ull},
}};

inline constexpr std::array<OpDesc, 16> kOps{{
  {"fused_transfer_gemm.s0.d0.c0", OpKind::kFusedTransferGemm, 0, 0, QueueKind::kCompute, "s0.d0.c0", 128},
  {"fused_transfer_gemm.s1.d1.c0", OpKind::kFusedTransferGemm, 1, 1, QueueKind::kCompute, "s1.d1.c0", 128},
  {"fused_transfer_gemm.s2.d2.c0", OpKind::kFusedTransferGemm, 2, 2, QueueKind::kCompute, "s2.d2.c0", 128},
  {"fused_transfer_gemm.s3.d3.c0", OpKind::kFusedTransferGemm, 3, 3, QueueKind::kCompute, "s3.d3.c0", 128},
  {"fused_transfer_gemm.s1.d0.c0", OpKind::kFusedTransferGemm, 0, 1, QueueKind::kCompute, "s1.d0.c0", 128},
  {"fused_transfer_gemm.s2.d1.c0", OpKind::kFusedTransferGemm, 1, 2, QueueKind::kCompute, "s2.d1.c0", 128},
  {"fused_transfer_gemm.s3.d2.c0", OpKind::kFusedTransferGemm, 2, 3, QueueKind::kCompute, "s3.d2.c0", 128},
  {"fused_transfer_gemm.s0.d3.c0", OpKind::kFusedTransferGemm, 3, 0, QueueKind::kCompute, "s0.d3.c0", 128},
  {"fused_transfer_gemm.s2.d0.c0", OpKind::kFusedTransferGemm, 0, 2, QueueKind::kCompute, "s2.d0.c0", 128},
  {"fused_transfer_gemm.s3.d1.c0", OpKind::kFusedTransferGemm, 1, 3, QueueKind::kCompute, "s3.d1.c0", 128},
  {"fused_transfer_gemm.s0.d2.c0", OpKind::kFusedTransferGemm, 2, 0, QueueKind::kCompute, "s0.d2.c0", 128},
  {"fused_transfer_gemm.s1.d3.c0", OpKind::kFusedTransferGemm, 3, 1, QueueKind::kCompute, "s1.d3.c0", 128},
  {"fused_transfer_gemm.s3.d0.c0", OpKind::kFusedTransferGemm, 0, 3, QueueKind::kCompute, "s3.d0.c0", 128},
  {"fused_transfer_gemm.s0.d1.c0", OpKind::kFusedTransferGemm, 1, 0, QueueKind::kCompute, "s0.d1.c0", 128},
  {"fused_transfer_gemm.s1.d2.c0", OpKind::kFusedTransferGemm, 2, 1, QueueKind::kCompute, "s1.d2.c0", 128},
  {"fused_transfer_gemm.s2.d3.c0", OpKind::kFusedTransferGemm, 3, 2, QueueKind::kCompute, "s2.d3.c0", 128},
}};

inline constexpr std::array<AccessDesc, 48> kAccesses{{
  {0, 0, false},
  {0, 1, false},
  {0, 2, true},
  {1, 3, false},
  {1, 4, false},
  {1, 5, true},
  {2, 6, false},
  {2, 7, false},
  {2, 8, true},
  {3, 9, false},
  {3, 10, false},
  {3, 11, true},
  {4, 12, false},
  {4, 1, false},
  {4, 13, true},
  {5, 14, false},
  {5, 4, false},
  {5, 15, true},
  {6, 16, false},
  {6, 7, false},
  {6, 17, true},
  {7, 18, false},
  {7, 10, false},
  {7, 19, true},
  {8, 20, false},
  {8, 1, false},
  {8, 21, true},
  {9, 22, false},
  {9, 4, false},
  {9, 23, true},
  {10, 24, false},
  {10, 7, false},
  {10, 25, true},
  {11, 26, false},
  {11, 10, false},
  {11, 27, true},
  {12, 28, false},
  {12, 1, false},
  {12, 29, true},
  {13, 30, false},
  {13, 4, false},
  {13, 31, true},
  {14, 32, false},
  {14, 7, false},
  {14, 33, true},
  {15, 34, false},
  {15, 10, false},
  {15, 35, true},
}};

inline constexpr std::array<DependencyDesc, 31> kDependencies{{
  {1, 4, DependencyKind::kQueue, -1},
  {0, 4, DependencyKind::kQueue, -1},
  {2, 5, DependencyKind::kQueue, -1},
  {4, 5, DependencyKind::kQueue, -1},
  {1, 5, DependencyKind::kQueue, -1},
  {3, 6, DependencyKind::kQueue, -1},
  {5, 6, DependencyKind::kQueue, -1},
  {2, 6, DependencyKind::kQueue, -1},
  {4, 7, DependencyKind::kQueue, -1},
  {6, 7, DependencyKind::kQueue, -1},
  {3, 7, DependencyKind::kQueue, -1},
  {6, 8, DependencyKind::kQueue, -1},
  {7, 8, DependencyKind::kQueue, -1},
  {4, 8, DependencyKind::kQueue, -1},
  {7, 9, DependencyKind::kQueue, -1},
  {5, 9, DependencyKind::kQueue, -1},
  {8, 10, DependencyKind::kQueue, -1},
  {6, 10, DependencyKind::kQueue, -1},
  {9, 11, DependencyKind::kQueue, -1},
  {7, 11, DependencyKind::kQueue, -1},
  {11, 12, DependencyKind::kQueue, -1},
  {10, 12, DependencyKind::kQueue, -1},
  {8, 12, DependencyKind::kQueue, -1},
  {12, 13, DependencyKind::kQueue, -1},
  {11, 13, DependencyKind::kQueue, -1},
  {9, 13, DependencyKind::kQueue, -1},
  {13, 14, DependencyKind::kQueue, -1},
  {10, 14, DependencyKind::kQueue, -1},
  {14, 15, DependencyKind::kQueue, -1},
  {12, 15, DependencyKind::kQueue, -1},
  {11, 15, DependencyKind::kQueue, -1},
}};

inline constexpr ProgramView program() {
  return {
      kTarget, kArch, kGemmProvider, kGemmKernel, kCommunicationApi, kStrategy, kProblem, kChunkRows, kBufferCount,
      std::span<const BufferDesc>{kBuffers},
      std::span<const OpDesc>{kOps},
      std::span<const AccessDesc>{kAccesses},
      std::span<const DependencyDesc>{kDependencies},
  };
}

}  // namespace edmir_generated
