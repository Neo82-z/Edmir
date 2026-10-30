#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace edmir::runtime {

enum class DType : std::uint8_t {
  kFloat16,
  kBFloat16,
  kFloat32,
};

enum class GemmProvider : std::uint8_t {
  kCuBLAS,
  kCUTLASS,
  kCudaReference,
};

enum class GemmKernel : std::uint8_t {
  kLibraryDefault,
  kSm80_128x128x32,
  kSm80_64x128x32,
  kSm80_64x128x32SplitK2,
  kSm80_64x128x32SplitK4,
  kSm80_64x128x32SplitK8,
  kSm80_64x128x32SplitK16,
  kSm80_128x64x32,
  kLsaNaiveF16,
};

enum class CommunicationApi : std::uint8_t {
  kHostNCCL,
  kNCCLDevice,
};

constexpr std::string_view gemm_kernel_name(GemmKernel kernel) {
  switch (kernel) {
    case GemmKernel::kLibraryDefault:
      return "library_default";
    case GemmKernel::kSm80_128x128x32:
      return "sm80_128x128x32";
    case GemmKernel::kSm80_64x128x32:
      return "sm80_64x128x32";
    case GemmKernel::kSm80_64x128x32SplitK2:
      return "sm80_64x128x32_splitk2";
    case GemmKernel::kSm80_64x128x32SplitK4:
      return "sm80_64x128x32_splitk4";
    case GemmKernel::kSm80_64x128x32SplitK8:
      return "sm80_64x128x32_splitk8";
    case GemmKernel::kSm80_64x128x32SplitK16:
      return "sm80_64x128x32_splitk16";
    case GemmKernel::kSm80_128x64x32:
      return "sm80_128x64x32";
    case GemmKernel::kLsaNaiveF16:
      return "lsa_naive_f16";
  }
  return "unknown";
}

constexpr std::int32_t gemm_kernel_threadblock_n(GemmKernel kernel) {
  switch (kernel) {
    case GemmKernel::kSm80_128x128x32:
    case GemmKernel::kSm80_64x128x32:
    case GemmKernel::kSm80_64x128x32SplitK2:
    case GemmKernel::kSm80_64x128x32SplitK4:
    case GemmKernel::kSm80_64x128x32SplitK8:
    case GemmKernel::kSm80_64x128x32SplitK16:
      return 128;
    case GemmKernel::kSm80_128x64x32:
      return 64;
    case GemmKernel::kLibraryDefault:
    case GemmKernel::kLsaNaiveF16:
      return 0;
  }
  return 0;
}

constexpr std::int32_t gemm_kernel_threadblock_m(GemmKernel kernel) {
  switch (kernel) {
    case GemmKernel::kSm80_128x128x32:
    case GemmKernel::kSm80_128x64x32:
      return 128;
    case GemmKernel::kSm80_64x128x32:
    case GemmKernel::kSm80_64x128x32SplitK2:
    case GemmKernel::kSm80_64x128x32SplitK4:
    case GemmKernel::kSm80_64x128x32SplitK8:
    case GemmKernel::kSm80_64x128x32SplitK16:
      return 64;
    case GemmKernel::kLibraryDefault:
    case GemmKernel::kLsaNaiveF16:
      return 0;
  }
  return 0;
}

constexpr std::int32_t gemm_kernel_split_k_slices(GemmKernel kernel) {
  switch (kernel) {
    case GemmKernel::kSm80_64x128x32SplitK2:
      return 2;
    case GemmKernel::kSm80_64x128x32SplitK4:
      return 4;
    case GemmKernel::kSm80_64x128x32SplitK8:
      return 8;
    case GemmKernel::kSm80_64x128x32SplitK16:
      return 16;
    default:
      return 1;
  }
}

constexpr std::string_view gemm_provider_name(GemmProvider provider) {
  switch (provider) {
    case GemmProvider::kCuBLAS:
      return "cublas";
    case GemmProvider::kCUTLASS:
      return "cutlass";
    case GemmProvider::kCudaReference:
      return "cuda_reference";
  }
  return "unknown";
}

enum class BufferRole : std::uint8_t {
  kInput,
  kWeight,
  kStaging,
  kOutput,
  kToken,
};

enum class OpKind : std::uint8_t {
  kTransfer,
  kGemm,
  kBarrier,
  kFusedTransferGemm,
};

enum class QueueKind : std::uint8_t {
  kComm,
  kCompute,
  kControl,
};

enum class DependencyKind : std::uint8_t {
  kQueue,
  kReadAfterWrite,
  kBufferReuse,
  kWriteAfterWrite,
};

struct ProblemDesc {
  std::int32_t world_size;
  std::int32_t rows_per_peer;
  std::int32_t k;
  std::int32_t n;
  DType dtype;
};

struct BufferDesc {
  std::string_view name;
  std::int32_t rank;
  BufferRole role;
  std::size_t size_bytes;
};

struct OpDesc {
  std::string_view id;
  OpKind kind;
  std::int32_t rank;
  std::int32_t peer_rank;
  QueueKind queue;
  std::string_view tile;
  std::int32_t rows;
};

struct AccessDesc {
  std::size_t op;
  std::size_t buffer;
  bool write;
};

struct DependencyDesc {
  std::size_t src_op;
  std::size_t dst_op;
  DependencyKind kind;
  std::int32_t buffer;
};

struct FusedScheduleDesc {
  std::int32_t n_tiles_per_cta = 1;
};

struct ProgramView {
  std::string_view target;
  std::string_view arch;
  GemmProvider gemm_provider;
  GemmKernel gemm_kernel;
  CommunicationApi communication_api;
  std::string_view strategy;
  ProblemDesc problem;
  std::int32_t chunk_rows;
  std::int32_t buffer_count;
  std::span<const BufferDesc> buffers;
  std::span<const OpDesc> ops;
  std::span<const AccessDesc> accesses;
  std::span<const DependencyDesc> dependencies;
  FusedScheduleDesc fused_schedule{};
};

constexpr std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kFloat32:
      return 4;
  }
  return 0;
}

}  // namespace edmir::runtime
