#include <array>
#include <cassert>
#include <stdexcept>

#include "edmir/runtime/program.hpp"
#include "edmir/runtime/validate.hpp"

using namespace edmir::runtime;

int main() {
  static_assert(gemm_kernel_threadblock_n(GemmKernel::kSm80_64x128x32) == 128);
  static_assert(gemm_kernel_threadblock_n(GemmKernel::kSm80_128x64x32) == 64);
  static_assert(gemm_kernel_threadblock_m(GemmKernel::kSm80_64x128x32) == 64);
  static_assert(gemm_kernel_threadblock_n(GemmKernel::kLsaNaiveF16) == 0);
  static_assert(gemm_kernel_split_k_slices(GemmKernel::kSm80_64x128x32) == 1);
  static_assert(
      gemm_kernel_split_k_slices(GemmKernel::kSm80_64x128x32SplitK2) == 2);
  static_assert(
      gemm_kernel_split_k_slices(GemmKernel::kSm80_64x128x32SplitK4) == 4);
  static_assert(
      gemm_kernel_split_k_slices(GemmKernel::kSm80_64x128x32SplitK8) == 8);
  static_assert(
      gemm_kernel_split_k_slices(GemmKernel::kSm80_64x128x32SplitK16) == 16);
  constexpr ProblemDesc problem{2, 16, 64, 64, DType::kFloat16};
  constexpr std::array buffers{
      BufferDesc{"input", 0, BufferRole::kInput, 2048},
      BufferDesc{"stage", 1, BufferRole::kStaging, 2048},
      BufferDesc{"weight", 1, BufferRole::kWeight, 8192},
      BufferDesc{"output", 1, BufferRole::kOutput, 2048},
  };
  constexpr std::array ops{
      OpDesc{"transfer", OpKind::kTransfer, 1, 0, QueueKind::kComm, "s0.d1.c0", 16},
      OpDesc{"gemm", OpKind::kGemm, 1, 0, QueueKind::kCompute, "s0.d1.c0", 16},
  };
  constexpr std::array accesses{
      AccessDesc{0, 0, false}, AccessDesc{0, 1, true},
      AccessDesc{1, 1, false}, AccessDesc{1, 2, false}, AccessDesc{1, 3, true},
  };
  constexpr std::array dependencies{
      DependencyDesc{0, 1, DependencyKind::kReadAfterWrite, 1},
  };
  const ProgramView program{
      "test", "host", GemmProvider::kCuBLAS, GemmKernel::kLibraryDefault,
      CommunicationApi::kHostNCCL, "pipelined", problem, 16, 1,
      buffers, ops, accesses, dependencies,
  };
  assert(validate_program(program).valid());
  const auto order = topological_order(program);
  assert((order == std::vector<std::size_t>{0, 1}));

  constexpr std::array invalid_buffers{
      BufferDesc{"input", 0, BufferRole::kInput, 2048},
      BufferDesc{"stage", 0, BufferRole::kStaging, 2048},
      BufferDesc{"weight", 1, BufferRole::kWeight, 8192},
      BufferDesc{"output", 1, BufferRole::kOutput, 2048},
  };
  const ProgramView invalid_transfer{
      "test", "host", GemmProvider::kCuBLAS, GemmKernel::kLibraryDefault,
      CommunicationApi::kHostNCCL, "pipelined", problem, 16, 1,
      invalid_buffers, ops, accesses, dependencies,
  };
  assert(!validate_program(invalid_transfer).valid());

  constexpr std::array cyclic_dependencies{
      DependencyDesc{0, 1, DependencyKind::kQueue, -1},
      DependencyDesc{1, 0, DependencyKind::kQueue, -1},
  };
  const ProgramView cyclic{
      "test", "host", GemmProvider::kCuBLAS, GemmKernel::kLibraryDefault,
      CommunicationApi::kHostNCCL, "pipelined", problem, 16, 1,
      buffers, ops, accesses, cyclic_dependencies,
  };
  assert(!validate_program(cyclic).valid());
  bool rejected = false;
  try {
    static_cast<void>(topological_order(cyclic));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  constexpr std::array fused_buffers{
      BufferDesc{"fused_input", 0, BufferRole::kInput, 2048},
      BufferDesc{"fused_weight", 1, BufferRole::kWeight, 8192},
      BufferDesc{"fused_output", 1, BufferRole::kOutput, 2048},
  };
  constexpr std::array fused_ops{
      OpDesc{
          "fused", OpKind::kFusedTransferGemm, 1, 0, QueueKind::kCompute,
          "s0.d1.c0", 16},
  };
  constexpr std::array fused_accesses{
      AccessDesc{0, 0, false}, AccessDesc{0, 1, false}, AccessDesc{0, 2, true},
  };
  constexpr std::array<DependencyDesc, 0> no_dependencies{};
  const ProgramView fused{
      "test", "sm_80", GemmProvider::kCudaReference, GemmKernel::kLsaNaiveF16,
      CommunicationApi::kNCCLDevice, "fused", problem, 16, 1,
      fused_buffers, fused_ops, fused_accesses, no_dependencies,
  };
  assert(validate_program(fused).valid());
  const ProgramView fused_without_device_api{
      "test", "sm_80", GemmProvider::kCudaReference, GemmKernel::kLsaNaiveF16,
      CommunicationApi::kHostNCCL, "fused", problem, 16, 1,
      fused_buffers, fused_ops, fused_accesses, no_dependencies,
  };
  assert(!validate_program(fused_without_device_api).valid());
  return 0;
}
