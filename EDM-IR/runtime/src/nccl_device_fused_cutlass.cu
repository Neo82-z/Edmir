#include "edmir/runtime/nccl_device_fused.hpp"

#include <nccl_device.h>

#include "cutlass/device_kernel.h"
#include "cutlass_sm80_gemm.cuh"

namespace edmir::runtime {
namespace {

template <class Gemm>
__global__ void lsa_cutlass_gemm(
    ncclWindow_t input_window,
    int source_rank,
    typename Gemm::GemmKernel::Params params) {
  using Kernel = typename Gemm::GemmKernel;
  using Element = typename Gemm::ElementA;

  params.ref_A.reset(static_cast<Element*>(
      ncclGetLsaPointer(input_window, 0, source_rank)));

  extern __shared__ int shared_storage_base[];
  auto* shared_storage =
      reinterpret_cast<typename Kernel::SharedStorage*>(shared_storage_base);
  Kernel{}(params, *shared_storage);
}

template <class Gemm>
cudaError_t launch(
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const void* weight,
    void* output,
    void* workspace,
    int split_k_slices,
    cudaStream_t stream) {
  using Element = typename Gemm::ElementA;
  using Kernel = typename Gemm::GemmKernel;
  using Swizzle = typename Gemm::ThreadblockSwizzle;

  typename Gemm::Arguments arguments{
      {m, n, k},
      {static_cast<const Element*>(weight), k},
      {static_cast<const Element*>(weight), n},
      {static_cast<Element*>(output), n},
      {static_cast<Element*>(output), n},
      {1.0f, 0.0f},
      split_k_slices,
  };
  if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }

  Swizzle swizzle;
  const cutlass::gemm::GemmCoord problem{m, n, k};
  const auto tiled_shape = swizzle.get_tiled_shape(
      problem,
      {Gemm::ThreadblockShape::kM,
       Gemm::ThreadblockShape::kN,
       Gemm::ThreadblockShape::kK},
      split_k_slices);
  typename Kernel::Params params{
      problem,
      tiled_shape,
      arguments.ref_A.non_const_ref(),
      arguments.ref_B.non_const_ref(),
      arguments.ref_C.non_const_ref(),
      arguments.ref_D,
      arguments.epilogue,
      static_cast<int*>(workspace),
  };

  const auto grid = swizzle.get_grid_shape(tiled_shape);
  const dim3 block(Kernel::kThreadCount, 1, 1);
  const int shared_bytes = static_cast<int>(sizeof(typename Kernel::SharedStorage));
  if (shared_bytes >= (48 << 10)) {
    const auto result = cudaFuncSetAttribute(
        lsa_cutlass_gemm<Gemm>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    if (result != cudaSuccess) return result;
  }
  lsa_cutlass_gemm<Gemm><<<grid, block, shared_bytes, stream>>>(
      input_window, source_rank, params);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_lsa_cutlass_gemm_f16_sm80(
    GemmKernel kernel,
    FusedScheduleDesc schedule,
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const void* weight,
    void* output,
    void* workspace,
    cudaStream_t stream) {
  if (schedule.n_tiles_per_cta == 2) {
    switch (kernel) {
      case GemmKernel::kSm80_128x128x32:
        return launch<detail::Sm80Gemm<128, 256, 64, 64>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
      case GemmKernel::kSm80_64x128x32:
        return launch<detail::Sm80Gemm<64, 256, 32, 64>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
      case GemmKernel::kSm80_64x128x32SplitK2:
        return launch<detail::Sm80Gemm<64, 256, 32, 64, true>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 2, stream);
      case GemmKernel::kSm80_64x128x32SplitK4:
        return launch<detail::Sm80Gemm<64, 256, 32, 64, true>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 4, stream);
      case GemmKernel::kSm80_64x128x32SplitK8:
        return launch<detail::Sm80Gemm<64, 256, 32, 64, true>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 8, stream);
      case GemmKernel::kSm80_64x128x32SplitK16:
        return launch<detail::Sm80Gemm<64, 256, 32, 64, true>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 16, stream);
      case GemmKernel::kSm80_128x64x32:
        return launch<detail::Sm80Gemm<128, 128, 64, 32>>(
            input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
      case GemmKernel::kLibraryDefault:
      case GemmKernel::kLsaNaiveF16:
        return cudaErrorInvalidValue;
    }
    return cudaErrorInvalidValue;
  }
  if (schedule.n_tiles_per_cta != 1) return cudaErrorInvalidValue;
  switch (kernel) {
    case GemmKernel::kSm80_128x128x32:
      return launch<detail::Sm80Gemm<128, 128, 64, 64>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
    case GemmKernel::kSm80_64x128x32:
      return launch<detail::Sm80Gemm<64, 128, 32, 64>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
    case GemmKernel::kSm80_64x128x32SplitK2:
      return launch<detail::Sm80Gemm<64, 128, 32, 64, true>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 2, stream);
    case GemmKernel::kSm80_64x128x32SplitK4:
      return launch<detail::Sm80Gemm<64, 128, 32, 64, true>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 4, stream);
    case GemmKernel::kSm80_64x128x32SplitK8:
      return launch<detail::Sm80Gemm<64, 128, 32, 64, true>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 8, stream);
    case GemmKernel::kSm80_64x128x32SplitK16:
      return launch<detail::Sm80Gemm<64, 128, 32, 64, true>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 16, stream);
    case GemmKernel::kSm80_128x64x32:
      return launch<detail::Sm80Gemm<128, 64, 64, 32>>(
          input_window, source_rank, m, n, k, weight, output, workspace, 1, stream);
    case GemmKernel::kLibraryDefault:
    case GemmKernel::kLsaNaiveF16:
      return cudaErrorInvalidValue;
  }
  return cudaErrorInvalidValue;
}

std::size_t lsa_cutlass_workspace_size(
    GemmKernel kernel, FusedScheduleDesc schedule, int m, int n) {
  if (gemm_kernel_split_k_slices(kernel) == 1) return 0;
  const auto tile_m = gemm_kernel_threadblock_m(kernel);
  const auto tile_n = gemm_kernel_threadblock_n(kernel) * schedule.n_tiles_per_cta;
  if (tile_m <= 0 || tile_n <= 0) return 0;
  const auto m_tiles = static_cast<std::size_t>((m + tile_m - 1) / tile_m);
  const auto n_tiles = static_cast<std::size_t>((n + tile_n - 1) / tile_n);
  return m_tiles * n_tiles * sizeof(int);
}

}  // namespace edmir::runtime
