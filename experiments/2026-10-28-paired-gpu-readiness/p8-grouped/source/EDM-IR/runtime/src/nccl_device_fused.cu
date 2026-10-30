#include "edmir/runtime/nccl_device_fused.hpp"

#include <cuda_fp16.h>
#include <nccl_device.h>

namespace edmir::runtime {
namespace {

__global__ void lsa_naive_gemm_f16(
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const __half* weight,
    __half* output) {
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= m || column >= n) return;

  const auto* input = static_cast<const __half*>(
      ncclGetLsaPointer(input_window, 0, source_rank));
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += __half2float(input[row * k + inner]) *
                   __half2float(weight[inner * n + column]);
  }
  output[row * n + column] = __float2half(accumulator);
}

}  // namespace

cudaError_t launch_lsa_naive_gemm_f16(
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const void* weight,
    void* output,
    cudaStream_t stream) {
  constexpr dim3 threads{16, 16};
  const dim3 blocks{
      static_cast<unsigned int>((n + threads.x - 1) / threads.x),
      static_cast<unsigned int>((m + threads.y - 1) / threads.y),
  };
  lsa_naive_gemm_f16<<<blocks, threads, 0, stream>>>(
      input_window, source_rank, m, n, k,
      static_cast<const __half*>(weight), static_cast<__half*>(output));
  return cudaGetLastError();
}

}  // namespace edmir::runtime
