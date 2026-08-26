#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

namespace edmir::runtime {

cudaError_t launch_lsa_naive_gemm_f16(
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const void* weight,
    void* output,
    cudaStream_t stream);

}  // namespace edmir::runtime
