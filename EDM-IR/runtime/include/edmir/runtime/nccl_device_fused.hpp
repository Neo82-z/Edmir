#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <nccl.h>

#include "edmir/runtime/program.hpp"

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

cudaError_t launch_lsa_cutlass_gemm_f16_sm80(
    GemmKernel kernel,
    ncclWindow_t input_window,
    int source_rank,
    int m,
    int n,
    int k,
    const void* weight,
    void* output,
    void* workspace,
    cudaStream_t stream);

std::size_t lsa_cutlass_workspace_size(GemmKernel kernel, int m, int n);

}  // namespace edmir::runtime
