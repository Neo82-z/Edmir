#pragma once

#include <cuda_runtime_api.h>

#include "edmir/runtime/program.hpp"

namespace edmir::runtime {

// Provider boundary kept free of CUTLASS types so the rest of the runtime can
// be built and tested without a CUTLASS checkout.
void launch_cutlass_gemm_f16_sm80(
    GemmKernel kernel, int m, int n, int k, const void* activation, const void* weight,
    void* output, cudaStream_t stream);

}  // namespace edmir::runtime
