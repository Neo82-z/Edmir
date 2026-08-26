#include "edmir/runtime/cutlass_gemm.hpp"

#include <stdexcept>
#include <string>

#include "cutlass_sm80_gemm.cuh"

namespace edmir::runtime {
namespace {

using detail::Sm80Element;
using detail::Sm80Gemm;

void check_cutlass(cutlass::Status status, const char* expression) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string(expression) + ": " + cutlassGetStatusString(status));
  }
}

template <class Gemm>
void launch(
    int m, int n, int k, const void* activation, const void* weight, void* output,
    cudaStream_t stream) {
  Gemm gemm;
  typename Gemm::Arguments arguments{
      {m, n, k},
      {static_cast<const Sm80Element*>(activation), k},
      {static_cast<const Sm80Element*>(weight), n},
      {static_cast<Sm80Element*>(output), n},
      {static_cast<Sm80Element*>(output), n},
      {1.0f, 0.0f},
  };
  check_cutlass(gemm.can_implement(arguments), "CUTLASS can_implement");
  check_cutlass(gemm(arguments, nullptr, stream), "CUTLASS GEMM launch");
}

}  // namespace

void launch_cutlass_gemm_f16_sm80(
    GemmKernel kernel, int m, int n, int k, const void* activation, const void* weight,
    void* output, cudaStream_t stream) {
  switch (kernel) {
    case GemmKernel::kSm80_128x128x32:
      return launch<Sm80Gemm<128, 128, 64, 64>>(
          m, n, k, activation, weight, output, stream);
    case GemmKernel::kSm80_64x128x32:
      return launch<Sm80Gemm<64, 128, 32, 64>>(
          m, n, k, activation, weight, output, stream);
    case GemmKernel::kSm80_64x128x32SplitK8:
      break;
    case GemmKernel::kSm80_128x64x32:
      return launch<Sm80Gemm<128, 64, 64, 32>>(
          m, n, k, activation, weight, output, stream);
    case GemmKernel::kLibraryDefault:
    case GemmKernel::kLsaNaiveF16:
      break;
  }
  throw std::invalid_argument("CUTLASS provider received an incompatible GEMM kernel");
}

}  // namespace edmir::runtime
