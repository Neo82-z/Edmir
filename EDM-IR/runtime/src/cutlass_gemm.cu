#include "edmir/runtime/cutlass_gemm.hpp"

#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace edmir::runtime {
namespace {

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using Accumulator = float;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, Accumulator, Accumulator>;
template <int ThreadblockM, int ThreadblockN, int WarpM, int WarpN>
using Sm80Gemm = cutlass::gemm::device::Gemm<
    Element,
    Layout,
    Element,
    Layout,
    Element,
    Layout,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, 32>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8,
    8>;

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
      {static_cast<const Element*>(activation), k},
      {static_cast<const Element*>(weight), n},
      {static_cast<Element*>(output), n},
      {static_cast<Element*>(output), n},
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
    case GemmKernel::kSm80_128x64x32:
      return launch<Sm80Gemm<128, 64, 64, 32>>(
          m, n, k, activation, weight, output, stream);
    case GemmKernel::kLibraryDefault:
      break;
  }
  throw std::invalid_argument("CUTLASS provider received an incompatible GEMM kernel");
}

}  // namespace edmir::runtime
