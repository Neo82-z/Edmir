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
using Gemm = cutlass::gemm::device::Gemm<
    Element,
    Layout,
    Element,
    Layout,
    Element,
    Layout,
    Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
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

}  // namespace

void launch_cutlass_gemm_f16_sm80(
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

}  // namespace edmir::runtime
