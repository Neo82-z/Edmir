#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace edmir::runtime::detail {

using Sm80Element = cutlass::half_t;
using Sm80Layout = cutlass::layout::RowMajor;
using Sm80Accumulator = float;
using Sm80Epilogue = cutlass::epilogue::thread::LinearCombination<
    Sm80Element, 8, Sm80Accumulator, Sm80Accumulator>;

template <
    int ThreadblockM,
    int ThreadblockN,
    int WarpM,
    int WarpN,
    bool SplitKSerial = false>
using Sm80Gemm = cutlass::gemm::device::Gemm<
    Sm80Element,
    Sm80Layout,
    Sm80Element,
    Sm80Layout,
    Sm80Element,
    Sm80Layout,
    Sm80Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, 32>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    Sm80Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    8,
    8,
    SplitKSerial>;

}  // namespace edmir::runtime::detail
