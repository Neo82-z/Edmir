#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include "edmir/runtime/program.hpp"

namespace edmir::runtime {
struct SplitKWorkspace {
  std::size_t semaphore_bytes;
  std::size_t partial_offset;
  std::size_t partial_bytes;
  std::size_t total_bytes;
};

// One semaphore per output tile, then a separate aligned FP32 C/D matrix.
// Each executed fused GEMM owns its workspace; no cross-stream sharing.
inline SplitKWorkspace split_k_workspace(
    GemmKernel kernel, FusedScheduleDesc schedule, int m, int n) {
  if (m <= 0 || n <= 0 || schedule.n_tiles_per_cta < 1 || schedule.n_tiles_per_cta > 2)
    throw std::invalid_argument("invalid split-K workspace dimensions");
  if (schedule.split_k_reduction != SplitKReduction::kSerialFP16 &&
      schedule.split_k_reduction != SplitKReduction::kSerialFP32)
    throw std::invalid_argument("invalid split-K reduction");
  if (gemm_kernel_split_k_slices(kernel) == 1) {
    if (schedule.split_k_reduction == SplitKReduction::kSerialFP32)
      throw std::invalid_argument("FP32 workspace requires split-K");
    return {};
  }
  const auto tm = std::size_t(gemm_kernel_threadblock_m(kernel));
  const auto tn = std::size_t(gemm_kernel_threadblock_n(kernel)) * schedule.n_tiles_per_cta;
  if (!tm || !tn) throw std::invalid_argument("unknown kernel geometry");
  const auto mt = (std::size_t(m) + tm - 1) / tm;
  const auto nt = (std::size_t(n) + tn - 1) / tn;
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (mt > (limit - 255) / sizeof(int) / nt)
    throw std::overflow_error("semaphore workspace overflow");
  const auto sem = mt * nt * sizeof(int);
  if (schedule.split_k_reduction == SplitKReduction::kSerialFP16) return {sem, 0, 0, sem};
  const auto offset = (sem + 255) / 256 * 256;
  if (std::size_t(m) > (limit - offset) / sizeof(float) / std::size_t(n))
    throw std::overflow_error("partial workspace overflow");
  const auto bytes = std::size_t(m) * n * sizeof(float);
  return {sem, offset, bytes, offset + bytes};
}
}  // namespace edmir::runtime
