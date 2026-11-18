#pragma once

#include <vector>
#include "edmir/runtime/program.hpp"

namespace edmir::runtime {

// Canonical order is [source * P + destination], independent of IR buffer order.
// This restricted adapter rejects chunking and unequal per-source weights.
struct UniformWorkload {
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  std::vector<std::size_t> weights;
  bool fused = false;
};

UniformWorkload uniform_workload(ProgramView program);
void validate_uniform_peer_policy(ProgramView program);
std::size_t merged_row_offset(int source, int rows_per_peer, int world_size);

}  // namespace edmir::runtime
