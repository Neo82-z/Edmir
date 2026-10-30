#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "edmir/runtime/program.hpp"

namespace edmir::runtime {

inline constexpr std::size_t kNoOperand = std::numeric_limits<std::size_t>::max();

struct PreparedOperands {
  std::size_t input = kNoOperand;
  std::size_t weight = kNoOperand;
  std::size_t output = kNoOperand;
  std::size_t transfer_bytes = 0;
  std::size_t transfer_elements = 0;
};

struct PreparedWait {
  std::int32_t rank;
  std::size_t predecessor;
  std::int32_t event_rank;
  bool operator==(const PreparedWait&) const = default;
};

struct PreparedOp {
  PreparedOperands operands;
  // Preserve the original source-then-destination order.
  std::vector<std::int32_t> devices;
  // Preserve first-seen predecessor order, then current-op device order.
  std::vector<PreparedWait> waits;
};

struct PreparedExecution {
  std::vector<std::size_t> order;
  std::vector<PreparedOp> ops;
};

// Pure host preparation. The caller must keep ProgramView storage alive and
// immutable until the executor is destroyed, including its backing arrays.
PreparedExecution prepare_execution(ProgramView program);

}  // namespace edmir::runtime
