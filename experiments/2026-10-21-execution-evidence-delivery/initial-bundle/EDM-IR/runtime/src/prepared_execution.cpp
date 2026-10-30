#include "edmir/runtime/prepared_execution.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "edmir/runtime/validate.hpp"

namespace edmir::runtime {

PreparedExecution prepare_execution(ProgramView p) {
  const auto report = validate_program(p);
  if (!report.valid()) {
    std::string message = "invalid ProgramView for execution preparation";
    for (const auto& error : report.errors) message += "; " + error;
    throw std::invalid_argument(message);
  }
  PreparedExecution result;
  result.order = topological_order(p);
  result.ops.resize(p.ops.size());
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    const auto& op = p.ops[i];
    auto& prepared = result.ops[i];
    if ((op.kind == OpKind::kTransfer || op.kind == OpKind::kFusedTransferGemm) &&
        op.peer_rank != op.rank) {
      prepared.devices.push_back(op.peer_rank);
    }
    prepared.devices.push_back(op.rank);
    if (op.kind == OpKind::kTransfer) {
      prepared.operands.transfer_elements =
          static_cast<std::size_t>(op.rows) * static_cast<std::size_t>(p.problem.k);
      prepared.operands.transfer_bytes = prepared.operands.transfer_elements * dtype_size(p.problem.dtype);
    }
  }
  // One pass over the access table. Keep the first matching role, as the
  // executor's original find_if did; token dependencies are not GEMM operands.
  for (const auto& access : p.accesses) {
    const auto kind = p.ops[access.op].kind;
    const auto role = p.buffers[access.buffer].role;
    auto& operands = result.ops[access.op].operands;
    if (kind == OpKind::kBarrier) continue;
    if (access.write) {
      if (operands.output == kNoOperand) operands.output = access.buffer;
    } else if (kind == OpKind::kTransfer || role == BufferRole::kInput || role == BufferRole::kStaging) {
      if (operands.input == kNoOperand) operands.input = access.buffer;
    } else if (role == BufferRole::kWeight) {
      if (operands.weight == kNoOperand) operands.weight = access.buffer;
    }
  }
  std::vector<std::vector<std::size_t>> incoming(p.ops.size());
  for (const auto& dependency : p.dependencies) {
    incoming[dependency.dst_op].push_back(dependency.src_op);
  }
  // Destination stamps deduplicate in O(V+E), preserving first-seen order.
  // Searching the growing predecessor vector costs O(E^2) for a wide fan-in.
  std::vector<std::size_t> seen(p.ops.size(), kNoOperand);
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    auto& current = result.ops[i];
    for (const auto predecessor : incoming[i]) {
      if (seen[predecessor] == i) continue;
      seen[predecessor] = i;
      const auto& devices = result.ops[predecessor].devices;
      for (const auto rank : current.devices) {
        const bool shared = std::find(devices.begin(), devices.end(), rank) != devices.end();
        current.waits.push_back({rank, predecessor, shared ? rank : p.ops[predecessor].rank});
      }
    }
  }
  return result;
}

}  // namespace edmir::runtime
