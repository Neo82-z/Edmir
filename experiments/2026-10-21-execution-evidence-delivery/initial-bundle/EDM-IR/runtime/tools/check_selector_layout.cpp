#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "edmir/runtime/prepared_execution.hpp"
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir/runtime/validate.hpp"
#include "edmir_program.hpp"

namespace {
using namespace edmir::runtime;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent reconstruction of the original executor's access scans and
// dependency-to-event mapping. These checks do not simulate GPU execution.
void check_prepared(ProgramView p) {
  const auto prepared = prepare_execution(p);
  require(prepared.order == topological_order(p), "prepared op order changed");
  auto devices = [](const OpDesc& op) {
    std::vector<std::int32_t> result{op.rank};
    if ((op.kind == OpKind::kTransfer || op.kind == OpKind::kFusedTransferGemm) &&
        op.peer_rank != op.rank) result.insert(result.begin(), op.peer_rank);
    return result;
  };
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    const auto& op = p.ops[i];
    const auto& actual = prepared.ops.at(i);
    require(actual.devices == devices(op), "prepared participating ranks changed");
    std::vector<std::size_t> predecessors;
    std::vector<PreparedWait> waits;
    for (const auto& dep : p.dependencies) {
      if (dep.dst_op != i ||
          std::find(predecessors.begin(), predecessors.end(), dep.src_op) != predecessors.end()) continue;
      predecessors.push_back(dep.src_op);
      const auto previous_devices = devices(p.ops[dep.src_op]);
      for (const auto rank : devices(op)) {
        const bool shared = std::count(previous_devices.begin(), previous_devices.end(), rank);
        waits.push_back({rank, dep.src_op, shared ? rank : p.ops[dep.src_op].rank});
      }
    }
    require(actual.waits == waits, "prepared dependency event order/rank changed");
    if (op.kind == OpKind::kBarrier) {
      require(actual.operands.input == kNoOperand && actual.operands.output == kNoOperand,
              "barrier acquired tensor operands");
      continue;
    }
    auto access = [&](bool write, BufferRole role) {
      for (const auto& a : p.accesses)
        if (a.op == i && a.write == write && p.buffers[a.buffer].role == role) return a.buffer;
      throw std::runtime_error("reference operand is missing");
    };
    const auto input_role = op.kind == OpKind::kGemm ? BufferRole::kStaging : BufferRole::kInput;
    const auto output_role = op.kind == OpKind::kTransfer ? BufferRole::kStaging : BufferRole::kOutput;
    require(actual.operands.input == access(false, input_role), "prepared input binding changed");
    require(actual.operands.output == access(true, output_role), "prepared output binding changed");
    if (op.kind == OpKind::kTransfer) {
      require(actual.operands.transfer_elements == static_cast<std::size_t>(op.rows)*p.problem.k &&
              actual.operands.transfer_bytes == static_cast<std::size_t>(op.rows)*p.problem.k*dtype_size(p.problem.dtype),
              "prepared transfer count changed");
      require(actual.operands.weight == kNoOperand, "transfer acquired weight operand");
    } else {
      require(actual.operands.weight == access(false, BufferRole::kWeight), "prepared weight binding changed");
      require(actual.operands.transfer_bytes == 0, "GEMM acquired transfer bytes");
    }
  }
}

void check_preparation_variants(ProgramView p) {
  check_prepared(p);
  std::vector<AccessDesc> accesses(p.accesses.begin(), p.accesses.end());
  std::reverse(accesses.begin(), accesses.end());
  p.accesses = accesses;
  check_prepared(p);  // No assumption about access-table grouping or direction.
  std::vector<DependencyDesc> dependencies(p.dependencies.begin(), p.dependencies.end());
  std::reverse(dependencies.begin(), dependencies.end());
  const auto original = dependencies;
  dependencies.insert(dependencies.end(), original.begin(), original.end());
  p.dependencies = dependencies;
  check_prepared(p);  // Preserve first-seen order; suppress repeated predecessor waits.
  p.dependencies = {};
  check_prepared(p);
  p.accesses = {};
  bool rejected = false;
  try { static_cast<void>(prepare_execution(p)); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "preparation accepted missing operands");
}
}  // namespace

int main() {
  using namespace edmir::runtime;
  try {
    const auto p = edmir_generated::program();
    check_preparation_variants(p);
    const auto w = uniform_workload(p);
    const auto count = p.problem.world_size;
    for (int s = 0; s < count; ++s) for (int d = 0; d < count; ++d) {
      const auto key = s*count+d;
      const auto tile = "s"+std::to_string(s)+".d"+std::to_string(d)+".c0";
      if (p.buffers[w.inputs[key]].name != "input."+tile ||
          p.buffers[w.outputs[key]].name != "output."+tile ||
          p.buffers[w.weights[d]].name != "weight.r"+std::to_string(d))
        throw std::runtime_error("canonical source/destination mapping differs from generated IR");
      for (int row = 0; row < p.problem.rows_per_peer; ++row) {
        const auto merged = merged_row_offset(s, p.problem.rows_per_peer, count)+row;
        if (merged/static_cast<std::size_t>(p.problem.rows_per_peer) != static_cast<std::size_t>(s) ||
            merged%p.problem.rows_per_peer != static_cast<std::size_t>(row))
          throw std::runtime_error("merged rows do not preserve source-local order");
      }
    }
    bool rejected = false;
    try { static_cast<void>(merged_row_offset(count, p.problem.rows_per_peer, count)); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("out-of-range source accepted");
    std::cout << "canonical matrices, merged rows, and prepared operand/event mappings checked\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
