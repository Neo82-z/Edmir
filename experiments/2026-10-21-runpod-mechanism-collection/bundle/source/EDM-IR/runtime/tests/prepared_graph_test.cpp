#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "edmir/runtime/prepared_execution.hpp"

using namespace edmir::runtime;

// Synthetic CPU graphs, not additional GPU benchmark shapes.
void check_graph(std::size_t count) {
  std::vector<std::string> names;
  for (std::size_t i = 0; i < count; ++i) names.push_back("barrier"+std::to_string(i));
  std::vector<OpDesc> ops;
  for (const auto& name : names)
    ops.push_back({name, OpKind::kBarrier, 0, 0, QueueKind::kControl, "", 0});
  std::vector<BufferDesc> buffers;
  std::vector<AccessDesc> accesses;
  for (std::size_t i = 0; i < count; ++i) {
    buffers.push_back({names[i], 0, BufferRole::kToken, 4});
    accesses.push_back({i, i, true});
  }
  std::vector<DependencyDesc> dependencies;
  // Reverse predecessor order, duplicate each edge, and put all predecessors
  // into TWO destinations so a broken stamp reset cannot accidentally pass.
  for (std::size_t i = count-2; i-- > 0;) {
    for (std::size_t dst : {count-2, count-1})
      for (int duplicate = 0; duplicate < 2; ++duplicate)
        dependencies.push_back({i, dst, DependencyKind::kQueue, -1});
  }
  ProgramView p{"cpu-graph-test", "host", GemmProvider::kCuBLAS, GemmKernel::kLibraryDefault,
                CommunicationApi::kHostNCCL, "test", {2, 16, 64, 64, DType::kFloat16},
                16, 1, buffers, ops, accesses, dependencies};
  const auto start = std::chrono::steady_clock::now();
  const auto plan = prepare_execution(p);
  const double milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now()-start).count();
  if (plan.order.size() != count) throw std::runtime_error("lost graph nodes");
  std::vector<std::size_t> positions(count);
  for (std::size_t i = 0; i < count; ++i) positions.at(plan.order[i]) = i;
  for (const auto& edge : dependencies)
    if (positions[edge.src_op] >= positions[edge.dst_op])
      throw std::runtime_error("topological dependency order changed");
  for (std::size_t dst : {count-2, count-1}) {
    const auto& waits = plan.ops[dst].waits;
    if (waits.size() != count-2) throw std::runtime_error("duplicate waits");
    for (std::size_t j = 0; j < waits.size(); ++j)
      if (waits[j].predecessor != count-3-j || waits[j].rank != 0 || waits[j].event_rank != 0)
        throw std::runtime_error("first-seen wait order changed");
  }
  std::cout << "{\"ops\":" << count << ",\"edges_including_duplicates\":" << dependencies.size()
            << ",\"prepare_ms\":" << milliseconds << "}\n";
  // A cycle must be rejected before any executor can wait on it.
  dependencies.push_back({count-1, 0, DependencyKind::kQueue, -1});
  p.dependencies = dependencies;
  bool rejected = false;
  try { (void)prepare_execution(p); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("cyclic graph accepted");
}

int main() {
  try {
    for (std::size_t count : {1000, 10000, 40000}) check_graph(count);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
