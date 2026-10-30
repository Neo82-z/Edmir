#include <iostream>
#include <stdexcept>
#include <string>
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir_program.hpp"

int main() {
  using namespace edmir::runtime;
  try {
    const auto p = edmir_generated::program();
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
    std::cout << "canonical matrices and merged row slices checked\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
