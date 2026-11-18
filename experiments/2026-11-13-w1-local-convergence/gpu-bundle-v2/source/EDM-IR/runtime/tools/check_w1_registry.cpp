// Host-only verification of the compiled descriptors used in a W1 shape process.
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir/runtime/prepared_execution.hpp"
#include "edmir_batch.hpp"

using namespace edmir::runtime;
int main() {
  try {
    std::map<std::string_view, ProgramView> programs;
    for (auto candidate : edmir_batch::candidates) {
      const auto p = candidate.program();
      const auto w = uniform_workload(p);
      if (w.fused) validate_uniform_peer_policy(p);
      if (w.inputs.size() != 64 || prepare_execution(p).order.size() != p.ops.size())
        throw std::runtime_error("incomplete W1 workload");
      if (!programs.emplace(candidate.name,p).second) throw std::runtime_error("duplicate candidate");
    }
    if (programs.contains("pipeline")) {
      const auto a = programs.at("pipeline"), b = programs.at("pipeline_serial");
      if (a.ops.size()!=b.ops.size() || a.buffer_count!=2 || b.buffer_count!=2 ||
          a.buffers.size()!=b.buffers.size() || a.accesses.size()!=b.accesses.size())
        throw std::runtime_error("unmatched pipeline pair");
      for (std::size_t i=0; i<a.ops.size(); ++i) {
        if (a.ops[i].id!=b.ops[i].id || a.ops[i].queue!=b.ops[i].queue || a.ops[i].rows!=b.ops[i].rows)
          throw std::runtime_error("mismatched work or peer order");
      }
      for (std::size_t i=0; i<a.buffers.size(); ++i)
        if (a.buffers[i].name!=b.buffers[i].name || a.buffers[i].size_bytes!=b.buffers[i].size_bytes)
          throw std::runtime_error("mismatched buffer");
      for (std::size_t i=0; i<a.accesses.size(); ++i)
        if (a.accesses[i].buffer!=b.accesses[i].buffer || a.accesses[i].op!=b.accesses[i].op ||
            a.accesses[i].write!=b.accesses[i].write) throw std::runtime_error("mismatched operands");
      const auto first = prepare_execution(a, true).order;
      if (first != prepare_execution(b, true).order || !std::is_sorted(first.begin(), first.end()))
        throw std::runtime_error("pipeline pair must submit the same ordered operations");
      using Edge = std::pair<std::size_t,std::size_t>;
      std::set<Edge> original, serial;
      for (auto e : a.dependencies) original.emplace(e.src_op,e.dst_op);
      for (auto e : b.dependencies) serial.emplace(e.src_op,e.dst_op);
      if (!std::includes(serial.begin(),serial.end(),original.begin(),original.end()) || serial==original)
        throw std::runtime_error("serialization did not strictly extend dependencies");
      for (int rank=0;rank<b.problem.world_size;++rank) {
        std::size_t previous=b.ops.size();
        for (std::size_t i=0;i<b.ops.size();++i) {
          const auto& op=b.ops[i];
          if (op.rank!=rank && !(op.kind==OpKind::kTransfer && op.peer_rank==rank)) continue;
          if (previous!=b.ops.size() && !serial.contains({previous,i}))
            throw std::runtime_error("unserialized touching operations");
          previous=i;
        }
      }
    }
    std::cout << programs.size() << " candidate identities validated\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
