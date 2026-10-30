// CPU API trace only; this program cannot execute GPU kernels.
#include <iostream>
#include <string>
#include <vector>
#include "cuda_api_trace.hpp"
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/validate.hpp"
#include "edmir_program.hpp"
using namespace edmir::runtime;
int main(int argc, char** argv) {
  const bool prepared = argc > 1 && std::string(argv[1]) == "prepared";
  const bool reverse = argc > 2 && std::string(argv[2]) == "reverse";
  const bool compute_only = argc > 3 && std::string(argv[3]) == "compute_only";
  auto p = edmir_generated::program();
  std::vector<DependencyDesc> deps;
  if (compute_only) {
    std::vector<int> last(p.problem.world_size,-1);
    for (std::size_t i=0;i<p.ops.size();++i) {
      const auto& op=p.ops[i];
      if (op.kind!=OpKind::kFusedTransferGemm) return 2;
      if (last[op.rank]>=0) deps.push_back({static_cast<std::size_t>(last[op.rank]),i,DependencyKind::kQueue,-1});
      last[op.rank]=static_cast<int>(i);
    }
    p.dependencies=deps; // Counterfactual test input, never saved into the compiler.
  }
  for (auto i:topological_order(p)) std::cout<<"ORDER "<<i<<' '<<p.ops[i].peer_rank<<' '<<p.ops[i].rank<<'\n';
  CudaBackendOptions options; options.prepare_metadata=prepared;
  for(int i=0;i<p.problem.world_size;++i)options.devices.push_back(reverse ? p.problem.world_size-1-i : i);
  test::reset_api_trace();
  CudaBackend backend(p,options); backend.prepare();
  std::cout<<"SETUP\n";
  for(const auto& s:test::api_trace()) std::cout<<s<<'\n';
  for(int i=0;i<2;++i) {
    test::clear_api_trace(); backend.execute(); backend.synchronize();
    std::cout<<"ITER "<<i<<'\n';
    for(const auto& s:test::api_trace()) std::cout<<s<<'\n';
  }
}
