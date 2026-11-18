#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <set>
#include <vector>
#include "cuda_api_trace.hpp"
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/merged_host_backend.hpp"
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir_program.hpp"
#include "edmir_host_program.hpp"
using namespace edmir::runtime;

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "unsafe operation was not rejected");
}
std::size_t count(const std::vector<std::string>& trace, const std::string& prefix) {
  return std::count_if(trace.begin(), trace.end(), [&](const auto& s) { return s.starts_with(prefix); });
}
std::string default_stream_sync(int device) {
  std::ostringstream text;
  text << "sync_stream device=" << device << ' ' << static_cast<void*>(nullptr);
  return text.str();
}
template<class Backend> void upload(Backend& backend, ProgramView p, bool reverse = false) {
  const auto w = uniform_workload(p);
  for (const auto* ids : {&w.inputs, &w.weights}) for (auto id : *ids) {
    // The API stub never dereferences tensor data. Avoid a large allocation.
    unsigned char dummy = 0;
    test::clear_api_trace();
    backend.copy_from_host(id, &dummy, p.buffers[id].size_bytes);
    const auto trace = test::api_trace();
    const auto device = reverse ? p.problem.world_size-1-p.buffers[id].rank : p.buffers[id].rank;
    require(trace.size() == 3 && trace[1].starts_with("copy device=") &&
            trace[2] == default_stream_sync(device), "upload is not complete on the owning device");
  }
}
void peer_contract(bool prepared, bool reverse) {
  test::reset_api_trace();
  auto p = edmir_generated::program();
  require(p.dependency_policy == DependencyPolicy::kUniformImmutablePeerV1, "fixture must be emitted with restricted policy");
  CudaBackendOptions options;
  options.prepare_metadata = prepared;
  for (int d=0; d<p.problem.world_size; ++d) options.devices.push_back(reverse ? p.problem.world_size-1-d : d);
  {
    CudaBackend backend(p, options);
    backend.prepare();
    // At prepare return all default-stream initialization is complete.
    const auto setup = test::api_trace();
    for (int d=0; d<p.problem.world_size; ++d)
      require(std::find(setup.begin(), setup.end(), default_stream_sync(d)) != setup.end(), "workspace initialization not completed");
    rejects([&] { backend.execute(); });
    unsigned char dummy = 0;
    const auto w = uniform_workload(p);
    const auto input = w.inputs[0];
    rejects([&] { backend.copy_from_host(input, &dummy, 1); });
    rejects([&] { backend.buffer_data(input); });
    require(std::as_const(backend).buffer_data(input) != nullptr, "const buffer accessor used mutable overload");
    backend.copy_from_host(input, &dummy, p.buffers[input].size_bytes);
    backend.copy_from_host(input, &dummy, p.buffers[input].size_bytes);
    rejects([&] { backend.execute(); }); // duplicate uploads must not satisfy missing matrices
    upload(backend,p,reverse);
    test::clear_api_trace();
    backend.execute();
    const auto calls = test::api_call_count();
    rejects([&] { backend.execute(); });
    rejects([&] { backend.copy_from_host(input, &dummy, p.buffers[input].size_bytes); });
    rejects([&] { backend.copy_to_host(w.outputs[0], &dummy, p.buffers[w.outputs[0]].size_bytes); });
    require(test::api_call_count() == calls, "rejected reuse submitted API work");
    backend.synchronize();
    const auto first = test::api_trace();
    require(count(first,"fused_cutlass ") == 16 && count(first,"wait_event ") == 36, "unexpected restricted execution graph");
    require(count(first,"sync_stream ") == 12, "all-rank completion was removed");
    std::set<std::string> workspaces;
    for (const auto& call:first) if (call.starts_with("fused_cutlass ")) {
      std::istringstream words(call); std::string value;
      for(int field=0;field<=11;++field) words>>value;
      workspaces.insert(value);
    }
    require(workspaces.size()==16, "split-K operations share workspace storage");
    test::count_api_calls_only();
    test::clear_api_trace();
    const auto start = std::chrono::steady_clock::now();
    for (int i=0;i<1000;++i) {
      backend.execute(); backend.synchronize();
      require(test::api_call_count() == (i+1)*first.size(), "iteration dispatch count drift");
    }
    std::cout << "restricted policy CPU dispatch: prepared=" << prepared << " reverse=" << reverse
              << " calls/iteration=" << first.size() << " 1000_iterations_ms="
              << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() << '\n';
    require(test::api_trace().empty(), "bounded iteration test accumulated strings");
  }
  // Explicit destruction with work in flight must complete before release.
  test::reset_api_trace();
  {
    CudaBackend backend(p,options); backend.prepare(); upload(backend,p,reverse);
    backend.execute(); test::clear_api_trace();
  }
  const auto teardown = test::api_trace();
  const auto first_free = std::find_if(teardown.begin(),teardown.end(),[](const auto& s) {
    return s.starts_with("free ") || s.starts_with("nccl_free ") || s.starts_with("deregister_window ");
  });
  require(std::count_if(teardown.begin(),first_free,[](const auto& s){return s.starts_with("sync_device ");}) == 4,
          "buffer/window release precedes device completion");
}
void reject_graph_changes() {
  auto p=edmir_generated::program();
  std::vector<DependencyDesc> edges(p.dependencies.begin(),p.dependencies.end());
  edges.pop_back(); p.dependencies=edges;
  test::reset_api_trace();
  rejects([&] { CudaBackend backend(p); backend.prepare(); });
  require(test::api_trace().empty(), "bad graph reached CUDA allocation");
  p=edmir_generated::program();
  std::vector<AccessDesc> accesses(p.accesses.begin(),p.accesses.end());
  const auto w=uniform_workload(p);
  for(auto& access:accesses) if(access.write && access.buffer==w.outputs[4]) access.buffer=w.outputs[0];
  p.accesses=accesses;
  rejects([&] { CudaBackend backend(p); backend.prepare(); });
  p=edmir_host_generated::program(); p.dependency_policy=DependencyPolicy::kUniformImmutablePeerV1;
  rejects([&] { CudaBackend backend(p); backend.prepare(); });
  rejects([&] { MergedHostBackend backend(p,MergedProvider::kCUTLASS); });
}
void host_contract() {
  const auto p=edmir_host_generated::program();
  for (auto provider : {MergedProvider::kCUTLASS, MergedProvider::kCuBLASLt}) {
    test::reset_api_trace();
    MergedHostBackend backend(p,provider); backend.prepare(); upload(backend,p);
    test::clear_api_trace(); backend.execute(); backend.synchronize();
    const auto trace = test::api_trace();
    require(count(trace,"send ")==12 && count(trace,"recv ")==12 && count(trace,"copy_async ")==4, "merged communication mismatch");
    require(count(trace,"sync_stream ")==8 && count(trace,"wait_event ")==4, "merged completion mismatch");
    require(count(trace,provider==MergedProvider::kCUTLASS ? "cutlass_gemm " : "lt_gemm ")==4, "merged launch count mismatch");
    require(count(trace,"lt_heuristic_synthetic")==0, "heuristic entered steady timing");
    require(!backend.algorithm_id().empty(), "algorithm identity missing");
  }
  for (bool blas : {false,true}) {
    auto host=p;
    if(blas) { host.gemm_provider=GemmProvider::kCuBLAS; host.gemm_kernel=GemmKernel::kLibraryDefault; }
    test::reset_api_trace();
    CudaBackend backend(host); backend.prepare(); upload(backend,host);
    test::clear_api_trace(); backend.execute(); backend.synchronize();
    require(count(test::api_trace(),blas ? "blas_gemm " : "cutlass_gemm ")==16, "per-source host launch count mismatch");
  }
}
}
int main() {
  try {
    for(bool prepared:{false,true}) for(bool reverse:{false,true}) peer_contract(prepared,reverse);
    reject_graph_changes(); host_contract();
    std::cout << "execution contract tests passed; CPU API stubs, no GPU numerics or timings\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
