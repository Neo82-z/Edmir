#include "edmir_batch.hpp"
#include "cuda_api_trace.hpp"
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/nccl_group.hpp"
#include "edmir/runtime/merged_host_backend.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace edmir::runtime;

int main() {
  try {
    test::reset_api_trace();
    std::vector<int> devices(8);
    std::iota(devices.begin(), devices.end(), 0);
    auto group = std::make_shared<NcclGroup>(devices);
    for (auto c : edmir_batch::candidates) {
      if (c.name == "host_merged_cutlass" || c.name == "host_merged_cublaslt") {
        MergedHostBackend backend(c.program(), c.name == "host_merged_cutlass" ?
                                    MergedProvider::kCUTLASS : MergedProvider::kCuBLASLt, group);
        backend.prepare();
      } else {
        CudaBackendOptions opts;
        opts.group = group;
        opts.prepare_metadata = true;
        opts.pipeline_control_order = c.program().strategy == "pipelined";
        CudaBackend backend(c.program(), opts);
        backend.prepare();
      }
    }
    const auto count = [](std::string_view prefix) {
      const auto trace = test::api_trace();
      return std::count_if(trace.begin(),trace.end(),[&](auto& e){return e.starts_with(prefix);});
    };
    if (count("comm_init ")!=8 || count("comm_destroy ")!=0) throw std::runtime_error("group lifetime");
    if (count("register_window ")==0 || count("register_window ")!=count("deregister_window "))
      throw std::runtime_error("peer windows must be released between candidates");
    std::reverse(devices.begin(), devices.end());
    try { group->for_devices(devices); return 2; } catch (const std::invalid_argument&) {}
    group.reset();
    if (count("comm_destroy ")!=8) throw std::runtime_error("group cleanup");
    std::cout << "one communicator set survived all candidates; windows and handles released\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
