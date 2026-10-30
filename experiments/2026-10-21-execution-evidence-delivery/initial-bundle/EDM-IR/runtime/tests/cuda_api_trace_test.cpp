#include <chrono>
#include <iostream>
#include <stdexcept>
#include "cuda_api_trace.hpp"
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir_program.hpp"

namespace {
using namespace edmir::runtime;
struct Run { std::vector<std::string> preparation, execution; };
Run capture(bool prepared, std::vector<int> devices) {
  test::reset_api_trace();
  CudaBackendOptions options;
  options.devices = std::move(devices);
  options.prepare_metadata = prepared;
  CudaBackend backend(edmir_generated::program(), options);
  backend.prepare();
  auto setup = test::api_trace();
  test::clear_api_trace();
  for (int iteration = 0; iteration < 2; ++iteration) {
    backend.execute();
    backend.synchronize();
  }
  return {setup, test::api_trace()};
}
void equal(const std::vector<std::string>& before, const std::vector<std::string>& after) {
  if (before.empty() || before.size() != after.size())
    throw std::runtime_error("API call counts differ or trace is empty");
  for (std::size_t i = 0; i < before.size(); ++i)
    if (before[i] != after[i])
      throw std::runtime_error("API call " + std::to_string(i) + " differs:\n" + before[i] + "\n" + after[i]);
}
void repeated_execution(bool prepared, std::size_t calls_per_iteration) {
  test::reset_api_trace();
  test::count_api_calls_only();
  CudaBackendOptions options;
  options.prepare_metadata = prepared;
  CudaBackend backend(edmir_generated::program(), options);
  backend.prepare();
  test::clear_api_trace();
  constexpr std::size_t iterations = 1000;
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    backend.execute();
    backend.synchronize();
    if (test::api_call_count() != (i+1)*calls_per_iteration)
      throw std::runtime_error("repeated execution changed API call count");
  }
  const auto elapsed = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now()-start).count();
  if (!test::api_trace().empty())
    throw std::runtime_error("count-only API test retained trace strings");
  std::cout << "{\"test\":\"cpu_stub_dispatch_only\",\"mode\":\""
            << (prepared ? "prepared" : "legacy") << "\",\"iterations\":" << iterations
            << ",\"calls_per_iteration\":" << calls_per_iteration
            << ",\"us_per_iteration\":" << elapsed/iterations << "}\n";
}
}

int main() {
  try {
    const auto count = edmir_generated::program().problem.world_size;
    for (bool reverse : {false, true}) {
      std::vector<int> devices;
      for (int i = 0; i < count; ++i) devices.push_back(reverse ? count-1-i : i);
      const auto before = capture(false, devices), after = capture(true, devices);
      equal(before.preparation, after.preparation);
      equal(before.execution, after.execution);
      std::cout << (reverse ? "reversed" : "identity") << " device map: "
                << after.execution.size() << " execute/synchronize API calls match across two iterations\n";
      if (!reverse) {
        repeated_execution(false, after.execution.size()/2);
        repeated_execution(true, after.execution.size()/2);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
