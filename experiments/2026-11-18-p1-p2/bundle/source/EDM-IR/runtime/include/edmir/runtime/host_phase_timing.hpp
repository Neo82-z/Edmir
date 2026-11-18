#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace edmir::runtime {

struct HostPhaseSample {
  std::int64_t submit_ns;
  std::int64_t completion_wait_ns;
  std::int64_t wall_ns;
};

// Diagnostic window only: the extra timestamp changes the measurement path.
// Submit can include blocking API calls and concurrent device execution.
// Completion wait is remaining host wait, NOT total device service time.
// Caller must prepare inputs, check correctness and warm up first.
template<class Backend, class Now>
std::vector<HostPhaseSample> measure_host_phases(Backend& backend, int count, Now now) {
  if (count < 1 || count > 64) throw std::invalid_argument("phase sample count must be 1..64");
  std::vector<HostPhaseSample> samples;
  samples.reserve(count);
  for (int i = 0; i < count; ++i) {
    const auto start = now();
    backend.execute();
    const auto submitted = now();
    backend.synchronize();
    const auto finished = now();
    if (submitted < start || finished < submitted)
      throw std::runtime_error("phase clock is not monotonic");
    // Integer timestamps share one origin so the partition identity is exact.
    const auto a = std::chrono::duration_cast<std::chrono::nanoseconds>(submitted-start).count();
    const auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(finished-start).count();
    samples.push_back({a, total-a, total});
  }
  return samples;
}

template<class Backend>
std::vector<HostPhaseSample> measure_host_phases(Backend& backend, int count) {
  return measure_host_phases(backend, count, [] { return std::chrono::steady_clock::now(); });
}

}  // namespace edmir::runtime
