#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "edmir/runtime/cuda_backend.hpp"
#include "edmir_program.hpp"

namespace {

bool is_external(edmir::runtime::BufferRole role) {
  return role == edmir::runtime::BufferRole::kInput ||
         role == edmir::runtime::BufferRole::kWeight;
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

int positive_argument(int argc, char** argv, const std::string& name, int fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      const auto value = std::stoi(argv[index + 1]);
      if (value <= 0) throw std::invalid_argument(name + " must be positive");
      return value;
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  return std::any_of(argv + 1, argv + argc, [&](const char* value) { return value == name; });
}

std::size_t op_device_count(const edmir::runtime::OpDesc& op) {
  return op.kind == edmir::runtime::OpKind::kTransfer && op.peer_rank != op.rank ? 2 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int warmup_iterations = positive_argument(argc, argv, "--warmup", 5);
    const int measured_iterations = positive_argument(argc, argv, "--iterations", 30);
    const auto program = edmir_generated::program();
    edmir::runtime::CudaBackend backend(
        program, {.devices = {}, .enable_nvtx = has_flag(argc, argv, "--nvtx")});
    backend.prepare();

    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      const auto& buffer = program.buffers[index];
      if (!is_external(buffer.role) || buffer.size_bytes == 0) {
        continue;
      }
      const std::vector<unsigned char> zeros(buffer.size_bytes, 0);
      backend.copy_from_host(index, zeros.data(), zeros.size());
    }

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      backend.execute();
      backend.synchronize();
    }

    std::vector<double> milliseconds;
    milliseconds.reserve(measured_iterations);
    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      backend.execute();
      backend.synchronize();
      const auto stop = std::chrono::steady_clock::now();
      milliseconds.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto mean = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
                      static_cast<double>(milliseconds.size());
    std::size_t event_records = 0;
    for (const auto& op : program.ops) event_records += op_device_count(op);
    std::vector<std::unordered_set<std::size_t>> predecessors(program.ops.size());
    std::size_t buffer_reuses = 0;
    for (const auto& dependency : program.dependencies) {
      predecessors[dependency.dst_op].insert(dependency.src_op);
      if (dependency.kind == edmir::runtime::DependencyKind::kBufferReuse) ++buffer_reuses;
    }
    std::size_t event_waits = 0;
    for (std::size_t op = 0; op < program.ops.size(); ++op) {
      event_waits += predecessors[op].size() * op_device_count(program.ops[op]);
    }

    std::cout << std::fixed << std::setprecision(4)
              << "bench target=" << program.target << " arch=" << program.arch
              << " gemm_provider=" << edmir::runtime::gemm_provider_name(program.gemm_provider)
              << " gemm_kernel=" << edmir::runtime::gemm_kernel_name(program.gemm_kernel)
              << " strategy=" << program.strategy << " ranks=" << program.problem.world_size
              << " rows_per_peer=" << program.problem.rows_per_peer
              << " k=" << program.problem.k << " n=" << program.problem.n
              << " chunk_rows=" << program.chunk_rows
              << " buffer_count=" << program.buffer_count
              << " ops=" << program.ops.size() << " op_dispatch=" << program.ops.size()
              << " event_record=" << event_records << " event_wait=" << event_waits
              << " buffer_reuse=" << buffer_reuses << " warmup=" << warmup_iterations
              << " iterations=" << measured_iterations << " min_ms=" << milliseconds.front()
              << " median_ms=" << percentile(milliseconds, 0.50)
              << " p95_ms=" << percentile(milliseconds, 0.95)
              << " max_ms=" << milliseconds.back() << " mean_ms=" << mean << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edmir benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
