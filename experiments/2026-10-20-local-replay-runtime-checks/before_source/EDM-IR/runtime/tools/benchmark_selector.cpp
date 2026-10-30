// Full-operator measurement for the prepared selector package.
// Inputs and CPU-FP32 references are generated once, outside GPU timing.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <nccl.h>
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/merged_host_backend.hpp"
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir_program.hpp"

namespace {
using namespace edmir::runtime;
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
std::string quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    if (c < 32) throw std::invalid_argument("unexpected control character in metadata");
    out += static_cast<char>(c);
  }
  return out+'"';
}
template<class T>
std::vector<T> read(std::ifstream& stream, std::size_t count) {
  std::vector<T> values(count);
  if (!stream.read(reinterpret_cast<char*>(values.data()), count*sizeof(T)))
    throw std::runtime_error("truncated experiment data file");
  return values;
}
void require_end(std::ifstream& stream) {
  if (stream.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("extra bytes in experiment data file");
}
struct Accuracy {
  double absolute = 0, relative = 0;
  std::size_t checked = 0, nonfinite = 0;
  bool passed() const { return !nonfinite && (absolute <= 0.02 || relative <= 0.02); }
};

template<class Backend>
Accuracy verify(Backend& backend, ProgramView p, const UniformWorkload& work, const std::string& path) {
  std::ifstream reference(path, std::ios::binary);
  if (!reference) throw std::runtime_error("cannot open CPU reference");
  Accuracy result;
  const auto count = static_cast<std::size_t>(p.problem.rows_per_peer)*p.problem.n;
  for (const auto output : work.outputs) {
    const auto expected = read<float>(reference, count);
    std::vector<__half> actual(count);
    backend.copy_to_host(output, actual.data(), count*sizeof(__half));
    for (std::size_t i = 0; i < count; ++i) {
      if (!std::isfinite(expected[i])) throw std::runtime_error("CPU reference is nonfinite");
      const double value = __half2float(actual[i]);
      ++result.checked;
      if (!std::isfinite(value)) { ++result.nonfinite; continue; }
      const double error = std::abs(value-expected[i]);
      result.absolute = std::max(result.absolute, error);
      result.relative = std::max(result.relative, error/std::max(std::abs(double(expected[i])), 1.0));
    }
  }
  require_end(reference);
  return result;
}

std::string implementation(ProgramView p, const UniformWorkload& w, const std::string& family) {
  if (family == "host_merged_cutlass") return "host_nccl_cutlass_sm80_merged_v1";
  if (family == "host_merged_cublaslt") return "host_nccl_cublaslt_merged_v1";
  if (!w.fused) {
    if (p.strategy != "sequential") throw std::invalid_argument("host schedule is not sequential");
    if (family == "host" && p.gemm_provider == GemmProvider::kCUTLASS &&
        p.gemm_kernel == GemmKernel::kSm80_64x128x32)
      return "host_nccl_cutlass_sm80_per_source_v1";
    if (family == "host_compiler_cublas" && p.gemm_provider == GemmProvider::kCuBLAS)
      return "host_nccl_cublas_per_source_v1";
  } else if (p.gemm_provider == GemmProvider::kCUTLASS) {
    const int q = gemm_kernel_split_k_slices(p.gemm_kernel), g = p.fused_schedule.n_tiles_per_cta;
    const auto expected = (q == 1 ? std::string("direct") : "splitk"+std::to_string(q))+
        (g == 1 ? "" : "_groupedn2");
    if (family == expected)
      return "direct_peer_cutlass_sm80_q"+std::to_string(q)+"_g"+std::to_string(g);
  }
  throw std::invalid_argument("family label does not match the compiled program");
}

std::string runtime_json(ProgramView p);

template<class Backend>
void measure(Backend& backend, ProgramView p, const UniformWorkload& w, const std::string& family,
             const std::string& inputs, const std::string& reference, int warmup, int iterations,
             const std::string& metadata_mode) {
  auto identity = implementation(p, w, family);
  if (metadata_mode == "prepared") identity += "_prepared_metadata_v2";
  backend.prepare();
  std::ifstream file(inputs, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open experiment inputs");
  for (const auto* ids : {&w.inputs, &w.weights}) {
    for (auto id : *ids) {
      const auto values = read<unsigned char>(file, p.buffers[id].size_bytes);
      backend.copy_from_host(id, values.data(), values.size());
    }
  }
  require_end(file);
  backend.execute(); backend.synchronize();
  const auto before = verify(backend, p, w, reference);
  if (!before.passed()) throw std::runtime_error("initial numerical correctness failed");
  for (int i = 0; i < warmup; ++i) { backend.execute(); backend.synchronize(); }
  std::vector<double> times;
  times.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    backend.execute(); backend.synchronize();
    const auto end = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end-start).count());
  }
  const auto after = verify(backend, p, w, reference);
  if (!after.passed()) throw std::runtime_error("post-timing numerical correctness failed");
  auto sorted = times;
  std::sort(sorted.begin(), sorted.end());
  const auto n = sorted.size();
  const double median = (sorted[(n-1)/2]+sorted[n/2])/2;
  const double total_ms = std::accumulate(times.begin(), times.end(), 0.0);
  std::string algorithm = std::string(gemm_kernel_name(p.gemm_kernel));
  if constexpr (requires { backend.algorithm_id(); }) algorithm = backend.algorithm_id();
  std::cout << std::setprecision(17)
    << "{\"family\":" << quote(family) << ",\"implementation_id\":" << quote(identity)
    << ",\"gemm_algorithm_id\":" << quote(algorithm)
    << ",\"shape\":[" << p.problem.world_size << ',' << p.problem.rows_per_peer << ','
    << p.problem.k << ',' << p.problem.n << "],\"warmup\":" << warmup << ",\"iterations\":" << iterations
    << ",\"median_ms\":" << median << ",\"correctness_passed\":true"
    << ",\"mean_ms\":" << total_ms/n << ",\"total_ms\":" << total_ms
    << ",\"min_ms\":" << sorted.front() << ",\"max_ms\":" << sorted.back()
    << ",\"metadata_mode\":" << quote(metadata_mode)
    << ",\"timing_scope\":\"host steady_clock around backend.execute() plus backend.synchronize()\""
    << ",\"measurement_policy\":{\"profiler\":false,\"nvtx\":false,\"discarded_samples\":0}"
    << ",\"max_abs_error\":" << std::max(before.absolute, after.absolute)
    << ",\"max_rel_error\":" << std::max(before.relative, after.relative)
    << ",\"checked_elements\":" << after.checked << ",\"nonfinite_outputs\":0"
    << ",\"atol\":0.02,\"rtol\":0.02,\"input_seed\":20261020"
    << ",\"correctness_reference\":\"CPU_FP32_on_identical_FP16_inputs\""
    << ",\"runtime\":" << runtime_json(p) << ",\"samples_ms\":[";
  for (std::size_t i = 0; i < times.size(); ++i) { if (i) std::cout << ','; std::cout << times[i]; }
  std::cout << "]}\n";
}

std::string runtime_json(ProgramView p) {
  int driver = 0, cuda = 0, nccl = 0, blas = 0;
  cuda_check(cudaDriverGetVersion(&driver));
  cuda_check(cudaRuntimeGetVersion(&cuda));
  if (ncclGetVersion(&nccl) != ncclSuccess) throw std::runtime_error("cannot get NCCL version");
  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cannot get cuBLAS version");
  const auto status = cublasGetVersion(handle, &blas);
  cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cannot get cuBLAS version");
  std::ostringstream device_json;
  device_json << '[';
  for (int d = 0; d < p.problem.world_size; ++d) {
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, d));
    char bus_id[32]{};
    cuda_check(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), d));
    if (d) device_json << ',';
    device_json << "{\"name\":" << quote(property.name) << ",\"pci_bus_id\":" << quote(bus_id)
                << ",\"major\":" << property.major
                << ",\"minor\":" << property.minor << ",\"sm_count\":" << property.multiProcessorCount
                << ",\"total_memory\":" << property.totalGlobalMem << ",\"peers\":[";
    for (int peer = 0; peer < p.problem.world_size; ++peer) {
      int access = 1;
      if (peer != d) cuda_check(cudaDeviceCanAccessPeer(&access, d, peer));
      if (peer) device_json << ',';
      device_json << (access ? "true" : "false");
    }
    device_json << "]}";
  }
  device_json << ']';
  std::ostringstream result;
  result << "{\"driver_version\":" << quote(std::to_string(driver))
    << ",\"cuda_version\":" << quote(std::to_string(cuda)) << ",\"nccl_version\":" << quote(std::to_string(nccl))
    << ",\"cublas_version\":" << quote(std::to_string(blas)) << ",\"devices\":" << device_json.str() << "}";
  return result.str();
}
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--inspect") {
      std::cout << runtime_json(edmir_generated::program()) << '\n';
      return 0;
    }
    if (argc != 6 && argc != 7)
      throw std::invalid_argument("usage: executable FAMILY INPUTS_FP16 REFERENCE_FP32 WARMUP ITERATIONS [legacy|prepared|merged]");
    if constexpr (std::endian::native != std::endian::little)
      throw std::invalid_argument("experiment binary format requires little endian");
    const std::string family = argv[1];
    const bool merged = family == "host_merged_cutlass" || family == "host_merged_cublaslt";
    const std::string metadata_mode = argc == 7 ? argv[6] : (merged ? "merged" : "legacy");
    if (merged ? metadata_mode != "merged" : (metadata_mode != "legacy" && metadata_mode != "prepared"))
      throw std::invalid_argument("metadata mode does not match backend");
    const int warmup = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
    if (warmup < 20 || iterations < 100) throw std::invalid_argument("minimum warmup=20, iterations=100");
    const auto p = edmir_generated::program();
    const auto w = uniform_workload(p);
    if (merged) {
      MergedHostBackend backend(p, family == "host_merged_cutlass" ? MergedProvider::kCUTLASS : MergedProvider::kCuBLASLt);
      measure(backend, p, w, family, argv[2], argv[3], warmup, iterations, metadata_mode);
    } else {
      CudaBackendOptions options;
      options.prepare_metadata = metadata_mode == "prepared";
      CudaBackend backend(p, options);
      measure(backend, p, w, family, argv[2], argv[3], warmup, iterations, metadata_mode);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "selector benchmark failed: " << e.what() << '\n';
    return 1;
  }
}
