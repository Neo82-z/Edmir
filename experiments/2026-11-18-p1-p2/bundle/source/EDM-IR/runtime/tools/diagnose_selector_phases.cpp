// Separate diagnostic executable. Never emits a selector latency-label row.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/host_phase_timing.hpp"
#include "edmir/runtime/uniform_workload.hpp"
#include "edmir_program.hpp"

namespace {
using namespace edmir::runtime;
template<class T> std::vector<T> read(std::ifstream& f, std::size_t count) {
  std::vector<T> data(count);
  if (!f.read(reinterpret_cast<char*>(data.data()), count*sizeof(T))) throw std::runtime_error("truncated input/reference");
  return data;
}
void end(std::ifstream& f) {
  if (f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("extra input/reference bytes");
}
double half_value(std::uint16_t bits) {
  const int exp = (bits >> 10) & 31, frac = bits & 1023;
  const double sign = bits & 32768 ? -1.0 : 1.0;
  if (exp == 31) return frac ? std::numeric_limits<double>::quiet_NaN() : sign*std::numeric_limits<double>::infinity();
  return sign*(exp ? std::ldexp(1.+frac/1024., exp-15) : std::ldexp(double(frac), -24));
}
void verify(CudaBackend& backend, ProgramView p, const UniformWorkload& w, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read reference");
  const auto count = std::size_t(p.problem.rows_per_peer)*p.problem.n;
  double absolute = 0., normalized = 0.;
  for (auto id : w.outputs) {
    auto reference = read<float>(f,count);
    std::vector<std::uint16_t> output(count);
    backend.copy_to_host(id,output.data(),count*sizeof(std::uint16_t));
    for (std::size_t i = 0; i < count; ++i) {
      const double value = half_value(output[i]);
      if (!std::isfinite(value) || !std::isfinite(reference[i])) throw std::runtime_error("nonfinite correctness data");
      const double error = std::abs(value-reference[i]);
      absolute = std::max(absolute,error);
      normalized = std::max(normalized,error/std::max(std::abs(double(reference[i])),1.));
    }
  }
  end(f);
  if (!(absolute <= .02 || normalized <= .02)) throw std::runtime_error("numerical correctness failed");
}
std::string family(ProgramView p, const UniformWorkload& w) {
  if (p.problem.dtype != DType::kFloat16 || p.gemm_provider != GemmProvider::kCUTLASS ||
      p.dependency_policy != DependencyPolicy::kResourceQueuesV1)
    throw std::invalid_argument("only FP16 per-source CUTLASS resource_queues_v1 diagnostic supported");
  if (!w.fused) {
    if (p.strategy != "sequential" || p.gemm_kernel != GemmKernel::kSm80_64x128x32)
      throw std::invalid_argument("host implementation differs");
    return "host";
  }
  const auto q = gemm_kernel_split_k_slices(p.gemm_kernel), g = p.fused_schedule.n_tiles_per_cta;
  if (gemm_kernel_threadblock_m(p.gemm_kernel) != 64 || gemm_kernel_threadblock_n(p.gemm_kernel) != 128 || (g != 1 && g != 2))
    throw std::invalid_argument("fused template differs");
  return (q == 1 ? std::string("direct") : "splitk"+std::to_string(q))+(g == 2 ? "_groupedn2" : "");
}
int integer(const char* raw) {
  std::string s(raw); std::size_t used = 0; int n = std::stoi(s,&used);
  if (used != s.size()) throw std::invalid_argument("invalid integer");
  return n;
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 7) throw std::invalid_argument("usage: phases FAMILY INPUT_FP16 REFERENCE_FP32 WARMUP SAMPLES legacy|prepared");
    if constexpr (std::endian::native != std::endian::little) throw std::invalid_argument("little endian required");
    const int warmup = integer(argv[4]), count = integer(argv[5]);
    const std::string mode = argv[6];
    if (warmup < 20 || warmup > 1000 || count < 1 || count > 64 || (mode != "legacy" && mode != "prepared"))
      throw std::invalid_argument("invalid diagnostic bounds/mode");
    const auto p = edmir_generated::program(); const auto w = uniform_workload(p);
    const auto compiled_family = family(p,w);
    if (compiled_family != argv[1]) throw std::invalid_argument("family does not match compiled program");
    CudaBackendOptions options; options.prepare_metadata = mode == "prepared";
    CudaBackend backend(p, options); backend.prepare();
    std::ifstream f(argv[2],std::ios::binary);
    if (!f) throw std::runtime_error("cannot read inputs");
    for (const auto* ids : {&w.inputs,&w.weights}) for (auto id : *ids) {
      auto data = read<unsigned char>(f,p.buffers[id].size_bytes);
      backend.copy_from_host(id,data.data(),data.size());
    }
    end(f); backend.execute(); backend.synchronize(); verify(backend,p,w,argv[3]);
    for (int i = 0; i < warmup; ++i) { backend.execute(); backend.synchronize(); }
    auto samples = measure_host_phases(backend,count);
    verify(backend,p,w,argv[3]);
    std::cout << "{\"kind\":\"host_phase_diagnostic\",\"training_eligible\":false,\"correctness_passed\":true"
      << ",\"family\":\"" << compiled_family << "\",\"metadata_mode\":\"" << mode << "\""
      << ",\"dependency_policy\":\"resource_queues_v1\",\"precision\":\"fp16_fp32acc_fp16out\""
      << ",\"shape\":[" << p.problem.world_size << ',' << p.problem.rows_per_peer << ',' << p.problem.k << ',' << p.problem.n << ']'
      << ",\"warmup\":" << warmup << ",\"nvtx_enabled\":false,\"external_profiler_status\":\"not_detected_by_executable\""
      << ",\"instrumentation\":\"extra host timestamp after execute\",\"device_service_us\":null,\"samples_ns\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& s = samples[i]; if (i) std::cout << ',';
      std::cout << "{\"submit\":" << s.submit_ns << ",\"completion_wait\":" << s.completion_wait_ns << ",\"wall\":" << s.wall_ns << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "phase diagnostic failed: " << e.what() << '\n'; return 1;
  }
}
