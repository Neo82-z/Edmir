#include "edmir/runtime/merged_host_backend.hpp"
#include "edmir/runtime/uniform_workload.hpp"

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <nccl.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#if defined(EDMIR_HAS_CUTLASS)
#include "edmir/runtime/cutlass_gemm.hpp"
#endif

namespace edmir::runtime {
namespace {
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt status "+std::to_string(s));
}
void check(ncclResult_t s) {
  if (s != ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
constexpr std::size_t workspace_bytes = 32ULL*1024*1024;
}

class MergedHostBackend::Impl {
 public:
  struct Rank {
    ncclComm_t comm = nullptr;
    cudaStream_t communication = nullptr, compute = nullptr;
    cudaEvent_t ready = nullptr;
    void* received = nullptr;
    void* output = nullptr;
    void* workspace = nullptr;
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    std::string algorithm;
  };
  ProgramView program;
  MergedProvider provider;
  UniformWorkload work;
  std::vector<Rank> ranks;
  std::vector<void*> external;
  bool prepared = false;

  Impl(ProgramView p, MergedProvider v)
      : program(p), provider(v), work(uniform_workload(p)), ranks(p.problem.world_size),
        external(p.buffers.size(), nullptr) {
    if (work.fused || p.dependency_policy != DependencyPolicy::kResourceQueuesV1 ||
        p.communication_api != CommunicationApi::kHostNCCL ||
        p.gemm_provider != GemmProvider::kCUTLASS ||
        p.gemm_kernel != GemmKernel::kSm80_64x128x32)
      throw std::invalid_argument("merged baseline requires the unchunked CUTLASS host manifest");
  }

  ~Impl() {
    // synchronize() is normally called by the runner. Best-effort cleanup also
    // covers partial prepare() failures; destruction must not throw.
    for (std::size_t d = 0; d < ranks.size(); ++d) {
      cudaSetDevice(static_cast<int>(d));
      auto& r = ranks[d];
      if (r.compute) cudaStreamSynchronize(r.compute);
      if (r.communication) cudaStreamSynchronize(r.communication);
      if (r.comm) ncclCommDestroy(r.comm);
      if (r.preference) cublasLtMatmulPreferenceDestroy(r.preference);
      if (r.operation) cublasLtMatmulDescDestroy(r.operation);
      if (r.a) cublasLtMatrixLayoutDestroy(r.a);
      if (r.b) cublasLtMatrixLayoutDestroy(r.b);
      if (r.c) cublasLtMatrixLayoutDestroy(r.c);
      if (r.lt) cublasLtDestroy(r.lt);
      if (r.workspace) cudaFree(r.workspace);
      if (r.received) cudaFree(r.received);
      if (r.output) cudaFree(r.output);
      if (r.ready) cudaEventDestroy(r.ready);
      if (r.compute) cudaStreamDestroy(r.compute);
      if (r.communication) cudaStreamDestroy(r.communication);
    }
    for (std::size_t i = 0; i < external.size(); ++i) {
      if (!external[i]) continue;
      cudaSetDevice(program.buffers[i].rank);
      cudaFree(external[i]);
    }
  }

  void prepare_lt(Rank& r) {
    const auto& p = program.problem;
    const int m = p.world_size*p.rows_per_peer;
    check(cublasLtCreate(&r.lt));
    check(cublasLtMatmulDescCreate(&r.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    check(cublasLtMatrixLayoutCreate(&r.a, CUDA_R_16F, m, p.k, p.k));
    check(cublasLtMatrixLayoutCreate(&r.b, CUDA_R_16F, p.k, p.n, p.n));
    check(cublasLtMatrixLayoutCreate(&r.c, CUDA_R_16F, m, p.n, p.n));
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    for (auto layout : {r.a, r.b, r.c})
      check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    check(cublasLtMatmulPreferenceCreate(&r.preference));
    check(cublasLtMatmulPreferenceSetAttribute(r.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_bytes, sizeof(workspace_bytes)));
    int returned = 0;
    // One documented heuristic choice, outside timing. No measurements or
    // test labels select an algorithm. A failure is never replaced by cuBLAS.
    check(cublasLtMatmulAlgoGetHeuristic(r.lt, r.operation, r.a, r.b, r.c, r.c,
                                       r.preference, 1, &r.heuristic, &returned));
    if (returned != 1 || r.heuristic.state != CUBLAS_STATUS_SUCCESS ||
        r.heuristic.workspaceSize > workspace_bytes)
      throw std::runtime_error("no cuBLASLt heuristic fits the frozen workspace limit");
    check(cudaMalloc(&r.workspace, workspace_bytes));
    const cublasLtMatmulAlgoConfigAttributes_t attributes[] = {
      CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_STAGES_ID,
    };
    std::ostringstream identity;
    for (auto attribute : attributes) {
      int value = 0;
      std::size_t written = 0;
      check(cublasLtMatmulAlgoConfigGetAttribute(&r.heuristic.algo, attribute, &value, sizeof(value), &written));
      if (written != sizeof(value)) throw std::runtime_error("unexpected cuBLASLt algorithm attribute size");
      identity << static_cast<int>(attribute) << '=' << value << ';';
    }
    identity << "workspace=" << r.heuristic.workspaceSize << ";budget=" << workspace_bytes;
    r.algorithm = identity.str();
  }

  void prepare() {
    if (prepared) throw std::logic_error("merged backend already prepared");
#if !defined(EDMIR_HAS_CUTLASS)
    if (provider == MergedProvider::kCUTLASS)
      throw std::runtime_error("CUTLASS merged baseline was not built");
#endif
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count < program.problem.world_size) throw std::runtime_error("insufficient visible GPUs");
    std::vector<ncclComm_t> comms(ranks.size());
    std::vector<int> devices(ranks.size());
    std::iota(devices.begin(), devices.end(), 0);
    const auto initialized = ncclCommInitAll(comms.data(), static_cast<int>(ranks.size()), devices.data());
    for (std::size_t d = 0; d < ranks.size(); ++d) ranks[d].comm = comms[d];
    check(initialized);
    const auto& p = program.problem;
    for (std::size_t d = 0; d < ranks.size(); ++d) {
      check(cudaSetDevice(static_cast<int>(d)));
      auto& r = ranks[d];
      check(cudaStreamCreateWithFlags(&r.communication, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&r.compute, cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&r.ready, cudaEventDisableTiming));
      check(cudaMalloc(&r.received, 2ULL*p.world_size*p.rows_per_peer*p.k));
      check(cudaMalloc(&r.output, 2ULL*p.world_size*p.rows_per_peer*p.n));
      if (provider == MergedProvider::kCuBLASLt) prepare_lt(r);
    }
    for (std::size_t i = 0; i < external.size(); ++i) {
      const auto& b = program.buffers[i];
      if (b.role != BufferRole::kInput && b.role != BufferRole::kWeight) continue;
      check(cudaSetDevice(b.rank));
      check(cudaMalloc(&external[i], b.size_bytes));
    }
    prepared = true;
  }

  void require_ready() const {
    if (!prepared) throw std::logic_error("merged backend is not prepared");
  }

  void execute() {
    require_ready();
    const auto& p = program.problem;
    const auto elements = static_cast<std::size_t>(p.rows_per_peer)*p.k;
    // Preserve the host baseline's cyclic source/destination order and one
    // send/recv group per remote pair. Only the destination storage is merged.
    for (int step = 0; step < p.world_size; ++step) {
      for (int destination = 0; destination < p.world_size; ++destination) {
        const int source = (destination+step)%p.world_size;
        const auto input = external[work.inputs[source*p.world_size+destination]];
        auto& dst = ranks[destination];
        auto* received = static_cast<unsigned char*>(dst.received)+
            2*merged_row_offset(source, p.rows_per_peer, p.world_size)*p.k;
        if (source == destination) {
          check(cudaSetDevice(destination));
          check(cudaMemcpyAsync(received, input, elements*2, cudaMemcpyDeviceToDevice, dst.communication));
        } else {
          check(ncclGroupStart());
          check(cudaSetDevice(source));
          const auto sent = ncclSend(input, elements, ncclFloat16, destination,
                                     ranks[source].comm, ranks[source].communication);
          check(cudaSetDevice(destination));
          const auto got = ncclRecv(received, elements, ncclFloat16, source, dst.comm, dst.communication);
          const auto ended = ncclGroupEnd();
          check(sent); check(got); check(ended);
        }
      }
    }
    for (int d = 0; d < p.world_size; ++d) {
      auto& r = ranks[d];
      check(cudaSetDevice(d));
      check(cudaEventRecord(r.ready, r.communication));
      check(cudaStreamWaitEvent(r.compute, r.ready, 0));
      const auto weight = external[work.weights[d]];
      if (provider == MergedProvider::kCuBLASLt) {
        const float alpha = 1.f, beta = 0.f;
        check(cublasLtMatmul(r.lt, r.operation, &alpha, r.received, r.a, weight, r.b,
                            &beta, r.output, r.c, r.output, r.c, &r.heuristic.algo,
                            r.workspace, workspace_bytes, r.compute));
      } else {
#if defined(EDMIR_HAS_CUTLASS)
        launch_cutlass_gemm_f16_sm80(GemmKernel::kSm80_64x128x32, p.world_size*p.rows_per_peer,
                                    p.n, p.k, r.received, weight, r.output, r.compute);
#else
        throw std::logic_error("unavailable CUTLASS path");
#endif
      }
    }
  }

  void synchronize() {
    require_ready();
    for (std::size_t d = 0; d < ranks.size(); ++d) {
      check(cudaSetDevice(static_cast<int>(d)));
      check(cudaStreamSynchronize(ranks[d].compute));
      check(cudaStreamSynchronize(ranks[d].communication));
    }
  }
};

MergedHostBackend::MergedHostBackend(ProgramView p, MergedProvider v) : impl_(std::make_unique<Impl>(p, v)) {}
MergedHostBackend::~MergedHostBackend() = default;
void MergedHostBackend::prepare() { impl_->prepare(); }
void MergedHostBackend::execute() { impl_->execute(); }
void MergedHostBackend::synchronize() { impl_->synchronize(); }
void MergedHostBackend::copy_from_host(std::size_t i, const void* data, std::size_t bytes) {
  impl_->require_ready();
  if (i >= impl_->external.size()) throw std::invalid_argument("invalid merged upload buffer");
  const auto& b = impl_->program.buffers[i];
  if (!impl_->external[i] || bytes != b.size_bytes)
    throw std::invalid_argument("merged upload requires a complete external matrix");
  check(cudaSetDevice(b.rank));
  check(cudaMemcpy(impl_->external[i], data, bytes, cudaMemcpyHostToDevice));
  // Match CudaBackend's completed-upload contract, outside steady timing.
  check(cudaStreamSynchronize(nullptr));
}
void MergedHostBackend::copy_to_host(std::size_t i, void* data, std::size_t bytes) const {
  impl_->require_ready();
  const auto& p = impl_->program.problem;
  if (i >= impl_->program.buffers.size() || bytes != 2ULL*p.rows_per_peer*p.n)
    throw std::invalid_argument("invalid merged output read");
  for (int s = 0; s < p.world_size; ++s) for (int d = 0; d < p.world_size; ++d) {
    if (impl_->work.outputs[s*p.world_size+d] != i) continue;
    check(cudaSetDevice(d));
    auto* pointer = static_cast<unsigned char*>(impl_->ranks[d].output)+
        2*merged_row_offset(s, p.rows_per_peer, p.world_size)*p.n;
    check(cudaMemcpy(data, pointer, bytes, cudaMemcpyDeviceToHost));
    return;
  }
  throw std::invalid_argument("buffer is not a uniform output");
}
std::string MergedHostBackend::algorithm_id() const {
  impl_->require_ready();
  if (impl_->provider == MergedProvider::kCUTLASS) return "sm80_64x128x32";
  std::string result;
  for (std::size_t d = 0; d < impl_->ranks.size(); ++d)
    result += "rank"+std::to_string(d)+":"+impl_->ranks[d].algorithm+"|";
  return result;
}
}  // namespace edmir::runtime
