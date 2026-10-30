#include "edmir/runtime/cuda_backend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>

#include "edmir/runtime/validate.hpp"
#include "edmir/runtime/prepared_execution.hpp"
#if defined(EDMIR_HAS_CUTLASS)
#include "edmir/runtime/cutlass_gemm.hpp"
#endif
#if defined(EDMIR_HAS_NCCL_DEVICE)
#include "edmir/runtime/nccl_device_fused.hpp"
#endif

namespace edmir::runtime {
namespace {

void check_cuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
  }
}

void check_cublas(cublasStatus_t result, const char* expression) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expression) + " failed with status " +
                             std::to_string(static_cast<int>(result)));
  }
}

void check_nccl(ncclResult_t result, const char* expression) {
  if (result != ncclSuccess) {
    throw std::runtime_error(std::string(expression) + ": " + ncclGetErrorString(result));
  }
}

#define EDMIR_CUDA(expr) check_cuda((expr), #expr)
#define EDMIR_CUBLAS(expr) check_cublas((expr), #expr)
#define EDMIR_NCCL(expr) check_nccl((expr), #expr)

cudaDataType_t cuda_dtype(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat32:
      return CUDA_R_32F;
  }
  throw std::invalid_argument("unsupported CUDA dtype");
}

ncclDataType_t nccl_dtype(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
      return ncclFloat16;
    case DType::kBFloat16:
      return ncclBfloat16;
    case DType::kFloat32:
      return ncclFloat32;
  }
  throw std::invalid_argument("unsupported NCCL dtype");
}

std::vector<std::int32_t> op_devices(const OpDesc& op) {
  if ((op.kind == OpKind::kTransfer || op.kind == OpKind::kFusedTransferGemm) &&
      op.peer_rank != op.rank) {
    return {op.peer_rank, op.rank};
  }
  return {op.rank};
}

class NvtxRange {
 public:
  NvtxRange(
      bool enabled, std::string_view prefix, std::string_view name, std::size_t iteration)
      : enabled_(enabled) {
    if (!enabled_) return;
    std::string text;
    text.reserve(prefix.size() + name.size() + 24);
    text.append(prefix).append(name).append(".iter").append(std::to_string(iteration));
    nvtxRangePushA(text.c_str());
  }
  ~NvtxRange() {
    if (enabled_) nvtxRangePop();
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  bool enabled_;
};

}  // namespace

class CudaBackend::Impl {
 public:
  explicit Impl(ProgramView program, CudaBackendOptions options)
      : program_(program), options_(std::move(options)) {}

  ~Impl() { release(); }

  void prepare() {
    if (prepared_) {
      return;
    }
    const auto report = validate_program(program_);
    if (!report.valid()) {
      std::string message = "invalid runtime ProgramView";
      for (const auto& error : report.errors) {
        message += "; " + error;
      }
      throw std::invalid_argument(message);
    }
#if !defined(EDMIR_HAS_CUTLASS)
    if (program_.gemm_provider == GemmProvider::kCUTLASS) {
      throw std::invalid_argument(
          "generated Program requests CUTLASS, but the runtime was built without "
          "EDMIR_ENABLE_CUTLASS=ON");
    }
#endif
#if !defined(EDMIR_HAS_NCCL_DEVICE)
    if (std::any_of(
            program_.ops.begin(), program_.ops.end(),
            [](const OpDesc& op) { return op.kind == OpKind::kFusedTransferGemm; })) {
      throw std::invalid_argument(
          "generated Program requests a fused kernel, but the runtime was "
          "built without EDMIR_ENABLE_NCCL_DEVICE=ON");
    }
#endif

    if (options_.prepare_metadata) prepared_execution_ = prepare_execution(program_);
    const auto world_size = static_cast<std::size_t>(program_.problem.world_size);
    configure_devices(world_size);
    ranks_.resize(world_size);
    try {
      std::vector<ncclComm_t> communicators(world_size);
      EDMIR_NCCL(ncclCommInitAll(
          communicators.data(), static_cast<int>(world_size), options_.devices.data()));
      for (std::size_t rank = 0; rank < world_size; ++rank) {
        ranks_[rank].communicator = communicators[rank];
      }

      for (std::size_t rank = 0; rank < world_size; ++rank) {
        EDMIR_CUDA(cudaSetDevice(device_id(rank)));
        EDMIR_CUDA(cudaStreamCreateWithFlags(&ranks_[rank].comm, cudaStreamNonBlocking));
        EDMIR_CUDA(cudaStreamCreateWithFlags(&ranks_[rank].compute, cudaStreamNonBlocking));
        EDMIR_CUDA(cudaStreamCreateWithFlags(&ranks_[rank].control, cudaStreamNonBlocking));
        if (program_.gemm_provider == GemmProvider::kCuBLAS) {
          EDMIR_CUBLAS(cublasCreate(&ranks_[rank].blas));
          EDMIR_CUBLAS(cublasSetStream(ranks_[rank].blas, ranks_[rank].compute));
        }
      }

      buffers_.resize(program_.buffers.size(), nullptr);
      nccl_allocated_.assign(program_.buffers.size(), false);
      const auto has_fused = std::any_of(
          program_.ops.begin(), program_.ops.end(),
          [](const OpDesc& op) { return op.kind == OpKind::kFusedTransferGemm; });
      for (std::size_t index = 0; index < program_.buffers.size(); ++index) {
        const auto& buffer = program_.buffers[index];
        if (buffer.size_bytes == 0) {
          continue;
        }
        EDMIR_CUDA(cudaSetDevice(device_id(buffer.rank)));
        if (has_fused && buffer.role == BufferRole::kInput) {
#if defined(EDMIR_HAS_NCCL_DEVICE)
          EDMIR_NCCL(ncclMemAlloc(&buffers_[index], buffer.size_bytes));
          nccl_allocated_[index] = true;
#else
          throw std::logic_error("fused input allocation reached an unavailable runtime path");
#endif
        } else {
          EDMIR_CUDA(cudaMalloc(&buffers_[index], buffer.size_bytes));
        }
      }

#if defined(EDMIR_HAS_NCCL_DEVICE)
      if (has_fused) prepare_fused_windows();
#if defined(EDMIR_HAS_NCCL_DEVICE_CUTLASS)
      if (has_fused) prepare_fused_cutlass_workspaces();
#endif
#endif

      events_.assign(program_.ops.size(), std::vector<cudaEvent_t>(world_size, nullptr));
      for (std::size_t op = 0; op < program_.ops.size(); ++op) {
        for (const auto rank : op_devices(program_.ops[op])) {
          EDMIR_CUDA(cudaSetDevice(device_id(rank)));
          EDMIR_CUDA(cudaEventCreateWithFlags(&events_[op][rank], cudaEventDisableTiming));
        }
      }
      order_ = options_.prepare_metadata ? prepared_execution_.order : topological_order(program_);
      incoming_.resize(program_.ops.size());
      for (const auto& dependency : program_.dependencies) {
        auto& predecessors = incoming_[dependency.dst_op];
        if (std::find(predecessors.begin(), predecessors.end(), dependency.src_op) ==
            predecessors.end()) {
          predecessors.push_back(dependency.src_op);
        }
      }
      prepared_ = true;
    } catch (...) {
      release();
      throw;
    }
  }

  void execute() {
    require_prepared();
    const auto iteration = execution_count_++;
    const NvtxRange execution_range(
        options_.enable_nvtx, "edm.phase:", "runtime.execute", iteration);
    for (const auto op_index : order_) {
      const auto& op = program_.ops[op_index];
      const NvtxRange op_range(
          options_.enable_nvtx, "edm.op:", op.id, iteration);
      wait_for_dependencies(op_index);
      switch (op.kind) {
        case OpKind::kTransfer:
          launch_transfer(op_index);
          break;
        case OpKind::kGemm:
          launch_gemm(op_index);
          break;
        case OpKind::kBarrier:
          break;
        case OpKind::kFusedTransferGemm:
          launch_fused_transfer_gemm(op_index);
          break;
      }
      record_completion(op_index);
    }
  }

  void synchronize() {
    require_prepared();
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      EDMIR_CUDA(cudaSetDevice(device_id(rank)));
      EDMIR_CUDA(cudaStreamSynchronize(ranks_[rank].comm));
      EDMIR_CUDA(cudaStreamSynchronize(ranks_[rank].compute));
      EDMIR_CUDA(cudaStreamSynchronize(ranks_[rank].control));
    }
  }

  std::size_t find_buffer(std::string_view name) const {
    for (std::size_t index = 0; index < program_.buffers.size(); ++index) {
      if (program_.buffers[index].name == name) {
        return index;
      }
    }
    throw std::out_of_range("unknown runtime buffer " + std::string(name));
  }

  void* buffer_data(std::size_t buffer) {
    require_buffer(buffer);
    return buffers_[buffer];
  }

  const void* buffer_data(std::size_t buffer) const {
    require_buffer(buffer);
    return buffers_[buffer];
  }

  void copy_from_host(std::size_t buffer, const void* source, std::size_t size_bytes) {
    require_prepared();
    require_copy(buffer, size_bytes);
    EDMIR_CUDA(cudaSetDevice(device_id(program_.buffers[buffer].rank)));
    EDMIR_CUDA(cudaMemcpy(buffers_[buffer], source, size_bytes, cudaMemcpyHostToDevice));
  }

  void copy_to_host(std::size_t buffer, void* destination, std::size_t size_bytes) const {
    require_prepared();
    require_copy(buffer, size_bytes);
    EDMIR_CUDA(cudaSetDevice(device_id(program_.buffers[buffer].rank)));
    EDMIR_CUDA(cudaMemcpy(destination, buffers_[buffer], size_bytes, cudaMemcpyDeviceToHost));
  }

 private:
  struct RankState {
    cudaStream_t comm = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t control = nullptr;
    cublasHandle_t blas = nullptr;
    ncclComm_t communicator = nullptr;
  };

#if defined(EDMIR_HAS_NCCL_DEVICE)
  struct FusedWindowState {
    std::vector<void*> participants;
    std::vector<bool> owns_participant;
    std::vector<ncclWindow_t> windows;
    bool registered = false;
  };
#endif

  void configure_devices(std::size_t world_size) {
    if (options_.devices.empty()) {
      options_.devices.resize(world_size);
      std::iota(options_.devices.begin(), options_.devices.end(), 0);
    }
    if (options_.devices.size() != world_size) {
      throw std::invalid_argument("CUDA device map size must equal Program world_size");
    }
    int available = 0;
    EDMIR_CUDA(cudaGetDeviceCount(&available));
    std::unordered_set<int> unique;
    for (const auto device : options_.devices) {
      if (device < 0 || device >= available) {
        throw std::invalid_argument("CUDA device map contains an unavailable device ordinal");
      }
      if (!unique.emplace(device).second) {
        throw std::invalid_argument("CUDA device map must not contain duplicate ordinals");
      }
    }
  }

  int device_id(std::size_t rank) const { return options_.devices.at(rank); }

  void require_prepared() const {
    if (!prepared_) {
      throw std::logic_error("CUDA backend must be prepared before use");
    }
  }

  void require_buffer(std::size_t buffer) const {
    require_prepared();
    if (buffer >= buffers_.size()) {
      throw std::out_of_range("runtime buffer index is out of range");
    }
  }

  void require_copy(std::size_t buffer, std::size_t size_bytes) const {
    require_buffer(buffer);
    if (size_bytes > program_.buffers[buffer].size_bytes) {
      throw std::out_of_range("host copy exceeds runtime buffer size");
    }
    if (size_bytes > 0 && buffers_[buffer] == nullptr) {
      throw std::logic_error("runtime buffer is not allocated");
    }
  }

  cudaStream_t stream_for(const OpDesc& op, std::int32_t device) {
    auto& rank = ranks_.at(static_cast<std::size_t>(device));
    if (op.kind == OpKind::kFusedTransferGemm && device != op.rank) {
      return rank.comm;
    }
    switch (op.queue) {
      case QueueKind::kComm:
        return rank.comm;
      case QueueKind::kCompute:
        return rank.compute;
      case QueueKind::kControl:
        return rank.control;
    }
    throw std::invalid_argument("unknown runtime queue");
  }

  void wait_for_dependencies(std::size_t current) {
    if (options_.prepare_metadata) {
      for (const auto& wait : prepared_execution_.ops[current].waits) {
        EDMIR_CUDA(cudaSetDevice(device_id(wait.rank)));
        EDMIR_CUDA(cudaStreamWaitEvent(
            stream_for(program_.ops[current], wait.rank), events_[wait.predecessor][wait.event_rank], 0));
      }
      return;
    }
    const auto current_devices = op_devices(program_.ops[current]);
    for (const auto predecessor : incoming_[current]) {
      const auto predecessor_devices = op_devices(program_.ops[predecessor]);
      for (const auto rank : current_devices) {
        const auto shared = std::find(predecessor_devices.begin(), predecessor_devices.end(), rank);
        const auto event_rank = shared == predecessor_devices.end()
                                    ? program_.ops[predecessor].rank
                                    : rank;
        EDMIR_CUDA(cudaSetDevice(device_id(rank)));
        EDMIR_CUDA(cudaStreamWaitEvent(
            stream_for(program_.ops[current], rank), events_[predecessor][event_rank], 0));
      }
    }
  }

  std::vector<std::size_t> accesses(std::size_t op, bool write) const {
    std::vector<std::size_t> result;
    for (const auto& access : program_.accesses) {
      if (access.op == op && access.write == write) {
        result.push_back(access.buffer);
      }
    }
    return result;
  }

  std::size_t find_access_by_role(
      const std::vector<std::size_t>& candidates, BufferRole role) const {
    const auto found = std::find_if(candidates.begin(), candidates.end(), [&](std::size_t index) {
      return program_.buffers[index].role == role;
    });
    if (found == candidates.end()) {
      throw std::logic_error("operation is missing a required buffer role");
    }
    return *found;
  }

#if defined(EDMIR_HAS_NCCL_DEVICE)
  void prepare_fused_windows() {
    const auto world_size = static_cast<std::size_t>(program_.problem.world_size);
    fused_windows_.resize(program_.ops.size());
    for (std::size_t op_index = 0; op_index < program_.ops.size(); ++op_index) {
      const auto& op = program_.ops[op_index];
      if (op.kind != OpKind::kFusedTransferGemm) continue;
      const auto reads = accesses(op_index, false);
      const auto input = find_access_by_role(reads, BufferRole::kInput);
      const auto size_bytes = program_.buffers[input].size_bytes;
      auto& state = fused_windows_[op_index];
      state.participants.assign(world_size, nullptr);
      state.owns_participant.assign(world_size, false);
      state.windows.assign(world_size, nullptr);
      for (std::size_t rank = 0; rank < world_size; ++rank) {
        EDMIR_CUDA(cudaSetDevice(device_id(rank)));
        if (static_cast<std::int32_t>(rank) == op.peer_rank) {
          state.participants[rank] = buffers_[input];
        } else {
          EDMIR_NCCL(ncclMemAlloc(&state.participants[rank], size_bytes));
          state.owns_participant[rank] = true;
        }
      }
      EDMIR_NCCL(ncclGroupStart());
      for (std::size_t rank = 0; rank < world_size; ++rank) {
        EDMIR_CUDA(cudaSetDevice(device_id(rank)));
        EDMIR_NCCL(ncclCommWindowRegister(
            ranks_[rank].communicator, state.participants[rank], size_bytes,
            &state.windows[rank], NCCL_WIN_COLL_SYMMETRIC));
      }
      EDMIR_NCCL(ncclGroupEnd());
      state.registered = true;
    }
  }

  void release_fused_windows() noexcept {
    for (auto& state : fused_windows_) {
      if (state.registered) {
        ncclGroupStart();
        for (std::size_t rank = 0; rank < state.windows.size(); ++rank) {
          if (state.windows[rank] == nullptr || rank >= ranks_.size()) continue;
          cudaSetDevice(device_id(rank));
          ncclCommWindowDeregister(ranks_[rank].communicator, state.windows[rank]);
        }
        ncclGroupEnd();
        state.registered = false;
      }
      for (std::size_t rank = 0; rank < state.participants.size(); ++rank) {
        if (!state.owns_participant[rank] || state.participants[rank] == nullptr) continue;
        cudaSetDevice(device_id(rank));
        ncclMemFree(state.participants[rank]);
        state.participants[rank] = nullptr;
      }
    }
    fused_windows_.clear();
  }

#if defined(EDMIR_HAS_NCCL_DEVICE_CUTLASS)
  void prepare_fused_cutlass_workspaces() {
    fused_cutlass_workspaces_.assign(program_.ops.size(), nullptr);
    if (program_.gemm_provider != GemmProvider::kCUTLASS) return;
    for (std::size_t index = 0; index < program_.ops.size(); ++index) {
      const auto& op = program_.ops[index];
      if (op.kind != OpKind::kFusedTransferGemm) continue;
      const auto bytes = lsa_cutlass_workspace_size(
          program_.gemm_kernel, program_.fused_schedule, op.rows,
          program_.problem.n);
      if (bytes == 0) continue;
      EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
      EDMIR_CUDA(cudaMalloc(&fused_cutlass_workspaces_[index], bytes));
      EDMIR_CUDA(cudaMemset(fused_cutlass_workspaces_[index], 0, bytes));
    }
  }

  void release_fused_cutlass_workspaces() noexcept {
    for (std::size_t index = 0; index < fused_cutlass_workspaces_.size(); ++index) {
      if (fused_cutlass_workspaces_[index] == nullptr) continue;
      cudaSetDevice(device_id(program_.ops[index].rank));
      cudaFree(fused_cutlass_workspaces_[index]);
    }
    fused_cutlass_workspaces_.clear();
  }
#endif
#endif

  void launch_transfer(std::size_t op_index) {
    const auto& op = program_.ops[op_index];
    std::size_t source, destination, size_bytes, count;
    if (options_.prepare_metadata) {
      const auto& bindings = prepared_execution_.ops[op_index].operands;
      source = bindings.input;
      destination = bindings.output;
      size_bytes = bindings.transfer_bytes;
      count = bindings.transfer_elements;
    } else {
      const auto reads = accesses(op_index, false);
      const auto writes = accesses(op_index, true);
      if (reads.size() != 1 || writes.size() != 1)
        throw std::logic_error("transfer access arity changed after validation");
      source = reads[0];
      destination = writes[0];
      size_bytes = static_cast<std::size_t>(op.rows) *
                   static_cast<std::size_t>(program_.problem.k) * dtype_size(program_.problem.dtype);
      if (program_.buffers[source].size_bytes < size_bytes)
        throw std::logic_error("transfer source buffer is too small");
      if (program_.buffers[destination].size_bytes < size_bytes)
        throw std::logic_error("transfer destination buffer is too small");
      count = size_bytes / dtype_size(program_.problem.dtype);
    }

    if (op.peer_rank == op.rank) {
      EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
      EDMIR_CUDA(cudaMemcpyAsync(
          buffers_[destination], buffers_[source], size_bytes, cudaMemcpyDeviceToDevice,
          ranks_[op.rank].comm));
      return;
    }

    EDMIR_NCCL(ncclGroupStart());
    EDMIR_CUDA(cudaSetDevice(device_id(op.peer_rank)));
    EDMIR_NCCL(ncclSend(
        buffers_[source], count, nccl_dtype(program_.problem.dtype), op.rank,
        ranks_[op.peer_rank].communicator, ranks_[op.peer_rank].comm));
    EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
    EDMIR_NCCL(ncclRecv(
        buffers_[destination], count, nccl_dtype(program_.problem.dtype), op.peer_rank,
        ranks_[op.rank].communicator, ranks_[op.rank].comm));
    EDMIR_NCCL(ncclGroupEnd());
  }

  void launch_gemm(std::size_t op_index) {
    const auto& op = program_.ops[op_index];
    std::size_t activation, weight, output;
    if (options_.prepare_metadata) {
      const auto& bindings = prepared_execution_.ops[op_index].operands;
      activation = bindings.input;
      weight = bindings.weight;
      output = bindings.output;
    } else {
      const auto reads = accesses(op_index, false);
      const auto writes = accesses(op_index, true);
      activation = find_access_by_role(reads, BufferRole::kStaging);
      weight = find_access_by_role(reads, BufferRole::kWeight);
      output = find_access_by_role(writes, BufferRole::kOutput);
    }
    const auto m = op.rows;
    const auto n = program_.problem.n;
    const auto k = program_.problem.k;
    EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
    if (program_.gemm_provider == GemmProvider::kCuBLAS) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      const auto type = cuda_dtype(program_.problem.dtype);
      // Row-major C=A*B is column-major C^T=B^T*A^T with dimensions n,m,k.
      EDMIR_CUBLAS(cublasGemmEx(
          ranks_[op.rank].blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
          buffers_[weight], type, n, buffers_[activation], type, k, &beta,
          buffers_[output], type, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      return;
    }
#if defined(EDMIR_HAS_CUTLASS)
    launch_cutlass_gemm_f16_sm80(
        program_.gemm_kernel, m, n, k, buffers_[activation], buffers_[weight], buffers_[output],
        ranks_[op.rank].compute);
#else
    throw std::logic_error("CUTLASS provider reached an unavailable runtime path");
#endif
  }

  void launch_fused_transfer_gemm(std::size_t op_index) {
#if defined(EDMIR_HAS_NCCL_DEVICE)
    const auto& op = program_.ops[op_index];
    std::size_t weight, output;
    if (options_.prepare_metadata) {
      const auto& bindings = prepared_execution_.ops[op_index].operands;
      weight = bindings.weight;
      output = bindings.output;
    } else {
      const auto reads = accesses(op_index, false);
      const auto writes = accesses(op_index, true);
      weight = find_access_by_role(reads, BufferRole::kWeight);
      output = find_access_by_role(writes, BufferRole::kOutput);
    }
    if (op_index >= fused_windows_.size() ||
        fused_windows_[op_index].windows.size() <= static_cast<std::size_t>(op.rank)) {
      throw std::logic_error("fused op is missing its symmetric input window");
    }
    EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
    if (program_.gemm_provider == GemmProvider::kCudaReference) {
      EDMIR_CUDA(launch_lsa_naive_gemm_f16(
          fused_windows_[op_index].windows[op.rank], op.peer_rank, op.rows,
          program_.problem.n, program_.problem.k, buffers_[weight], buffers_[output],
          ranks_[op.rank].compute));
      return;
    }
#if defined(EDMIR_HAS_NCCL_DEVICE_CUTLASS)
    EDMIR_CUDA(launch_lsa_cutlass_gemm_f16_sm80(
        program_.gemm_kernel, program_.fused_schedule,
        fused_windows_[op_index].windows[op.rank], op.peer_rank, op.rows,
        program_.problem.n, program_.problem.k,
        buffers_[weight], buffers_[output], fused_cutlass_workspaces_[op_index],
        ranks_[op.rank].compute));
#else
    throw std::logic_error(
        "fused CUTLASS op requires EDMIR_ENABLE_CUTLASS=ON together with the Device API");
#endif
#else
    static_cast<void>(op_index);
    throw std::logic_error("fused op reached an unavailable NCCL Device API runtime path");
#endif
  }

  void record_completion(std::size_t op_index) {
    const auto& op = program_.ops[op_index];
    if (op.kind == OpKind::kFusedTransferGemm) {
      EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
      EDMIR_CUDA(cudaEventRecord(events_[op_index][op.rank], ranks_[op.rank].compute));
      if (op.peer_rank != op.rank) {
        EDMIR_CUDA(cudaSetDevice(device_id(op.peer_rank)));
        EDMIR_CUDA(cudaStreamWaitEvent(
            ranks_[op.peer_rank].comm, events_[op_index][op.rank], 0));
        EDMIR_CUDA(cudaEventRecord(
            events_[op_index][op.peer_rank], ranks_[op.peer_rank].comm));
      }
      return;
    }
    if (options_.prepare_metadata) {
      for (const auto rank : prepared_execution_.ops[op_index].devices) {
        EDMIR_CUDA(cudaSetDevice(device_id(rank)));
        EDMIR_CUDA(cudaEventRecord(events_[op_index][rank], stream_for(op, rank)));
      }
      return;
    }
    for (const auto rank : op_devices(op)) {
      EDMIR_CUDA(cudaSetDevice(device_id(rank)));
      EDMIR_CUDA(cudaEventRecord(events_[op_index][rank], stream_for(op, rank)));
    }
  }

  void release() noexcept {
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      if (rank < options_.devices.size()) {
        cudaSetDevice(device_id(rank));
        cudaDeviceSynchronize();
      }
    }
    for (std::size_t op = 0; op < events_.size(); ++op) {
      for (std::size_t rank = 0; rank < events_[op].size(); ++rank) {
        if (events_[op][rank] != nullptr) {
          cudaSetDevice(device_id(rank));
          cudaEventDestroy(events_[op][rank]);
        }
      }
    }
#if defined(EDMIR_HAS_NCCL_DEVICE)
    release_fused_windows();
#endif
#if defined(EDMIR_HAS_NCCL_DEVICE_CUTLASS)
    release_fused_cutlass_workspaces();
#endif
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
      if (buffers_[index] != nullptr) {
        cudaSetDevice(device_id(program_.buffers[index].rank));
        if (index < nccl_allocated_.size() && nccl_allocated_[index]) {
          ncclMemFree(buffers_[index]);
        } else {
          cudaFree(buffers_[index]);
        }
      }
    }
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
      cudaSetDevice(device_id(rank));
      if (ranks_[rank].blas != nullptr) cublasDestroy(ranks_[rank].blas);
      if (ranks_[rank].comm != nullptr) cudaStreamDestroy(ranks_[rank].comm);
      if (ranks_[rank].compute != nullptr) cudaStreamDestroy(ranks_[rank].compute);
      if (ranks_[rank].control != nullptr) cudaStreamDestroy(ranks_[rank].control);
      if (ranks_[rank].communicator != nullptr) ncclCommDestroy(ranks_[rank].communicator);
    }
    prepared_ = false;
  }

  ProgramView program_;
  CudaBackendOptions options_;
  bool prepared_ = false;
  std::vector<RankState> ranks_;
  std::vector<void*> buffers_;
  std::vector<bool> nccl_allocated_;
#if defined(EDMIR_HAS_NCCL_DEVICE)
  std::vector<FusedWindowState> fused_windows_;
#endif
#if defined(EDMIR_HAS_NCCL_DEVICE_CUTLASS)
  std::vector<void*> fused_cutlass_workspaces_;
#endif
  std::vector<std::vector<cudaEvent_t>> events_;
  std::vector<std::size_t> order_;
  std::vector<std::vector<std::size_t>> incoming_;
  PreparedExecution prepared_execution_;
  std::size_t execution_count_ = 0;
};

CudaBackend::CudaBackend(ProgramView program, CudaBackendOptions options)
    : impl_(std::make_unique<Impl>(program, std::move(options))) {}
CudaBackend::~CudaBackend() = default;
CudaBackend::CudaBackend(CudaBackend&&) noexcept = default;
CudaBackend& CudaBackend::operator=(CudaBackend&&) noexcept = default;

void CudaBackend::prepare() { impl_->prepare(); }
void CudaBackend::execute() { impl_->execute(); }
void CudaBackend::synchronize() { impl_->synchronize(); }
std::size_t CudaBackend::find_buffer(std::string_view name) const {
  return impl_->find_buffer(name);
}
void* CudaBackend::buffer_data(std::size_t buffer) { return impl_->buffer_data(buffer); }
const void* CudaBackend::buffer_data(std::size_t buffer) const {
  return impl_->buffer_data(buffer);
}
void CudaBackend::copy_from_host(
    std::size_t buffer, const void* source, std::size_t size_bytes) {
  impl_->copy_from_host(buffer, source, size_bytes);
}
void CudaBackend::copy_to_host(
    std::size_t buffer, void* destination, std::size_t size_bytes) const {
  impl_->copy_to_host(buffer, destination, size_bytes);
}

}  // namespace edmir::runtime
