#include "edmir/runtime/cuda_backend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "edmir/runtime/validate.hpp"

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
  if (op.kind == OpKind::kTransfer && op.peer_rank != op.rank) {
    return {op.peer_rank, op.rank};
  }
  return {op.rank};
}

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
        EDMIR_CUBLAS(cublasCreate(&ranks_[rank].blas));
        EDMIR_CUBLAS(cublasSetStream(ranks_[rank].blas, ranks_[rank].compute));
      }

      buffers_.resize(program_.buffers.size(), nullptr);
      for (std::size_t index = 0; index < program_.buffers.size(); ++index) {
        const auto& buffer = program_.buffers[index];
        if (buffer.size_bytes == 0) {
          continue;
        }
        EDMIR_CUDA(cudaSetDevice(device_id(buffer.rank)));
        EDMIR_CUDA(cudaMalloc(&buffers_[index], buffer.size_bytes));
      }

      events_.assign(program_.ops.size(), std::vector<cudaEvent_t>(world_size, nullptr));
      for (std::size_t op = 0; op < program_.ops.size(); ++op) {
        for (const auto rank : op_devices(program_.ops[op])) {
          EDMIR_CUDA(cudaSetDevice(device_id(rank)));
          EDMIR_CUDA(cudaEventCreateWithFlags(&events_[op][rank], cudaEventDisableTiming));
        }
      }
      order_ = topological_order(program_);
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
    for (const auto op_index : order_) {
      wait_for_dependencies(op_index);
      const auto& op = program_.ops[op_index];
      switch (op.kind) {
        case OpKind::kTransfer:
          launch_transfer(op_index);
          break;
        case OpKind::kGemm:
          launch_gemm(op_index);
          break;
        case OpKind::kBarrier:
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

  void launch_transfer(std::size_t op_index) {
    const auto& op = program_.ops[op_index];
    const auto reads = accesses(op_index, false);
    const auto writes = accesses(op_index, true);
    if (reads.size() != 1 || writes.size() != 1) {
      throw std::logic_error("transfer access arity changed after validation");
    }
    const auto source = reads[0];
    const auto destination = writes[0];
    const auto size_bytes = static_cast<std::size_t>(op.rows) *
                            static_cast<std::size_t>(program_.problem.k) *
                            dtype_size(program_.problem.dtype);
    if (program_.buffers[source].size_bytes < size_bytes) {
      throw std::logic_error("transfer source buffer is too small");
    }
    if (program_.buffers[destination].size_bytes < size_bytes) {
      throw std::logic_error("transfer destination buffer is too small");
    }
    const auto count = size_bytes / dtype_size(program_.problem.dtype);

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
    const auto reads = accesses(op_index, false);
    const auto writes = accesses(op_index, true);
    const auto activation = find_access_by_role(reads, BufferRole::kStaging);
    const auto weight = find_access_by_role(reads, BufferRole::kWeight);
    const auto output = find_access_by_role(writes, BufferRole::kOutput);
    const auto m = op.rows;
    const auto n = program_.problem.n;
    const auto k = program_.problem.k;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto type = cuda_dtype(program_.problem.dtype);

    EDMIR_CUDA(cudaSetDevice(device_id(op.rank)));
    // Row-major C=A*B is column-major C^T=B^T*A^T with dimensions n,m,k.
    EDMIR_CUBLAS(cublasGemmEx(
        ranks_[op.rank].blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        buffers_[weight], type, n, buffers_[activation], type, k, &beta,
        buffers_[output], type, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  void record_completion(std::size_t op_index) {
    const auto& op = program_.ops[op_index];
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
    for (std::size_t index = 0; index < buffers_.size(); ++index) {
      if (buffers_[index] != nullptr) {
        cudaSetDevice(device_id(program_.buffers[index].rank));
        cudaFree(buffers_[index]);
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
  std::vector<std::vector<cudaEvent_t>> events_;
  std::vector<std::size_t> order_;
  std::vector<std::vector<std::size_t>> incoming_;
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
