#include "cuda_api_trace.hpp"

#include <cstdint>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>

#include "edmir/runtime/cutlass_gemm.hpp"
#include "edmir/runtime/nccl_device_fused.hpp"

namespace {
std::uintptr_t next_handle = 4096;
int active_device = 0;
std::vector<std::string> calls;
template<class T> T handle() {
  next_handle += 256;
  return reinterpret_cast<T>(next_handle);
}
template<class... Args> void log_call(const char* name, const Args&... args) {
  std::ostringstream stream;
  stream << name << " device=" << active_device;
  ((stream << ' ' << args), ...);
  calls.push_back(stream.str());
}
}

namespace edmir::runtime::test {
void reset_api_trace() { next_handle = 4096; active_device = 0; calls.clear(); }
void clear_api_trace() { calls.clear(); }
std::vector<std::string> api_trace() { return calls; }
}

extern "C" {
const char* cudaGetErrorString(cudaError_t) { return "CPU API trace stub"; }
cudaError_t cudaGetDeviceCount(int* count) { *count = 8; return cudaSuccess; }
cudaError_t cudaSetDevice(int device) { active_device = device; log_call("set_device", device); return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags) {
  *stream = handle<cudaStream_t>(); log_call("create_stream", *stream, flags); return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
  *event = handle<cudaEvent_t>(); log_call("create_event", *event, flags); return cudaSuccess;
}
cudaError_t cudaMalloc(void** ptr, size_t bytes) {
  *ptr = handle<void*>(); log_call("malloc", *ptr, bytes); return cudaSuccess;
}
cudaError_t cudaFree(void* ptr) { log_call("free", ptr); return cudaSuccess; }
cudaError_t cudaMemset(void* ptr, int value, size_t bytes) {
  log_call("memset", ptr, value, bytes); return cudaSuccess;
}
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  log_call("copy", dst, src, bytes, kind); return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  log_call("copy_async", dst, src, bytes, kind, stream); return cudaSuccess;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
  log_call("wait_event", stream, event, flags); return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  log_call("record_event", event, stream); return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { log_call("sync_stream", stream); return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { log_call("sync_device"); return cudaSuccess; }
cudaError_t cudaEventDestroy(cudaEvent_t event) { log_call("destroy_event", event); return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t stream) { log_call("destroy_stream", stream); return cudaSuccess; }

cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = handle<cublasHandle_t>(); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy(cublasHandle_t h) { log_call("destroy_blas", h); return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t stream) {
  log_call("blas_stream", h, stream); return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                          int m, int n, int k, const void* alpha,
                          const void* a, cudaDataType at, int lda,
                          const void* b, cudaDataType bt, int ldb,
                          const void* beta, void* c, cudaDataType ct, int ldc,
                          cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
  log_call("blas_gemm", h, ta, tb, m, n, k, *static_cast<const float*>(alpha),
           a, at, lda, b, bt, ldb, *static_cast<const float*>(beta), c, ct, ldc, compute, algorithm);
  return CUBLAS_STATUS_SUCCESS;
}

const char* ncclGetErrorString(ncclResult_t) { return "CPU API trace stub"; }
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int count, const int* devices) {
  for (int i = 0; i < count; ++i) {
    comm[i] = handle<ncclComm_t>(); log_call("comm_init", comm[i], devices[i]);
  }
  return ncclSuccess;
}
ncclResult_t ncclCommDestroy(ncclComm_t comm) { log_call("comm_destroy", comm); return ncclSuccess; }
ncclResult_t ncclMemAlloc(void** ptr, size_t bytes) {
  *ptr = handle<void*>(); log_call("nccl_malloc", *ptr, bytes); return ncclSuccess;
}
ncclResult_t ncclMemFree(void* ptr) { log_call("nccl_free", ptr); return ncclSuccess; }
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* ptr, size_t bytes, ncclWindow_t* win, int flags) {
  *win = handle<ncclWindow_t>(); log_call("register_window", comm, ptr, bytes, *win, flags); return ncclSuccess;
}
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  log_call("deregister_window", comm, win); return ncclSuccess;
}
ncclResult_t ncclGroupStart() { log_call("group_start"); return ncclSuccess; }
ncclResult_t ncclGroupEnd() { log_call("group_end"); return ncclSuccess; }
ncclResult_t ncclSend(const void* ptr, size_t count, ncclDataType_t type, int peer, ncclComm_t comm, cudaStream_t stream) {
  log_call("send", ptr, count, type, peer, comm, stream); return ncclSuccess;
}
ncclResult_t ncclRecv(void* ptr, size_t count, ncclDataType_t type, int peer, ncclComm_t comm, cudaStream_t stream) {
  log_call("recv", ptr, count, type, peer, comm, stream); return ncclSuccess;
}
}

namespace edmir::runtime {
void launch_cutlass_gemm_f16_sm80(GemmKernel kernel, int m, int n, int k,
                                const void* a, const void* b, void* c, cudaStream_t stream) {
  log_call("cutlass_gemm", static_cast<int>(kernel), m, n, k, a, b, c, stream);
}
cudaError_t launch_lsa_naive_gemm_f16(ncclWindow_t window, int source, int m, int n, int k,
                                    const void* b, void* c, cudaStream_t stream) {
  log_call("fused_reference", window, source, m, n, k, b, c, stream); return cudaSuccess;
}
cudaError_t launch_lsa_cutlass_gemm_f16_sm80(GemmKernel kernel, FusedScheduleDesc schedule,
    ncclWindow_t window, int source, int m, int n, int k,
    const void* b, void* c, void* workspace, cudaStream_t stream) {
  log_call("fused_cutlass", static_cast<int>(kernel), schedule.n_tiles_per_cta,
           window, source, m, n, k, b, c, workspace, stream);
  return cudaSuccess;
}
std::size_t lsa_cutlass_workspace_size(GemmKernel kernel, FusedScheduleDesc, int, int) {
  // Exercise both workspace branches; this is not the actual kernel allocation size.
  return gemm_kernel_split_k_slices(kernel) > 1 ? 128 : 0;
}
}
