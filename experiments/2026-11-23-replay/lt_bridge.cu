// Small C ABI around the existing CUTLASS path and the cuBLASLt heuristics.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstring>
#include <new>
#include "edmir/runtime/cutlass_gemm.hpp"

namespace {
constexpr std::size_t budget = 32ULL * 1024 * 1024;
struct Plan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulPreference_t preference{};
  std::array<cublasLtMatmulHeuristicResult_t,8> algorithms{};
  int count{};
  ~Plan() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
  }
};
#define LT(expr) do { const auto status = (expr); if (status != CUBLAS_STATUS_SUCCESS) return status; } while (0)

int setup(Plan& p, int m, int n, int k) {
  LT(cublasLtCreate(&p.handle));
  LT(cublasLtMatmulDescCreate(&p.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  const cublasOperation_t transpose = CUBLAS_OP_T;
  LT(cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_TRANSB,&transpose,sizeof(transpose)));
  LT(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_16F,m,k,k));
  LT(cublasLtMatrixLayoutCreate(&p.b,CUDA_R_16F,n,k,k));
  LT(cublasLtMatrixLayoutCreate(&p.c,CUDA_R_16F,m,n,n));
  const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
  for (auto layout : {p.a,p.b,p.c})
    LT(cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)));
  LT(cublasLtMatmulPreferenceCreate(&p.preference));
  LT(cublasLtMatmulPreferenceSetAttribute(p.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));
  LT(cublasLtMatmulAlgoGetHeuristic(p.handle,p.operation,p.a,p.b,p.c,p.c,p.preference,
                                  p.algorithms.size(),p.algorithms.data(),&p.count));
  return 0;
}
}
extern "C" int edmir_lt_create(int m,int n,int k,void** out) {
  auto* plan = new(std::nothrow) Plan;
  if (!plan) return CUBLAS_STATUS_ALLOC_FAILED;
  const int status = setup(*plan,m,n,k);
  if (status) { delete plan; return status; }
  *out = plan;
  return 0;
}
extern "C" void edmir_lt_destroy(Plan* plan) { delete plan; }
extern "C" int edmir_lt_count(Plan* p) { return p->count; }
extern "C" int edmir_lt_algo(Plan* p,int i,void* out,std::size_t size) {
  if (i<0 || i>=p->count || size!=sizeof(p->algorithms[i].algo) ||
      p->algorithms[i].state!=CUBLAS_STATUS_SUCCESS || p->algorithms[i].workspaceSize>budget) return -1;
  const auto* source = reinterpret_cast<const unsigned char*>(&p->algorithms[i].algo);
  auto* dest = static_cast<unsigned char*>(out);
  for (std::size_t j=0;j<size;++j) dest[j]=source[j];
  return 0;
}
extern "C" int edmir_lt_algo_size() { return sizeof(cublasLtMatmulAlgo_t); }
extern "C" int edmir_lt_run(Plan* p,const void* a,const void* b,void* c,void* work,
                             const void* algorithm,cudaStream_t stream) {
  const float alpha=1.f,beta=0.f;
  cublasLtMatmulAlgo_t selected{};
  std::memcpy(&selected,algorithm,sizeof(selected));
  return cublasLtMatmul(
      p->handle,p->operation,&alpha,a,p->a,b,p->b,&beta,c,p->c,c,p->c,
      &selected,work,budget,stream);
}
extern "C" int edmir_bridge_cutlass(int m,int n,int k,const void* a,const void* b,void* c,cudaStream_t stream) {
  try {
    using namespace edmir::runtime;
    launch_cutlass_gemm_f16_sm80(GemmKernel::kSm80_64x128x32,m,n,k,a,b,c,stream);
    return 0;
  } catch (...) {
    return -1;
  }
}
