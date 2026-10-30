#include <iostream>
#include <stdexcept>
#include "cutlass_sm80_gemm.cuh"

// Host-only inspection of the same template types as the actual launcher.
// No device calls, kernel launches, timing claims or register-count estimates.
template<int Group, bool Split>
void check(int m, int k, int n, int q) {
  using Gemm = edmir::runtime::detail::Sm80Gemm<64, 128*Group, 32, 64, Split>;
  using Kernel = typename Gemm::GemmKernel;
  using Swizzle = typename Gemm::ThreadblockSwizzle;
  const cutlass::gemm::GemmCoord problem(m, n, k);
  const auto tiled = Swizzle::get_tiled_shape(problem, {64, 128*Group, 32}, q);
  const auto grid = Swizzle::get_grid_shape(tiled);
  typename Kernel::Params params(
      problem, tiled, {nullptr, k}, {nullptr, n}, {nullptr, n}, {nullptr, n});
  const int expected_ctas = ((m+63)/64)*((n+128*Group-1)/(128*Group))*q;
  if (static_cast<int>(grid.x*grid.y*grid.z) != expected_ctas)
    throw std::runtime_error("analytical CTA count disagrees with CUTLASS");
  if (params.gemm_k_size != k/q || Kernel::kSplitKSerial != Split)
    throw std::runtime_error("K partition or serial reduction contract differs");
  std::cout << "{\"m\":" << m << ",\"k\":" << k << ",\"n\":" << n
            << ",\"q\":" << q << ",\"g\":" << Group << ",\"ctas\":" << expected_ctas
            << ",\"partition_k\":" << params.gemm_k_size
            << ",\"threads_per_cta\":" << Kernel::kThreadCount
            << ",\"shared_storage_bytes\":" << sizeof(typename Kernel::SharedStorage)
            << ",\"serial_split_k\":" << (Split ? "true" : "false") << "}\n";
}

int main() {
  try {
    for (auto shape : {cutlass::gemm::GemmCoord{64, 512, 1024},
                       cutlass::gemm::GemmCoord{64, 2048, 4096},
                       cutlass::gemm::GemmCoord{64, 512, 8192},
                       cutlass::gemm::GemmCoord{256, 512, 8192}}) {
      for (int q : {1, 2, 4, 8, 16}) {
        if (q == 1) {
          check<1, false>(shape.m(), shape.k(), shape.n(), q);
          check<2, false>(shape.m(), shape.k(), shape.n(), q);
        } else {
          check<1, true>(shape.m(), shape.k(), shape.n(), q);
          check<2, true>(shape.m(), shape.k(), shape.n(), q);
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
