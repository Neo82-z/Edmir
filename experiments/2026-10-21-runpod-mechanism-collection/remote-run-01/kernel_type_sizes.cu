// Host-only type introspection of the pinned CUTLASS definitions.
// These are compiled resource requirements, not measured occupancy or traffic.
#include <iostream>
#include "cutlass_sm80_gemm.cuh"

template <int N, bool Split>
void record(const char* label, bool comma) {
  using Gemm = edmir::runtime::detail::Sm80Gemm<64,N,32,64,Split>;
  using Kernel = typename Gemm::GemmKernel;
  if (comma) std::cout << ',';
  std::cout << "{\"name\":\"" << label << "\",\"threads_per_cta\":" << Kernel::kThreadCount
            << ",\"shared_storage_bytes\":" << sizeof(typename Kernel::SharedStorage)
            << ",\"shared_storage_alignment\":" << alignof(typename Kernel::SharedStorage)
            << ",\"tile_m\":64,\"tile_n\":" << N << ",\"tile_k\":32,\"stages\":3}";
}
int main() {
  std::cout << "{\"scope\":\"host-only type sizes; not achieved occupancy\",\"types\":[";
  record<128,false>("g1_direct",false);
  record<128,true>("g1_serial_splitk",true);
  record<256,false>("g2_direct",true);
  record<256,true>("g2_serial_splitk",true);
  std::cout << "]}\n";
}
