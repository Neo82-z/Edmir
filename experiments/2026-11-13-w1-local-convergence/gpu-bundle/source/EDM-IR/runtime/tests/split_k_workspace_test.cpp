#include "edmir/runtime/split_k_workspace.hpp"
#include <cassert>
#include <climits>

using namespace edmir::runtime;
int main() {
  auto kernel = GemmKernel::kSm80_64x128x32SplitK8;
  for (int g : {1, 2}) for (int m : {1, 64, 256, 512, 513}) {
    auto old = split_k_workspace(kernel, {g}, m, 512);
    auto fp32 = split_k_workspace(kernel, {g, SplitKReduction::kSerialFP32}, m, 512);
    assert(old.total_bytes == std::size_t((m+63)/64)*(512/(128*g))*sizeof(int));
    assert(fp32.semaphore_bytes == old.total_bytes);
    assert(fp32.partial_offset >= fp32.semaphore_bytes);
    assert(fp32.partial_offset % 256 == 0);
    assert(fp32.partial_bytes == std::size_t(m)*512*sizeof(float));
    assert(fp32.total_bytes == fp32.partial_offset+fp32.partial_bytes);
  }
  assert(split_k_workspace(GemmKernel::kSm80_64x128x32, {}, 64, 512).total_bytes == 0);
  bool rejected=false;
  try { split_k_workspace(kernel, {0}, 64, 512); } catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  rejected=false;
  try { split_k_workspace(GemmKernel::kSm80_64x128x32, {1,SplitKReduction::kSerialFP32}, 64, 512); }
  catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  // Exercise large-size arithmetic without allocating memory.
  const auto big=split_k_workspace(kernel,{1,SplitKReduction::kSerialFP32},INT_MAX,512);
  assert(big.partial_bytes == std::size_t(INT_MAX)*512*sizeof(float));
  rejected=false;
  try { split_k_workspace(kernel,{1,SplitKReduction::kSerialFP32},INT_MAX,INT_MAX); }
  catch (const std::overflow_error&) { rejected=true; }
  assert(rejected);
}
