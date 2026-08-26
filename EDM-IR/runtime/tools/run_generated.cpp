#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "edmir/runtime/cuda_backend.hpp"
#include "edmir_program.hpp"

namespace {

bool is_external(edmir::runtime::BufferRole role) {
  return role == edmir::runtime::BufferRole::kInput ||
         role == edmir::runtime::BufferRole::kWeight;
}

}  // namespace

int main() {
  try {
    const auto program = edmir_generated::program();
    edmir::runtime::CudaBackend backend(program);
    backend.prepare();

    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      const auto& buffer = program.buffers[index];
      if (!is_external(buffer.role) || buffer.size_bytes == 0) {
        continue;
      }
      const std::vector<unsigned char> zeros(buffer.size_bytes, 0);
      backend.copy_from_host(index, zeros.data(), zeros.size());
    }

    backend.execute();
    backend.synchronize();

    std::size_t output_bytes = 0;
    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      const auto& buffer = program.buffers[index];
      if (buffer.role != edmir::runtime::BufferRole::kOutput || buffer.size_bytes == 0) {
        continue;
      }
      std::vector<unsigned char> output(buffer.size_bytes);
      backend.copy_to_host(index, output.data(), output.size());
      output_bytes += output.size();
      for (const auto value : output) {
        if (value != 0) {
          std::cerr << "non-zero value in zero-input output buffer " << buffer.name << '\n';
          return 2;
        }
      }
    }

    std::cout << "executed target=" << program.target << " arch=" << program.arch
              << " gemm_provider=" << edmir::runtime::gemm_provider_name(program.gemm_provider)
              << " gemm_kernel=" << edmir::runtime::gemm_kernel_name(program.gemm_kernel)
              << " strategy=" << program.strategy << " ranks=" << program.problem.world_size
              << " ops=" << program.ops.size() << " checked_output_bytes=" << output_bytes
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edmir runtime failed: " << error.what() << '\n';
    return 1;
  }
}
