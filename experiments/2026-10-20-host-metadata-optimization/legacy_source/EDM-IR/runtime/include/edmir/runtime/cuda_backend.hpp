#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "edmir/runtime/program.hpp"

namespace edmir::runtime {

struct CudaBackendOptions {
  // Maps logical Program ranks to CUDA device ordinals. An empty vector uses
  // {0, ..., world_size - 1}, which composes naturally with CUDA_VISIBLE_DEVICES.
  std::vector<int> devices;
  // Emit stable edm.phase/runtime and edm.op NVTX ranges. Disabled by default
  // so benchmark calibration does not include annotation overhead.
  bool enable_nvtx = false;
};

class CudaBackend {
 public:
  explicit CudaBackend(ProgramView program, CudaBackendOptions options = {});
  ~CudaBackend();

  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;
  CudaBackend(CudaBackend&&) noexcept;
  CudaBackend& operator=(CudaBackend&&) noexcept;

  void prepare();
  void execute();
  void synchronize();

  std::size_t find_buffer(std::string_view name) const;
  void* buffer_data(std::size_t buffer);
  const void* buffer_data(std::size_t buffer) const;
  void copy_from_host(std::size_t buffer, const void* source, std::size_t size_bytes);
  void copy_to_host(std::size_t buffer, void* destination, std::size_t size_bytes) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace edmir::runtime
