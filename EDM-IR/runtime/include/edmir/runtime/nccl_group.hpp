#pragma once

#include <nccl.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>

namespace edmir::runtime {

// Each backend retains its own streams, events, peer windows and matrices.
// Only one candidate may use the group at a time.
class NcclGroup {
 public:
  explicit NcclGroup(const std::vector<int>& devices)
      : devices_(devices), handles_(devices.size(), nullptr) {
    if (devices.empty()) throw std::invalid_argument("empty NCCL device map");
    const auto status = ncclCommInitAll(handles_.data(), static_cast<int>(devices.size()), devices.data());
    if (status != ncclSuccess) {
      for (auto comm : handles_) if (comm) ncclCommAbort(comm);
      throw std::runtime_error(ncclGetErrorString(status));
    }
  }
  NcclGroup(const NcclGroup&) = delete;
  NcclGroup& operator=(const NcclGroup&) = delete;
  ~NcclGroup() {
    for (std::size_t i = 0; i < devices_.size(); ++i) {
      cudaSetDevice(devices_[i]);
      ncclCommDestroy(handles_[i]);
    }
  }
  const std::vector<ncclComm_t>& for_devices(const std::vector<int>& devices) const {
    if (devices != devices_) throw std::invalid_argument("NCCL group device map mismatch");
    return handles_;
  }
 private:
  std::vector<int> devices_;
  std::vector<ncclComm_t> handles_;
};

}  // namespace edmir::runtime
