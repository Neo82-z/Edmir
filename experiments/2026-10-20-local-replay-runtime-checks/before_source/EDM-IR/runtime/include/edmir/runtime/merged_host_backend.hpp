#pragma once

#include <memory>
#include <string>
#include "edmir/runtime/cuda_backend.hpp"

namespace edmir::runtime {
enum class MergedProvider { kCUTLASS, kCuBLASLt };

// Experiment-only materialize-then-GEMM baseline. Receives land directly in
// source-ordered [P*M,K] storage. It is not a compiler fallback implementation.
class MergedHostBackend {
 public:
  MergedHostBackend(ProgramView program, MergedProvider provider);
  ~MergedHostBackend();
  MergedHostBackend(const MergedHostBackend&) = delete;
  MergedHostBackend& operator=(const MergedHostBackend&) = delete;
  void prepare();
  void execute();
  void synchronize();
  void copy_from_host(std::size_t buffer, const void* source, std::size_t bytes);
  void copy_to_host(std::size_t buffer, void* destination, std::size_t bytes) const;
  std::string algorithm_id() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace edmir::runtime
