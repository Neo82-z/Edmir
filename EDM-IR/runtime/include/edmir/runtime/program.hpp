#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace edmir::runtime {

enum class DType : std::uint8_t {
  kFloat16,
  kBFloat16,
  kFloat32,
};

enum class BufferRole : std::uint8_t {
  kInput,
  kWeight,
  kStaging,
  kOutput,
  kToken,
};

enum class OpKind : std::uint8_t {
  kTransfer,
  kGemm,
  kBarrier,
};

enum class QueueKind : std::uint8_t {
  kComm,
  kCompute,
  kControl,
};

enum class DependencyKind : std::uint8_t {
  kQueue,
  kReadAfterWrite,
  kBufferReuse,
  kWriteAfterWrite,
};

struct ProblemDesc {
  std::int32_t world_size;
  std::int32_t rows_per_peer;
  std::int32_t k;
  std::int32_t n;
  DType dtype;
};

struct BufferDesc {
  std::string_view name;
  std::int32_t rank;
  BufferRole role;
  std::size_t size_bytes;
};

struct OpDesc {
  std::string_view id;
  OpKind kind;
  std::int32_t rank;
  std::int32_t peer_rank;
  QueueKind queue;
  std::string_view tile;
  std::int32_t rows;
};

struct AccessDesc {
  std::size_t op;
  std::size_t buffer;
  bool write;
};

struct DependencyDesc {
  std::size_t src_op;
  std::size_t dst_op;
  DependencyKind kind;
  std::int32_t buffer;
};

struct ProgramView {
  std::string_view target;
  std::string_view arch;
  std::string_view strategy;
  ProblemDesc problem;
  std::int32_t chunk_rows;
  std::int32_t buffer_count;
  std::span<const BufferDesc> buffers;
  std::span<const OpDesc> ops;
  std::span<const AccessDesc> accesses;
  std::span<const DependencyDesc> dependencies;
};

constexpr std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kFloat32:
      return 4;
  }
  return 0;
}

}  // namespace edmir::runtime

