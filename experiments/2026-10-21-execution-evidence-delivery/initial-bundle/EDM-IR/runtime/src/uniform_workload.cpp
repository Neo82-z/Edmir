#include "edmir/runtime/uniform_workload.hpp"

#include <limits>
#include <stdexcept>
#include "edmir/runtime/validate.hpp"

namespace edmir::runtime {
namespace {
constexpr auto missing = std::numeric_limits<std::size_t>::max();

std::size_t access(ProgramView p, std::size_t op, bool write, BufferRole role) {
  auto result = missing;
  for (const auto& a : p.accesses) {
    if (a.op != op || a.write != write || p.buffers[a.buffer].role != role) continue;
    if (result != missing) throw std::invalid_argument("duplicate uniform workload access");
    result = a.buffer;
  }
  if (result == missing) throw std::invalid_argument("missing uniform workload access");
  return result;
}
void assign_once(std::vector<std::size_t>& ids, std::size_t key, std::size_t value) {
  if (ids.at(key) != missing) throw std::invalid_argument("chunked or duplicate source/destination");
  ids[key] = value;
}
}

std::size_t merged_row_offset(int source, int rows_per_peer, int world_size) {
  if (source < 0 || source >= world_size || rows_per_peer <= 0)
    throw std::invalid_argument("invalid merged row coordinates");
  return static_cast<std::size_t>(source)*rows_per_peer;
}

UniformWorkload uniform_workload(ProgramView p) {
  const auto report = validate_program(p);
  if (!report.valid()) throw std::invalid_argument(report.errors.front());
  const int count = p.problem.world_size;
  if (p.problem.dtype != DType::kFloat16 || count < 2)
    throw std::invalid_argument("uniform experiment requires FP16 and at least two ranks");
  if (static_cast<std::int64_t>(count)*p.problem.rows_per_peer > std::numeric_limits<int>::max())
    throw std::invalid_argument("merged row dimension exceeds GEMM int range");
  UniformWorkload w;
  w.inputs.assign(static_cast<std::size_t>(count)*count, missing);
  w.outputs = w.inputs;
  w.weights.assign(count, missing);
  std::vector<std::size_t> stages(w.inputs.size(), missing);
  bool saw_host = false, saw_fused = false;
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    const auto& op = p.ops[i];
    if (op.kind == OpKind::kBarrier) continue;
    if (op.rows != p.problem.rows_per_peer || op.rank < 0 || op.rank >= count ||
        op.peer_rank < 0 || op.peer_rank >= count)
      throw std::invalid_argument("uniform experiment rejects chunked or invalid operations");
    const auto key = static_cast<std::size_t>(op.peer_rank)*count+op.rank;
    if (op.kind == OpKind::kTransfer) {
      saw_host = true;
      assign_once(w.inputs, key, access(p, i, false, BufferRole::kInput));
      assign_once(stages, key, access(p, i, true, BufferRole::kStaging));
    } else if (op.kind == OpKind::kFusedTransferGemm) {
      saw_fused = true;
      assign_once(w.inputs, key, access(p, i, false, BufferRole::kInput));
    } else if (op.kind != OpKind::kGemm) {
      throw std::invalid_argument("unexpected uniform experiment operation");
    }
    if (op.kind == OpKind::kTransfer) continue;
    assign_once(w.outputs, key, access(p, i, true, BufferRole::kOutput));
    const auto weight = access(p, i, false, BufferRole::kWeight);
    if (w.weights[op.rank] != missing && w.weights[op.rank] != weight)
      throw std::invalid_argument("merged GEMM requires one shared weight per destination");
    w.weights[op.rank] = weight;
  }
  if (saw_fused == saw_host) throw std::invalid_argument("mixed or empty uniform workload");
  w.fused = saw_fused;
  for (std::size_t i = 0; i < p.ops.size(); ++i) {
    const auto& op = p.ops[i];
    if (op.kind != OpKind::kGemm) continue;
    const auto key = static_cast<std::size_t>(op.peer_rank)*count+op.rank;
    if (access(p, i, false, BufferRole::kStaging) != stages[key])
      throw std::invalid_argument("GEMM uses a different source block");
  }
  for (int s = 0; s < count; ++s) {
    for (int d = 0; d < count; ++d) {
      const auto key = static_cast<std::size_t>(s)*count+d;
      if (w.inputs[key] == missing || w.outputs[key] == missing)
        throw std::invalid_argument("incomplete uniform workload");
      const auto& a = p.buffers[w.inputs[key]];
      const auto& c = p.buffers[w.outputs[key]];
      if (a.rank != s || c.rank != d ||
          a.size_bytes != 2ULL*p.problem.rows_per_peer*p.problem.k ||
          c.size_bytes != 2ULL*p.problem.rows_per_peer*p.problem.n)
        throw std::invalid_argument("uniform matrix layout mismatch");
    }
    if (w.weights[s] == missing || p.buffers[w.weights[s]].rank != s ||
        p.buffers[w.weights[s]].size_bytes != 2ULL*p.problem.k*p.problem.n)
      throw std::invalid_argument("uniform weight layout mismatch");
  }
  return w;
}
}  // namespace edmir::runtime
