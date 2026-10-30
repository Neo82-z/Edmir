#include "edmir/runtime/validate.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace edmir::runtime {
namespace {

bool valid_rank(std::int32_t rank, std::int32_t world_size) {
  return rank >= 0 && rank < world_size;
}

std::vector<std::int32_t> op_devices(const OpDesc& op) {
  if ((op.kind == OpKind::kTransfer || op.kind == OpKind::kFusedTransferGemm) &&
      op.peer_rank != op.rank) {
    return {op.peer_rank, op.rank};
  }
  return {op.rank};
}

std::size_t count_role(
    ProgramView program, const std::vector<std::size_t>& buffers, BufferRole role) {
  return static_cast<std::size_t>(
      std::count_if(buffers.begin(), buffers.end(), [&](std::size_t index) {
        return program.buffers[index].role == role;
      }));
}

}  // namespace

ValidationReport validate_program(ProgramView program) {
  ValidationReport report;
  const auto world_size = program.problem.world_size;
  if (program.target.empty() || program.arch.empty() || program.strategy.empty()) {
    report.errors.emplace_back("target, arch, and strategy must not be empty");
  }
  if (world_size < 2) {
    report.errors.emplace_back("world_size must be >= 2");
  }
  if (program.problem.rows_per_peer <= 0 || program.problem.k <= 0 || program.problem.n <= 0) {
    report.errors.emplace_back("problem dimensions must be positive");
  }
  if (dtype_size(program.problem.dtype) == 0) {
    report.errors.emplace_back("problem dtype is invalid");
  }
  if (program.fused_schedule.n_tiles_per_cta <= 0) {
    report.errors.emplace_back("fused n_tiles_per_cta must be positive");
  } else if (program.fused_schedule.n_tiles_per_cta > 2) {
    report.errors.emplace_back(
        "grouped-N runtime supports only n_tiles_per_cta=1 or 2");
  } else if (program.fused_schedule.n_tiles_per_cta == 2 &&
             program.gemm_provider != GemmProvider::kCUTLASS) {
    report.errors.emplace_back("grouped-N runtime requires CUTLASS");
  } else if (program.fused_schedule.n_tiles_per_cta == 2 &&
             program.communication_api != CommunicationApi::kNCCLDevice) {
    report.errors.emplace_back("grouped-N runtime requires NCCL Device API fusion");
  }
  if (program.gemm_provider == GemmProvider::kCUTLASS) {
    if (program.arch != "sm_80") {
      report.errors.emplace_back("CUTLASS v0.1 requires arch=sm_80");
    }
    if (program.problem.dtype != DType::kFloat16) {
      report.errors.emplace_back("CUTLASS v0.1 requires float16 GEMM");
    }
    if (program.problem.k % 8 != 0 || program.problem.n % 8 != 0) {
      report.errors.emplace_back("CUTLASS v0.1 requires K and N divisible by 8");
    }
    if (program.gemm_kernel == GemmKernel::kLibraryDefault) {
      report.errors.emplace_back("CUTLASS requires an explicit generated kernel");
    }
    const auto split_k_slices = gemm_kernel_split_k_slices(program.gemm_kernel);
    if (split_k_slices > 1 &&
        program.communication_api != CommunicationApi::kNCCLDevice) {
      report.errors.emplace_back("CUTLASS split-K requires NCCL Device API fusion");
    }
    if (split_k_slices > 1 &&
        program.problem.k % (32 * split_k_slices) != 0) {
      report.errors.emplace_back(
          "CUTLASS split-K requires K divisible by 32 times the slice count");
    }
  } else if (program.gemm_provider == GemmProvider::kCudaReference) {
    if (program.arch != "sm_80" || program.problem.dtype != DType::kFloat16) {
      report.errors.emplace_back("CUDA reference fused runtime requires SM80 float16");
    }
    if (program.gemm_kernel != GemmKernel::kLsaNaiveF16) {
      report.errors.emplace_back("CUDA reference fused runtime requires lsa_naive_f16");
    }
    if (program.communication_api != CommunicationApi::kNCCLDevice) {
      report.errors.emplace_back("CUDA reference fused runtime requires NCCL Device API");
    }
  } else if (program.gemm_kernel != GemmKernel::kLibraryDefault) {
    report.errors.emplace_back("cuBLAS requires the library_default GEMM kernel");
  }
  if (program.chunk_rows <= 0 ||
      (program.problem.rows_per_peer > 0 &&
       program.problem.rows_per_peer % program.chunk_rows != 0)) {
    report.errors.emplace_back("chunk_rows must divide rows_per_peer");
  }
  if (program.buffer_count <= 0) {
    report.errors.emplace_back("buffer_count must be positive");
  }
  const auto has_fused = std::any_of(
      program.ops.begin(), program.ops.end(),
      [](const OpDesc& op) { return op.kind == OpKind::kFusedTransferGemm; });
  if (program.fused_schedule.n_tiles_per_cta == 2 && !has_fused) {
    report.errors.emplace_back("grouped-N runtime requires a fused Program");
  }

  std::unordered_set<std::string> buffer_names;
  for (const auto& buffer : program.buffers) {
    if (buffer.name.empty()) {
      report.errors.emplace_back("buffer name must not be empty");
    } else if (!buffer_names.emplace(buffer.name).second) {
      report.errors.emplace_back("duplicate buffer name: " + std::string(buffer.name));
    }
    if (!valid_rank(buffer.rank, world_size)) {
      report.errors.emplace_back("buffer rank is outside the target world");
    }
  }

  std::unordered_set<std::string> op_ids;
  for (const auto& op : program.ops) {
    if (op.id.empty()) {
      report.errors.emplace_back("op id must not be empty");
    } else if (!op_ids.emplace(op.id).second) {
      report.errors.emplace_back("duplicate op id: " + std::string(op.id));
    }
    if (!valid_rank(op.rank, world_size)) {
      report.errors.emplace_back("op rank is outside the target world");
    }
    switch (op.kind) {
      case OpKind::kTransfer:
        if (!valid_rank(op.peer_rank, world_size)) {
          report.errors.emplace_back("transfer peer rank is outside the target world");
        }
        if (op.queue != QueueKind::kComm || op.rows <= 0) {
          report.errors.emplace_back("transfer requires comm queue and positive rows");
        }
        break;
      case OpKind::kGemm:
        if (op.queue != QueueKind::kCompute || op.rows <= 0) {
          report.errors.emplace_back("GEMM requires compute queue and positive rows");
        }
        break;
      case OpKind::kBarrier:
        if (op.queue != QueueKind::kControl) {
          report.errors.emplace_back("barrier requires control queue");
        }
        break;
      case OpKind::kFusedTransferGemm:
        if (!valid_rank(op.peer_rank, world_size)) {
          report.errors.emplace_back("fused op peer rank is outside the target world");
        }
        if (op.queue != QueueKind::kCompute || op.rows <= 0) {
          report.errors.emplace_back("fused op requires compute queue and positive rows");
        }
        if (program.communication_api != CommunicationApi::kNCCLDevice) {
          report.errors.emplace_back("fused op requires NCCL Device API");
        }
        break;
    }
  }

  std::vector<std::size_t> reads(program.ops.size(), 0);
  std::vector<std::size_t> writes(program.ops.size(), 0);
  std::vector<std::vector<std::size_t>> read_buffers(program.ops.size());
  std::vector<std::vector<std::size_t>> write_buffers(program.ops.size());
  for (const auto& access : program.accesses) {
    if (access.op >= program.ops.size() || access.buffer >= program.buffers.size()) {
      report.errors.emplace_back("buffer access index is out of range");
      continue;
    }
    (access.write ? writes : reads)[access.op] += 1;
    (access.write ? write_buffers : read_buffers)[access.op].push_back(access.buffer);
  }
  for (std::size_t index = 0; index < program.ops.size(); ++index) {
    const auto& op = program.ops[index];
    const auto kind = op.kind;
    if (kind == OpKind::kTransfer && (reads[index] != 1 || writes[index] != 1)) {
      report.errors.emplace_back("transfer requires exactly one read and one write access");
    }
    if (kind == OpKind::kTransfer && reads[index] == 1 && writes[index] == 1) {
      const auto& source = program.buffers[read_buffers[index][0]];
      const auto& destination = program.buffers[write_buffers[index][0]];
      const auto expected_bytes = static_cast<std::size_t>(op.rows) *
                                  static_cast<std::size_t>(program.problem.k) *
                                  dtype_size(program.problem.dtype);
      if (source.role != BufferRole::kInput || destination.role != BufferRole::kStaging) {
        report.errors.emplace_back("transfer requires input -> staging buffer roles");
      }
      if (source.rank != op.peer_rank || destination.rank != op.rank) {
        report.errors.emplace_back("transfer buffer ranks do not match source/destination ranks");
      }
      if (source.size_bytes < expected_bytes || destination.size_bytes < expected_bytes) {
        report.errors.emplace_back("transfer buffer is smaller than its tensor tile");
      }
    }
    if (kind == OpKind::kGemm && (reads[index] < 2 || writes[index] != 1)) {
      report.errors.emplace_back("GEMM requires at least two reads and exactly one write access");
    }
    if (kind == OpKind::kGemm && reads[index] >= 2 && writes[index] == 1) {
      if (count_role(program, read_buffers[index], BufferRole::kStaging) != 1 ||
          count_role(program, read_buffers[index], BufferRole::kWeight) != 1 ||
          count_role(program, read_buffers[index], BufferRole::kToken) > 1 ||
          program.buffers[write_buffers[index][0]].role != BufferRole::kOutput) {
        report.errors.emplace_back("GEMM requires staging + weight -> output buffer roles");
      }
      for (const auto buffer : read_buffers[index]) {
        const auto role = program.buffers[buffer].role;
        if (role != BufferRole::kStaging && role != BufferRole::kWeight &&
            role != BufferRole::kToken) {
          report.errors.emplace_back("GEMM has an unsupported read buffer role");
        }
      }
      for (const auto buffer : read_buffers[index]) {
        if (program.buffers[buffer].rank != op.rank) {
          report.errors.emplace_back("GEMM read buffer is not local to its execution rank");
        }
      }
      const auto& output = program.buffers[write_buffers[index][0]];
      if (output.rank != op.rank) {
        report.errors.emplace_back("GEMM output buffer is not local to its execution rank");
      }
      const auto element_size = dtype_size(program.problem.dtype);
      const auto output_bytes = static_cast<std::size_t>(op.rows) *
                                static_cast<std::size_t>(program.problem.n) * element_size;
      if (output.size_bytes < output_bytes) {
        report.errors.emplace_back("GEMM output buffer is smaller than its tensor tile");
      }
      const auto staging = std::find_if(
          read_buffers[index].begin(), read_buffers[index].end(), [&](std::size_t buffer) {
            return program.buffers[buffer].role == BufferRole::kStaging;
          });
      const auto weight = std::find_if(
          read_buffers[index].begin(), read_buffers[index].end(), [&](std::size_t buffer) {
            return program.buffers[buffer].role == BufferRole::kWeight;
          });
      const auto staging_bytes = static_cast<std::size_t>(op.rows) *
                                 static_cast<std::size_t>(program.problem.k) * element_size;
      const auto weight_bytes = static_cast<std::size_t>(program.problem.k) *
                                static_cast<std::size_t>(program.problem.n) * element_size;
      if (staging != read_buffers[index].end() &&
          program.buffers[*staging].size_bytes < staging_bytes) {
        report.errors.emplace_back("GEMM staging buffer is smaller than its tensor tile");
      }
      if (weight != read_buffers[index].end() &&
          program.buffers[*weight].size_bytes < weight_bytes) {
        report.errors.emplace_back("GEMM weight buffer is smaller than its matrix");
      }
    }
    if (kind == OpKind::kFusedTransferGemm &&
        (reads[index] != 2 || writes[index] != 1)) {
      report.errors.emplace_back("fused op requires exactly two reads and one write access");
    }
    if (kind == OpKind::kFusedTransferGemm && reads[index] == 2 && writes[index] == 1) {
      if (count_role(program, read_buffers[index], BufferRole::kInput) != 1 ||
          count_role(program, read_buffers[index], BufferRole::kWeight) != 1 ||
          program.buffers[write_buffers[index][0]].role != BufferRole::kOutput) {
        report.errors.emplace_back("fused op requires input + weight -> output buffer roles");
      }
      const auto input = std::find_if(
          read_buffers[index].begin(), read_buffers[index].end(), [&](std::size_t buffer) {
            return program.buffers[buffer].role == BufferRole::kInput;
          });
      const auto weight = std::find_if(
          read_buffers[index].begin(), read_buffers[index].end(), [&](std::size_t buffer) {
            return program.buffers[buffer].role == BufferRole::kWeight;
          });
      const auto& output = program.buffers[write_buffers[index][0]];
      if (input != read_buffers[index].end() &&
          program.buffers[*input].rank != op.peer_rank) {
        report.errors.emplace_back("fused input rank does not match peer rank");
      }
      if (weight != read_buffers[index].end() &&
          program.buffers[*weight].rank != op.rank) {
        report.errors.emplace_back("fused weight is not local to the destination rank");
      }
      if (output.rank != op.rank) {
        report.errors.emplace_back("fused output is not local to the destination rank");
      }
      const auto element_size = dtype_size(program.problem.dtype);
      const auto input_bytes = static_cast<std::size_t>(op.rows) *
                               static_cast<std::size_t>(program.problem.k) * element_size;
      const auto weight_bytes = static_cast<std::size_t>(program.problem.k) *
                                static_cast<std::size_t>(program.problem.n) * element_size;
      const auto output_bytes = static_cast<std::size_t>(op.rows) *
                                static_cast<std::size_t>(program.problem.n) * element_size;
      if (input != read_buffers[index].end() &&
          program.buffers[*input].size_bytes < input_bytes) {
        report.errors.emplace_back("fused input buffer is smaller than its tensor tile");
      }
      if (weight != read_buffers[index].end() &&
          program.buffers[*weight].size_bytes < weight_bytes) {
        report.errors.emplace_back("fused weight buffer is smaller than its matrix");
      }
      if (output.size_bytes < output_bytes) {
        report.errors.emplace_back("fused output buffer is smaller than its tensor tile");
      }
    }
    if (kind == OpKind::kBarrier) {
      if (writes[index] != 1 ||
          program.buffers[write_buffers[index][0]].role != BufferRole::kToken) {
        report.errors.emplace_back("barrier requires exactly one token output");
      }
      for (const auto buffer : read_buffers[index]) {
        if (program.buffers[buffer].rank != op.rank) {
          report.errors.emplace_back("barrier input is not local to its execution rank");
        }
      }
    }
  }

  for (const auto& dependency : program.dependencies) {
    if (dependency.src_op >= program.ops.size() || dependency.dst_op >= program.ops.size()) {
      report.errors.emplace_back("dependency op index is out of range");
    }
    if (dependency.src_op == dependency.dst_op) {
      report.errors.emplace_back("dependency self-cycle is invalid");
    }
    if (dependency.buffer < -1 ||
        dependency.buffer >= static_cast<std::int32_t>(program.buffers.size())) {
      report.errors.emplace_back("dependency buffer index is out of range");
    }
  }

  try {
    static_cast<void>(topological_order(program));
  } catch (const std::invalid_argument& error) {
    report.errors.emplace_back(error.what());
  }
  return report;
}

std::vector<std::size_t> topological_order(ProgramView program) {
  std::vector<std::size_t> indegree(program.ops.size(), 0);
  std::vector<std::vector<std::size_t>> successors(program.ops.size());
  std::unordered_set<std::string> edges;
  for (const auto& dependency : program.dependencies) {
    if (dependency.src_op >= program.ops.size() || dependency.dst_op >= program.ops.size()) {
      continue;
    }
    const auto key = std::to_string(dependency.src_op) + ":" +
                     std::to_string(dependency.dst_op);
    if (!edges.emplace(key).second) {
      continue;
    }
    successors[dependency.src_op].push_back(dependency.dst_op);
    indegree[dependency.dst_op] += 1;
  }

  std::deque<std::size_t> ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push_back(index);
    }
  }
  std::vector<std::size_t> order;
  while (!ready.empty()) {
    const auto current = ready.front();
    ready.pop_front();
    order.push_back(current);
    for (const auto successor : successors[current]) {
      indegree[successor] -= 1;
      if (indegree[successor] == 0) {
        ready.push_back(successor);
      }
    }
  }
  if (order.size() != program.ops.size()) {
    throw std::invalid_argument("runtime dependency graph contains a cycle");
  }
  return order;
}

}  // namespace edmir::runtime
