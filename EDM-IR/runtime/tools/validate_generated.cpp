#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "edmir/runtime/cuda_backend.hpp"
#include "edmir/runtime/validate.hpp"
#include "edmir_program.hpp"

namespace {

using edmir::runtime::BufferRole;
using edmir::runtime::OpKind;
using edmir::runtime::ProgramView;

std::vector<std::size_t> accesses(
    ProgramView program, std::size_t op, bool write) {
  std::vector<std::size_t> result;
  for (const auto& access : program.accesses) {
    if (access.op == op && access.write == write) {
      result.push_back(access.buffer);
    }
  }
  return result;
}

std::size_t by_role(
    ProgramView program, const std::vector<std::size_t>& candidates, BufferRole role) {
  const auto found = std::find_if(candidates.begin(), candidates.end(), [&](std::size_t index) {
    return program.buffers[index].role == role;
  });
  if (found == candidates.end()) {
    throw std::logic_error("operation is missing a required buffer role");
  }
  return *found;
}

bool is_external(BufferRole role) {
  return role == BufferRole::kInput || role == BufferRole::kWeight;
}

void initialize_external(
    ProgramView program, std::vector<std::vector<float>>& host_buffers) {
  for (std::size_t buffer = 0; buffer < program.buffers.size(); ++buffer) {
    const auto& desc = program.buffers[buffer];
    if (!is_external(desc.role)) {
      continue;
    }
    auto& values = host_buffers[buffer];
    if (desc.role == BufferRole::kInput) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        const int numerator = static_cast<int>((index * 17 + buffer * 5) % 9) - 4;
        values[index] = static_cast<float>(numerator) / 8.0f;
      }
      continue;
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
      const int numerator = static_cast<int>((index * 7 + buffer * 3) % 5) - 2;
      values[index] = static_cast<float>(numerator) / 16.0f;
    }
  }
}

void interpret_on_cpu(ProgramView program, std::vector<std::vector<float>>& buffers) {
  const auto order = edmir::runtime::topological_order(program);
  for (const auto op_index : order) {
    const auto& op = program.ops[op_index];
    if (op.kind == OpKind::kBarrier) {
      continue;
    }
    const auto reads = accesses(program, op_index, false);
    const auto writes = accesses(program, op_index, true);
    if (op.kind == OpKind::kTransfer) {
      if (reads.size() != 1 || writes.size() != 1) {
        throw std::logic_error("transfer access arity changed after validation");
      }
      const auto elements = static_cast<std::size_t>(op.rows) *
                            static_cast<std::size_t>(program.problem.k);
      std::copy_n(buffers[reads[0]].begin(), elements, buffers[writes[0]].begin());
      continue;
    }

    const auto activation = by_role(program, reads, BufferRole::kStaging);
    const auto weight = by_role(program, reads, BufferRole::kWeight);
    const auto output = by_role(program, writes, BufferRole::kOutput);
    const auto m = static_cast<std::size_t>(op.rows);
    const auto n = static_cast<std::size_t>(program.problem.n);
    const auto k = static_cast<std::size_t>(program.problem.k);
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        float value = 0.0f;
        for (std::size_t inner = 0; inner < k; ++inner) {
          value += buffers[activation][row * k + inner] *
                   buffers[weight][inner * n + column];
        }
        buffers[output][row * n + column] = value;
      }
    }
  }
}

}  // namespace

int main() {
  try {
    const auto program = edmir_generated::program();
    if (program.problem.dtype != edmir::runtime::DType::kFloat32) {
      throw std::invalid_argument(
          "numerical validation currently requires a generated float32 Program");
    }

    std::vector<std::vector<float>> reference(program.buffers.size());
    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      const auto size = program.buffers[index].size_bytes / sizeof(float);
      reference[index].assign(size, 0.0f);
    }
    initialize_external(program, reference);

    edmir::runtime::CudaBackend backend(program);
    backend.prepare();
    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      if (!is_external(program.buffers[index].role)) {
        continue;
      }
      backend.copy_from_host(
          index, reference[index].data(), reference[index].size() * sizeof(float));
    }

    interpret_on_cpu(program, reference);
    backend.execute();
    backend.synchronize();

    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    std::size_t checked_elements = 0;
    std::string worst_buffer;
    for (std::size_t index = 0; index < program.buffers.size(); ++index) {
      if (program.buffers[index].role != BufferRole::kOutput) {
        continue;
      }
      std::vector<float> actual(reference[index].size());
      backend.copy_to_host(index, actual.data(), actual.size() * sizeof(float));
      for (std::size_t element = 0; element < actual.size(); ++element) {
        const auto absolute = std::abs(
            static_cast<double>(actual[element]) - static_cast<double>(reference[index][element]));
        const auto scale = std::max(std::abs(static_cast<double>(reference[index][element])), 1.0);
        const auto relative = absolute / scale;
        if (absolute > max_absolute_error) {
          max_absolute_error = absolute;
          worst_buffer = std::string(program.buffers[index].name);
        }
        max_relative_error = std::max(max_relative_error, relative);
        ++checked_elements;
      }
    }

    constexpr double kAbsoluteTolerance = 1.0e-4;
    constexpr double kRelativeTolerance = 1.0e-4;
    std::cout << "validated target=" << program.target << " arch=" << program.arch
              << " strategy=" << program.strategy << " ranks=" << program.problem.world_size
              << " ops=" << program.ops.size() << " checked_elements=" << checked_elements
              << " max_abs_error=" << max_absolute_error
              << " max_rel_error=" << max_relative_error
              << " worst_buffer=" << worst_buffer << '\n';
    if (max_absolute_error > kAbsoluteTolerance &&
        max_relative_error > kRelativeTolerance) {
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edmir numerical validation failed: " << error.what() << '\n';
    return 1;
  }
}
