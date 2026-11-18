#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "edmir/runtime/program.hpp"

namespace edmir::runtime {

struct ValidationReport {
  std::vector<std::string> errors;

  bool valid() const { return errors.empty(); }
};

ValidationReport validate_program(ProgramView program);
std::vector<std::size_t> topological_order(ProgramView program);

}  // namespace edmir::runtime

