#pragma once
#include <array>
#define edmir_generated program_pipeline_serial
#include "pipeline_serial/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_pipeline
#include "pipeline/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk4
#include "splitk4/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk4_groupedn2
#include "splitk4_groupedn2/edmir_program.hpp"
#undef edmir_generated
namespace edmir_batch {
struct Candidate { std::string_view name; edmir::runtime::ProgramView (*program)(); };
inline constexpr std::array candidates{
  Candidate{"pipeline_serial", program_pipeline_serial::program},
  Candidate{"pipeline", program_pipeline::program},
  Candidate{"splitk4", program_splitk4::program},
  Candidate{"splitk4_groupedn2", program_splitk4_groupedn2::program},
};
}
