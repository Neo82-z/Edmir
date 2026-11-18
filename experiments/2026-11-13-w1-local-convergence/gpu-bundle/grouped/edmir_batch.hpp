#pragma once
#include <array>
#define edmir_generated program_host
#include "host/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_direct
#include "direct/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk2
#include "splitk2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk4
#include "splitk4/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk8
#include "splitk8/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk16
#include "splitk16/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_direct_groupedn2
#include "direct_groupedn2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk2_groupedn2
#include "splitk2_groupedn2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk4_groupedn2
#include "splitk4_groupedn2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk8_groupedn2
#include "splitk8_groupedn2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk16_groupedn2
#include "splitk16_groupedn2/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk8_fp32reduce
#include "splitk8_fp32reduce/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk8_groupedn2_fp32reduce
#include "splitk8_groupedn2_fp32reduce/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk16_fp32reduce
#include "splitk16_fp32reduce/edmir_program.hpp"
#undef edmir_generated
#define edmir_generated program_splitk16_groupedn2_fp32reduce
#include "splitk16_groupedn2_fp32reduce/edmir_program.hpp"
#undef edmir_generated
namespace edmir_batch {
struct Candidate { std::string_view name; edmir::runtime::ProgramView (*program)(); };
inline constexpr std::array candidates{
  Candidate{"host", program_host::program},
  Candidate{"host_merged_cutlass", program_host::program},
  Candidate{"host_merged_cublaslt", program_host::program},
  Candidate{"direct", program_direct::program},
  Candidate{"splitk2", program_splitk2::program},
  Candidate{"splitk4", program_splitk4::program},
  Candidate{"splitk8", program_splitk8::program},
  Candidate{"splitk16", program_splitk16::program},
  Candidate{"direct_groupedn2", program_direct_groupedn2::program},
  Candidate{"splitk2_groupedn2", program_splitk2_groupedn2::program},
  Candidate{"splitk4_groupedn2", program_splitk4_groupedn2::program},
  Candidate{"splitk8_groupedn2", program_splitk8_groupedn2::program},
  Candidate{"splitk16_groupedn2", program_splitk16_groupedn2::program},
  Candidate{"splitk8_fp32reduce", program_splitk8_fp32reduce::program},
  Candidate{"splitk8_groupedn2_fp32reduce", program_splitk8_groupedn2_fp32reduce::program},
  Candidate{"splitk16_fp32reduce", program_splitk16_fp32reduce::program},
  Candidate{"splitk16_groupedn2_fp32reduce", program_splitk16_groupedn2_fp32reduce::program},
};
}
