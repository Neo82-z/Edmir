#include <iostream>
#include "edmir/runtime/validate.hpp"
#define edmir_generated candidate_0
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/grouped-splitk16_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_1
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/grouped-splitk16_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_2
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/grouped-splitk8_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_3
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/grouped-splitk8_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_4
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/large_n-splitk16_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_5
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/large_n-splitk16_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_6
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/large_n-splitk8_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_7
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/large_n-splitk8_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_8
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m256-splitk16_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_9
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m256-splitk16_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_10
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m256-splitk8_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_11
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m256-splitk8_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_12
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m512-splitk16_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_13
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m512-splitk16_groupedn2_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_14
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m512-splitk8_fp32reduce.hpp"
#undef edmir_generated
#define edmir_generated candidate_15
#include "/Users/neo/code/llm-serving/experiments/2026-11-13-w1-local-convergence/round-05/fp32-manifests/m512-splitk8_groupedn2_fp32reduce.hpp"
#undef edmir_generated
int main() {
int failures = 0;
{ const auto result = edmir::runtime::validate_program(candidate_0::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_1::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_2::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_3::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_4::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_5::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_6::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_7::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_8::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_9::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_10::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_11::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_12::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_13::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_14::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
{ const auto result = edmir::runtime::validate_program(candidate_15::program());
for (auto& error : result.errors) { std::cerr << error << "\n"; ++failures; } }
return failures != 0;
}
