# CMake generated Testfile for 
# Source directory: /workspace/precision-bundle/source/EDM-IR/runtime
# Build directory: /workspace/build-precision
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[w1_m512_layout]=] "/workspace/build-precision/w1_m512_layout")
set_tests_properties([=[w1_m512_layout]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/precision-bundle/generated_targets.cmake;5;add_test;/workspace/precision-bundle/generated_targets.cmake;0;;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;149;include;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[w1_large_n_layout]=] "/workspace/build-precision/w1_large_n_layout")
set_tests_properties([=[w1_large_n_layout]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/precision-bundle/generated_targets.cmake;17;add_test;/workspace/precision-bundle/generated_targets.cmake;0;;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;149;include;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[w1_grouped_layout]=] "/workspace/build-precision/w1_grouped_layout")
set_tests_properties([=[w1_grouped_layout]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/precision-bundle/generated_targets.cmake;29;add_test;/workspace/precision-bundle/generated_targets.cmake;0;;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;149;include;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[w1_m256_layout]=] "/workspace/build-precision/w1_m256_layout")
set_tests_properties([=[w1_m256_layout]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/precision-bundle/generated_targets.cmake;41;add_test;/workspace/precision-bundle/generated_targets.cmake;0;;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;149;include;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[edmir_runtime_program_test]=] "/workspace/build-precision/edmir_runtime_program_test")
set_tests_properties([=[edmir_runtime_program_test]=] PROPERTIES  TIMEOUT "30" _BACKTRACE_TRIPLES "/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;156;add_test;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[edmir_prepared_graph_test]=] "/workspace/build-precision/edmir_prepared_graph_test")
set_tests_properties([=[edmir_prepared_graph_test]=] PROPERTIES  TIMEOUT "30" _BACKTRACE_TRIPLES "/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;159;add_test;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
add_test([=[edmir_split_k_workspace_test]=] "/workspace/build-precision/edmir_split_k_workspace_test")
set_tests_properties([=[edmir_split_k_workspace_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;162;add_test;/workspace/precision-bundle/source/EDM-IR/runtime/CMakeLists.txt;0;")
