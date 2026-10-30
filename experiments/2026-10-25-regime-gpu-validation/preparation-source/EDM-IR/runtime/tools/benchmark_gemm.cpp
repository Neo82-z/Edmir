#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
  }
}

void check_cublas(cublasStatus_t result, const char* expression) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expression) + " failed with status " +
                             std::to_string(static_cast<int>(result)));
  }
}

#define EDMIR_CUDA(expr) check_cuda((expr), #expr)
#define EDMIR_CUBLAS(expr) check_cublas((expr), #expr)

int integer_argument(int argc, char** argv, const std::string& name, int fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return std::stoi(argv[index + 1]);
  }
  return fallback;
}

std::vector<int> list_argument(int argc, char** argv, const std::string& name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] != name) continue;
    std::vector<int> values;
    std::stringstream stream(argv[index + 1]);
    std::string item;
    while (std::getline(stream, item, ',')) values.push_back(std::stoi(item));
    return values;
  }
  return {16, 32, 64, 128};
}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto m_values = list_argument(argc, argv, "--m");
    const int n = integer_argument(argc, argv, "--n", 4096);
    const int k = integer_argument(argc, argv, "--k", 4096);
    const int warmup = integer_argument(argc, argv, "--warmup", 10);
    const int trials = integer_argument(argc, argv, "--trials", 7);
    const int iterations = integer_argument(argc, argv, "--iterations", 100);
    if (m_values.empty() || n <= 0 || k <= 0 || warmup < 0 || trials <= 0 || iterations <= 0) {
      throw std::invalid_argument("GEMM dimensions/trials/iterations must be positive");
    }

    int device_count = 0;
    EDMIR_CUDA(cudaGetDeviceCount(&device_count));
    for (int device = 0; device < device_count; ++device) {
      EDMIR_CUDA(cudaSetDevice(device));
      cudaStream_t stream = nullptr;
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      cublasHandle_t handle = nullptr;
      EDMIR_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      EDMIR_CUDA(cudaEventCreate(&start));
      EDMIR_CUDA(cudaEventCreate(&stop));
      EDMIR_CUBLAS(cublasCreate(&handle));
      EDMIR_CUBLAS(cublasSetStream(handle, stream));

      for (const int m : m_values) {
        if (m <= 0) throw std::invalid_argument("every m value must be positive");
        void* activation = nullptr;
        void* weight = nullptr;
        void* output = nullptr;
        EDMIR_CUDA(cudaMalloc(&activation, static_cast<std::size_t>(m) * k * 2));
        EDMIR_CUDA(cudaMalloc(&weight, static_cast<std::size_t>(k) * n * 2));
        EDMIR_CUDA(cudaMalloc(&output, static_cast<std::size_t>(m) * n * 2));
        EDMIR_CUDA(cudaMemsetAsync(activation, 0, static_cast<std::size_t>(m) * k * 2, stream));
        EDMIR_CUDA(cudaMemsetAsync(weight, 0, static_cast<std::size_t>(k) * n * 2, stream));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        auto launch = [&] {
          EDMIR_CUBLAS(cublasGemmEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, weight, CUDA_R_16F, n,
              activation, CUDA_R_16F, k, &beta, output, CUDA_R_16F, n,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        for (int iteration = 0; iteration < warmup; ++iteration) launch();
        EDMIR_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> samples;
        for (int trial = 0; trial < trials; ++trial) {
          EDMIR_CUDA(cudaEventRecord(start, stream));
          for (int iteration = 0; iteration < iterations; ++iteration) launch();
          EDMIR_CUDA(cudaEventRecord(stop, stream));
          EDMIR_CUDA(cudaEventSynchronize(stop));
          float elapsed_ms = 0.0f;
          EDMIR_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
          samples.push_back(elapsed_ms * 1000.0f / static_cast<float>(iterations));
        }
        const double latency_us = median(samples);
        const double tflops = 2.0 * static_cast<double>(m) * n * k /
                              (latency_us * 1.0e6);
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"device\":" << device << ",\"dtype\":\"float16\",\"m\":" << m
                  << ",\"n\":" << n << ",\"k\":" << k
                  << ",\"median_us\":" << latency_us << ",\"tflops\":" << tflops
                  << "}\n";
        EDMIR_CUDA(cudaFree(output));
        EDMIR_CUDA(cudaFree(weight));
        EDMIR_CUDA(cudaFree(activation));
      }
      EDMIR_CUBLAS(cublasDestroy(handle));
      EDMIR_CUDA(cudaEventDestroy(stop));
      EDMIR_CUDA(cudaEventDestroy(start));
      EDMIR_CUDA(cudaStreamDestroy(stream));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edmir GEMM benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
