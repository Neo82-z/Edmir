#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t result, const char* expression) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
  }
}

#define EDMIR_CUDA(expr) check_cuda((expr), #expr)

int integer_argument(int argc, char** argv, const std::string& name, int fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return std::stoi(argv[index + 1]);
  }
  return fallback;
}

std::vector<std::size_t> sizes_argument(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string(argv[index]) != "--sizes") continue;
    std::vector<std::size_t> values;
    std::stringstream stream(argv[index + 1]);
    std::string item;
    while (std::getline(stream, item, ',')) values.push_back(std::stoull(item));
    return values;
  }
  return {131072, 262144, 524288, 1048576};
}

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto sizes = sizes_argument(argc, argv);
    const int warmup = integer_argument(argc, argv, "--warmup", 10);
    const int trials = integer_argument(argc, argv, "--trials", 7);
    const int iterations = integer_argument(argc, argv, "--iterations", 100);
    if (sizes.empty() || warmup < 0 || trials <= 0 || iterations <= 0) {
      throw std::invalid_argument("copy sizes/trials/iterations must be positive");
    }
    int device_count = 0;
    EDMIR_CUDA(cudaGetDeviceCount(&device_count));
    for (int device = 0; device < device_count; ++device) {
      EDMIR_CUDA(cudaSetDevice(device));
      cudaStream_t stream = nullptr;
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      EDMIR_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      EDMIR_CUDA(cudaEventCreate(&start));
      EDMIR_CUDA(cudaEventCreate(&stop));
      for (const auto size : sizes) {
        if (size == 0) throw std::invalid_argument("every copy size must be positive");
        void* source = nullptr;
        void* destination = nullptr;
        EDMIR_CUDA(cudaMalloc(&source, size));
        EDMIR_CUDA(cudaMalloc(&destination, size));
        EDMIR_CUDA(cudaMemsetAsync(source, 0, size, stream));
        for (int iteration = 0; iteration < warmup; ++iteration) {
          EDMIR_CUDA(cudaMemcpyAsync(destination, source, size, cudaMemcpyDeviceToDevice, stream));
        }
        EDMIR_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> samples;
        for (int trial = 0; trial < trials; ++trial) {
          EDMIR_CUDA(cudaEventRecord(start, stream));
          for (int iteration = 0; iteration < iterations; ++iteration) {
            EDMIR_CUDA(cudaMemcpyAsync(destination, source, size, cudaMemcpyDeviceToDevice, stream));
          }
          EDMIR_CUDA(cudaEventRecord(stop, stream));
          EDMIR_CUDA(cudaEventSynchronize(stop));
          float elapsed_ms = 0.0f;
          EDMIR_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
          samples.push_back(elapsed_ms * 1000.0f / static_cast<float>(iterations));
        }
        const double latency_us = median(samples);
        const double bandwidth_gbps = static_cast<double>(size) / (latency_us * 1000.0);
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"device\":" << device << ",\"size_bytes\":" << size
                  << ",\"median_us\":" << latency_us
                  << ",\"bandwidth_gbps\":" << bandwidth_gbps << "}\n";
        EDMIR_CUDA(cudaFree(destination));
        EDMIR_CUDA(cudaFree(source));
      }
      EDMIR_CUDA(cudaEventDestroy(stop));
      EDMIR_CUDA(cudaEventDestroy(start));
      EDMIR_CUDA(cudaStreamDestroy(stream));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edmir copy benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
