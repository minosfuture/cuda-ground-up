#pragma once
#include "host_utils.h"

#include <numeric>
#include <vector>

#include <cuda_runtime.h>

struct KernelProfiler {
  cudaEvent_t start_evt;
  cudaEvent_t stop_evt;
  std::vector<float> time_in_ms;

  KernelProfiler() {
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_evt, cudaStreamPerThread)); }
  void stop() {
    CUDA_CHECK(cudaEventRecord(stop_evt, cudaStreamPerThread));
    float elapsed_ms;
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    time_in_ms.push_back(elapsed_ms);
  }

  double log_gemm_stats(const unsigned int M, const unsigned int N,
                        const unsigned int K) {
    double avg_time_ms =
        std::accumulate(time_in_ms.begin(), time_in_ms.end(), 0.0) /
        time_in_ms.size();
    double total_flop = 2.0 * M * N * K;
    double gflops = total_flop / (avg_time_ms * 1e6);
    time_in_ms.clear();
    return gflops;
  }

  double log_vec_add_stats(const unsigned int N) {
    double avg_time_ms =
        std::accumulate(time_in_ms.begin(), time_in_ms.end(), 0.0) /
        time_in_ms.size();
    double total_flop = N;
    double gflops = total_flop / (avg_time_ms * 1e6);
    time_in_ms.clear();
    return gflops;
  }
};
