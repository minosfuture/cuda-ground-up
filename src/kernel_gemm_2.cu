#include "kernel_gemm.h"

#include <cmath>
#include <cuda_runtime.h>

__global__ void kernel_gemm_2(half *A, half *B, half *C, int M, int N, int K,
                              half alpha, half beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    half value = 0.0f;

    for (int k = 0; k < K; ++k) {
      value += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * value + beta * C[row * N + col];
  }
}

void kernel_gemm_2_launch(GemmData &data, const unsigned int num_runs) {
  auto func = [&](dim3 block_size) {
    dim3 grid_size(std::ceil(data.dim_n / block_size.x),
                   std::ceil(data.dim_m / block_size.y));

    KernelProfiler profiler;
    for (int i = 0; i < num_runs; i++) {
      profiler.start();
      kernel_gemm_2<<<grid_size, block_size, 0, cudaStreamPerThread>>>(
          data.dev_A, data.dev_B, data.dev_C, data.dim_m, data.dim_n,
          data.dim_k, data.alpha, data.beta);
      profiler.stop();
      // moved correctness check here because results accumulate on C
      if (i == 0) {
        data.check_out();
      }
    }
    CUDA_CHECK(cudaPeekAtLastError());

    std::cout << "kernel 2 (coalesced gmem access) (blockDim(" << block_size.x
              << "," << block_size.y << ")) GFLOPS for size (" << data.dim_m
              << "x" << data.dim_n << "x" << data.dim_k << "): "
              << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
              << std::endl;
  };
  func(dim3(16, 16));
  data.reset_c();
  func(dim3(32, 8));
  data.reset_c();
  func(dim3(64, 4));
  data.reset_c();
  func(dim3(128, 2));
  data.reset_c();
  func(dim3(256, 1));
}
