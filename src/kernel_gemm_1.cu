#include "kernel_gemm.h"

#include <cuda_runtime.h>

__global__ void kernel_gemm_1(half *A, half *B, half *C, int M, int N, int K,
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

void kernel_gemm_1_launch(GemmData &data, const unsigned int num_runs) {
  dim3 blockSize(16, 16);
  dim3 gridSize((data.dim_n + blockSize.x - 1) / blockSize.x,
                (data.dim_m + blockSize.y - 1) / blockSize.y);

  KernelProfiler profiler;
  for (int i = 0; i < num_runs; i++) {
    profiler.start();
    kernel_gemm_1<<<gridSize, blockSize>>>(data.dev_A, data.dev_B, data.dev_C,
                                           data.dim_m, data.dim_n, data.dim_k,
                                           data.alpha, data.beta);
    profiler.stop();
    // moved correctness check here because results accumulate on C
    if (i == 0) {
      data.check_out();
    }
  }
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << __FUNCTION__ << " GFLOPS for size (" << data.dim_m << "x"
            << data.dim_n << "x" << data.dim_k << "): "
            << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
            << std::endl;
}
