#include "kernel_gemm.h"

#include <cmath>
#include <cuda_runtime.h>

__global__ void kernel_gemm_1(half *A, half *B, half *C, int M, int N, int K,
                              half alpha, half beta) {
  // mismatch between x/y dimention and row/col dimention, causing uncoalsced
  // gmem access.
  // in threadIdx, x is the major.
  // in matrix C, row is the major.
  // so, in order to achieve coalsced memory access, x should be used as col
  // index here (it IS in the first place).
  // let's fix this in kernel 2 for comparison.
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    half value = 0.0f;

    for (int k = 0; k < K; ++k) {
      value += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * value + beta * C[row * N + col];
  }
}

void kernel_gemm_1_launch(GemmData &data, const unsigned int num_runs) {
  constexpr int kBlockSize = 32;
  dim3 block_size(kBlockSize, kBlockSize);
  dim3 grid_size(std::ceil(data.dim_n / block_size.x),
                 std::ceil(data.dim_m / block_size.y));

  auto kernel_func = [&]() {
    kernel_gemm_1<<<grid_size, block_size, 0, cudaStreamPerThread>>>(
        data.dev_A, data.dev_B, data.dev_C, data.dim_m, data.dim_n, data.dim_k,
        data.alpha, data.beta);
  };
  kernel_func();
  data.check_out();

  KernelProfiler profiler;
  for (int i = 0; i < num_runs; i++) {
    profiler.start();
    kernel_func();
    profiler.stop();
  }
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << "kernel 1 naive GFLOPS for size (" << data.dim_m << "x"
            << data.dim_n << "x" << data.dim_k << "): "
            << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
            << std::endl;
}
