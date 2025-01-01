#include "kernel_gemm.h"

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

constexpr int kTileK = 32;

__global__ void kernel_gemm_3(half *A, half *B, half *C, int M, int N, int K,
                              half alpha, half beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  half value = 0.0f;
  extern __shared__ half shmem[];
  half *A_shmem = shmem;
  half *B_shmem = &shmem[blockDim.y * kTileK];

  A += blockIdx.y * blockDim.y * K;
  B += blockIdx.x * blockDim.x;
  C += blockIdx.y * blockDim.y * N;

  for (int tile_idx = 0; tile_idx < K / kTileK; tile_idx++) {
    // copy into shared mem
    A_shmem[threadIdx.y * kTileK + threadIdx.x] =
        A[threadIdx.y * K + threadIdx.x];
    B_shmem[threadIdx.y * blockDim.x + threadIdx.x] =
        B[threadIdx.y * N + threadIdx.x];
    __syncthreads();
    A += kTileK;
    B += kTileK * N;
    for (int k = 0; k < kTileK; k++) {
      value += A_shmem[threadIdx.y * kTileK + k] *
               B_shmem[k * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }

  C[threadIdx.y * N + col] = alpha * value + beta * C[threadIdx.y * N + col];
}

void kernel_gemm_3_launch(GemmData &data, const unsigned int num_runs) {
  auto func = [&](dim3 block_size) {
    dim3 grid_size(std::ceil(data.dim_n / block_size.x),
                   std::ceil(data.dim_m / block_size.y));

    const int kSharedMemSize =
        (block_size.x * kTileK + block_size.y * kTileK) * sizeof(half);

    KernelProfiler profiler;
    for (int i = 0; i < num_runs; i++) {
      profiler.start();
      kernel_gemm_3<<<grid_size, block_size, kSharedMemSize,
                      cudaStreamPerThread>>>(data.dev_A, data.dev_B, data.dev_C,
                                             data.dim_m, data.dim_n, data.dim_k,
                                             data.alpha, data.beta);
      profiler.stop();
      // moved correctness check here because results accumulate on C
      if (i == 0) {
        data.check_out();
      }
    }
    CUDA_CHECK(cudaPeekAtLastError());

    std::cout << "kernel 3 (shmem) (blockDim(" << block_size.x << ","
              << block_size.y << ")) GFLOPS for size (" << data.dim_m << "x"
              << data.dim_n << "x" << data.dim_k << "): "
              << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
              << std::endl;
  };
  func(dim3(32, 32));
}
