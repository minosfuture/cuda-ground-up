#include "kernel_gemm.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

constexpr int kTileK = 32;
constexpr int kTileYDim = 8;

__global__ void kernel_gemm_4(half *A, half *B, half *C, int M, int N, int K,
                              half alpha, half beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y * kTileYDim;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  extern __shared__ half shmem[];
  half *A_shmem = shmem;
  half *B_shmem = &shmem[blockDim.y * kTileYDim * kTileK];

  A += blockIdx.y * blockDim.y * kTileYDim * K;
  B += blockIdx.x * blockDim.x;
  C += blockIdx.y * blockDim.y * kTileYDim * N + blockIdx.x * blockDim.x;

  half thread_tile_res[kTileYDim] = {0.0};

  const int start_y_idx = threadIdx.y * kTileYDim;

  for (int tile_idx = 0; tile_idx < K / kTileK; tile_idx++) {
    // copy into shared mem
    for (int tile_k_idx = 0; tile_k_idx < kTileK / blockDim.x; tile_k_idx++) {
      for (int tile_y_idx = 0; tile_y_idx < kTileYDim; tile_y_idx++) {
        A_shmem[(start_y_idx + tile_y_idx) * kTileK + tile_k_idx * blockDim.x +
                threadIdx.x] = A[(start_y_idx + tile_y_idx) * K +
                                 tile_k_idx * blockDim.x + threadIdx.x];
      }
    }
    for (int tile_k_idx = 0; tile_k_idx < kTileK / blockDim.y; tile_k_idx++) {
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //  printf(
      //      "thread(%d,%d),block(%d,%d),tile_k_idx(%d),B_shmem_idx(%d),B_idx(%"
      //      "d) %f\n",
      //      threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tile_k_idx,
      //      (threadIdx.y + tile_k_idx * blockDim.y) * blockDim.x +
      //      threadIdx.x, (threadIdx.y + tile_k_idx * blockDim.y) * N +
      //      threadIdx.x, (float)
      //          B[(threadIdx.y + tile_k_idx * blockDim.y) * N + threadIdx.x]);
      //}
      B_shmem[(threadIdx.y + tile_k_idx * blockDim.y) * blockDim.x +
              threadIdx.x] =
          B[(threadIdx.y + tile_k_idx * blockDim.y) * N + threadIdx.x];
    }
    __syncthreads();
    A += kTileK;
    B += kTileK * N;
    for (int k = 0; k < kTileK; k++) {
      half B_reg = B_shmem[k * blockDim.x + threadIdx.x];
      for (int tile_y_idx = 0; tile_y_idx < kTileYDim; tile_y_idx++) {
        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 &&
        //    blockIdx.y == 0) {
        //  printf("thread(%d,%d),block(%d,%d),k(%d),tile_idx(%d) %f * %f\n",
        //         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, k,
        //         tile_idx, (float)A_shmem[(start_y_idx + tile_y_idx) * kTileK
        //         + k], (float)B_reg);
        //}
        thread_tile_res[tile_y_idx] +=
            A_shmem[(start_y_idx + tile_y_idx) * kTileK + k] * B_reg;
      }
    }
    __syncthreads();
  }

  for (int tile_y_idx = 0; tile_y_idx < kTileYDim; tile_y_idx++) {
    half &C_elem = C[(start_y_idx + tile_y_idx) * N + threadIdx.x];
    C_elem = alpha * thread_tile_res[tile_y_idx] + beta * C_elem;
  }
}

void kernel_gemm_4_launch(GemmData &data, const unsigned int num_runs) {
  auto func = [&](dim3 block_size) {
    dim3 grid_size(std::ceil(data.dim_n / block_size.x),
                   std::ceil(data.dim_m / block_size.y / kTileYDim));

    const int kSharedMemSize =
        (block_size.x * kTileK + block_size.y * kTileYDim * kTileK) *
        sizeof(half);

    KernelProfiler profiler;
    for (int i = 0; i < num_runs; i++) {
      profiler.start();
      kernel_gemm_4<<<grid_size, block_size, kSharedMemSize,
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

    std::cout << "kernel 4 (1D tiling) (blockDim(" << block_size.x << ","
              << block_size.y << ")) GFLOPS for size (" << data.dim_m << "x"
              << data.dim_n << "x" << data.dim_k << "): "
              << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
              << std::endl;
  };
  func(dim3(32, 16));
}