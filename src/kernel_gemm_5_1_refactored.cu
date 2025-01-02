#include "kernel_gemm.h"

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

constexpr int kBlockDimX = 32;
constexpr int kBlockDimY = 16;

// number of elements per register tile
constexpr int kRegTileDimM = 8;
constexpr int kRegTileDimK = 1;
constexpr int kRegTileDimN = 8;

// number of register tiles per warp (when using tensor core) or thread (when
// using cuda core)
constexpr int kRegTileNumDimM = 1;
constexpr int kRegTileNumDimK = 1;
constexpr int kRegTileNumDimN = 1;

constexpr int kRegTileDataDimM = kRegTileNumDimM * kRegTileDimM;
constexpr int kRegTileDataDimK = kRegTileNumDimK * kRegTileDimK;
constexpr int kRegTileDataDimN = kRegTileNumDimN * kRegTileDimN;

// number of shm tiles per shm
constexpr int kShmTileNumDimM = kBlockDimY;
constexpr int kShmTileNumDimK = 32; // looped over
constexpr int kShmTileNumDimN = kBlockDimX;

constexpr int kShmTileDataDimM = kShmTileNumDimM * kRegTileDataDimM;
constexpr int kShmTileDataDimK = kShmTileNumDimK * kRegTileDataDimK;
constexpr int kShmTileDataDimN = kShmTileNumDimN * kRegTileDataDimN;

__global__ void kernel_gemm_5_1(half *A, half *B, half *C, int M, int N, int K,
                                half alpha, half beta) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  int row = by * kBlockDimY * kRegTileDataDimM + ty * kRegTileDataDimM;
  int col = bx * kBlockDimX * kRegTileDataDimN + tx * kRegTileDataDimN;

  if (row >= M || col >= N) {
    return;
  }

  extern __shared__ half shmem[];
  half *A_shmem = shmem;
  half *B_shmem = &shmem[kBlockDimY * kRegTileDataDimM * kShmTileNumDimK];

  A += by * kBlockDimY * kRegTileDataDimM * K;
  B += bx * kBlockDimX * kRegTileDataDimN;
  C += by * kBlockDimY * kRegTileDataDimM * N;

  half thread_tile_res[kRegTileDataDimM][kRegTileDataDimN] = {0.0};
  half A_reg[kRegTileDataDimM];
  half B_reg[kRegTileDataDimN];

  const int start_y_idx = ty * kRegTileDataDimM;
  const int start_x_idx = tx * kRegTileDataDimN;

  for (int tile_idx = 0; tile_idx < K / kShmTileNumDimK; tile_idx++) {
    // copy into shared mem
    for (int tile_k_idx = 0; tile_k_idx < kShmTileNumDimK / kBlockDimX;
         tile_k_idx++) {
      for (int tile_y_idx = 0; tile_y_idx < kRegTileDataDimM; tile_y_idx++) {
        A_shmem[(start_y_idx + tile_y_idx) * kShmTileNumDimK +
                tile_k_idx * kBlockDimX + tx] =
            A[(start_y_idx + tile_y_idx) * K + tile_k_idx * kBlockDimX + tx];
      }
    }
    for (int tile_k_idx = 0; tile_k_idx < kShmTileNumDimK / kBlockDimY;
         tile_k_idx++) {
      for (int tile_x_idx = 0; tile_x_idx < kRegTileDataDimN; tile_x_idx++) {
        B_shmem[(ty + tile_k_idx * kBlockDimY) * kBlockDimX * kRegTileDataDimN +
                start_x_idx + tile_x_idx] =
            B[(ty + tile_k_idx * kBlockDimY) * N + start_x_idx + tile_x_idx];
      }
    }
    __syncthreads();
    A += kShmTileNumDimK;
    B += kShmTileNumDimK * N;
    for (int k = 0; k < kShmTileNumDimK; k++) {
      for (int tile_y_idx = 0; tile_y_idx < kRegTileDataDimM; tile_y_idx++) {
        A_reg[tile_y_idx] =
            A_shmem[(ty * kRegTileDataDimM + tile_y_idx) * kShmTileNumDimK + k];
      }
      for (int tile_x_idx = 0; tile_x_idx < kRegTileDataDimN; tile_x_idx++) {
        B_reg[tile_x_idx] =
            B_shmem[(k * kBlockDimX + tx) * kRegTileDataDimN + tile_x_idx];
      }
      for (int tile_y_idx = 0; tile_y_idx < kRegTileDataDimM; tile_y_idx++) {
        for (int tile_x_idx = 0; tile_x_idx < kRegTileDataDimN; tile_x_idx++) {
          thread_tile_res[tile_y_idx][tile_x_idx] +=
              A_reg[tile_y_idx] * B_reg[tile_x_idx];
        }
      }
    }
    __syncthreads();
  }

  for (int tile_y_idx = 0; tile_y_idx < kRegTileDataDimM; tile_y_idx++) {
    for (int tile_x_idx = 0; tile_x_idx < kRegTileDataDimN; tile_x_idx++) {
      half &C_elem = C[(start_y_idx + tile_y_idx) * N + col + tile_x_idx];
      C_elem = alpha * thread_tile_res[tile_y_idx][tile_x_idx] + beta * C_elem;
    }
  }
}

void kernel_gemm_5_1_launch(GemmData &data, const unsigned int num_runs) {
  auto func = [&](dim3 block_size) {
    dim3 grid_size(std::ceil(data.dim_n / block_size.x / kRegTileDataDimN),
                   std::ceil(data.dim_m / block_size.y / kRegTileDataDimM));

    const int kSharedMemSize =
        (block_size.x * kRegTileDataDimN * kShmTileNumDimK +
         block_size.y * kRegTileDataDimM * kShmTileNumDimK) *
        sizeof(half);

    auto kernel_func = [&]() {
      kernel_gemm_5_1<<<grid_size, block_size, kSharedMemSize,
                        cudaStreamPerThread>>>(
          data.dev_A, data.dev_B, data.dev_C, data.dim_m, data.dim_n,
          data.dim_k, data.alpha, data.beta);
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

    std::cout << "kernel 5 (2D tiling refactored) (blockDim(" << block_size.x
              << "," << block_size.y << ")) GFLOPS for size (" << data.dim_m
              << "x" << data.dim_n << "x" << data.dim_k << "): "
              << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
              << std::endl;
  };
  func(dim3(32, 16));
}
