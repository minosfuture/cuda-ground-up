#include "utils.h"
#include <cuda.h>
#include <cuda_fp16.h>

constexpr int kBlockSize = 512;
constexpr int kTileCount = 32; // number of elements processed by each thread
constexpr int kPerBlockDataSize = kTileCount * kBlockSize;

__global__ void kernel_vector_add_2(half *A, half *B, half *C,
                                    const unsigned int N) {
  const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < kTileCount; i++) {
    int idx = start_idx + i * kBlockSize;
    C[idx] = A[idx] + B[idx];
  }
}

void kernel_vector_add_2_launch(const unsigned int N,
                                const unsigned int num_runs = 10) {
  half *dev_A, *dev_B, *dev_C;
  CUDA_CHECK(cudaMalloc(&dev_A, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_B, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_C, N * sizeof(half)));

  dim3 grid_dim(N / kPerBlockDataSize);
  dim3 block_dim(kBlockSize);

  KernelProfiler profiler;
  for (int i = 0; i < num_runs; i++) {
    profiler.start();
    kernel_vector_add_2<<<grid_dim, block_dim, 0, cudaStreamPerThread>>>(
        dev_A, dev_B, dev_C, N);
    profiler.stop();
  }
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << __FUNCTION__ << " GFLOPS for size (" << N
            << "): " << profiler.logVectorAddKernelStats(N) << std::endl;
}
