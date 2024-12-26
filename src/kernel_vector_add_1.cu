#include "utils.h"
#include <cuda.h>
#include <cuda_fp16.h>

constexpr int kBlockSize = 512;

__global__ void kernel_vector_add_1(half *A, half *B, half *C,
                                    const unsigned int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

void kernel_vector_add_1_launch(const unsigned int N,
                                const unsigned int num_runs = 10) {
  half *dev_A, *dev_B, *dev_C;
  CUDA_CHECK(cudaMalloc(&dev_A, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_B, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_C, N * sizeof(half)));

  dim3 grid_dim(N / kBlockSize);
  dim3 block_dim(kBlockSize);

  KernelProfiler profiler;
  for (int i = 0; i < num_runs; i++) {
    profiler.start();
    kernel_vector_add_1<<<grid_dim, block_dim, 0, cudaStreamPerThread>>>(
        dev_A, dev_B, dev_C, N);
    profiler.stop();
  }
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << "kernel 1 (naive impl) GFLOPS for size (" << N
            << "): " << profiler.logVectorAddKernelStats(N) << std::endl;
}
