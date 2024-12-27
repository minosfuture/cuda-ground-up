#include "utils.h"
#include <cuda.h>
#include <cuda_fp16.h>

constexpr int kBlockSize = 512;

__global__ void kernel_vector_add_1(half *A, half *B, half *C,
                                    const unsigned int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  C[idx] = A[idx] + B[idx];
}

void kernel_vector_add_1_launch(half *dev_A, half *dev_B, half *dev_C,
                                const unsigned int N,
                                const unsigned int num_runs = 10) {
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

  std::cout << __FUNCTION__ << " GFLOPS for size (" << N
            << "): " << profiler.log_vec_add_stats(N) << std::endl;
}
