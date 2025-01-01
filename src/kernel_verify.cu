#include "host_utils.h"
#include "kernel_gemm.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void kernel_verify(half *src, half *target, bool *result, int M,
                              int N, half epsilon = 0.5) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }
  half src_val = src[row * N + col];
  half target_val = target[row * N + col];
  half gap = target_val - src_val;
  if ((gap > epsilon || gap < -epsilon) && *result) {
    printf("(%d,%d) mismatch %f vs. %f, diff %f\n", row, col, (float)src_val,
           (float)target_val, (float)gap);
    *result = false;
  }
}

bool kernel_verify_launch(GemmData &data) {

  dim3 block_size(32, 32);
  dim3 grid_size(std::ceil(data.dim_n / block_size.x),
                 std::ceil(data.dim_m / block_size.y));
  bool result = true;
  bool *dev_result = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_result, sizeof(result)));
  CUDA_CHECK(
      cudaMemcpy(dev_result, &result, sizeof(result), cudaMemcpyDefault));

  kernel_verify<<<grid_size, block_size>>>(data.dev_C, data.dev_C_ref,
                                           dev_result, data.dim_m, data.dim_n);

  CUDA_CHECK(
      cudaMemcpy(&result, dev_result, sizeof(result), cudaMemcpyDefault));
  return result;
}
