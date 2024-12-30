#include "kernel_gemm.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_fp16.h>

void kernel_gemm_1_launch(GemmData &data, const unsigned int num_runs);
void kernel_gemm_2_launch(GemmData &data, const unsigned int num_runs);
void kernel_gemm_3_launch(GemmData &data, const unsigned int num_runs);
void kernel_gemm_4_launch(GemmData &data, const unsigned int num_runs);

void kernel_gemm(const unsigned int M, const unsigned int N,
                 const unsigned int K, half alpha, half beta,
                 const unsigned int num_runs) {
  GemmData data(M, N, K, alpha, beta);
  data.gen_host();
  data.gen_dev();

  kernel_gemm_1_launch(data, num_runs);
  data.reset_c();
  kernel_gemm_2_launch(data, num_runs);
  data.reset_c();
  kernel_gemm_3_launch(data, num_runs);
  data.reset_c();
  kernel_gemm_4_launch(data, num_runs);
  data.reset_c();
}
