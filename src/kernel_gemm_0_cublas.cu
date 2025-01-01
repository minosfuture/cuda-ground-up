#include "kernel_gemm.h"

#include <cublas_v2.h>

#define CUBLAS_CHECK(status)                                                   \
  do {                                                                         \
    cublasStatus_t error = status;                                             \
    if (error != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "cuBLAS error: " << __FILE__ << ":" << __LINE__ << " ";     \
      switch (status) {                                                        \
      case CUBLAS_STATUS_NOT_INITIALIZED:                                      \
        std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED\n";                        \
        break;                                                                 \
      case CUBLAS_STATUS_ALLOC_FAILED:                                         \
        std::cerr << "CUBLAS_STATUS_ALLOC_FAILED\n";                           \
        break;                                                                 \
      case CUBLAS_STATUS_INVALID_VALUE:                                        \
        std::cerr << "CUBLAS_STATUS_INVALID_VALUE\n";                          \
        break;                                                                 \
      case CUBLAS_STATUS_ARCH_MISMATCH:                                        \
        std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH\n";                          \
        break;                                                                 \
      case CUBLAS_STATUS_EXECUTION_FAILED:                                     \
        std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED\n";                       \
        break;                                                                 \
      case CUBLAS_STATUS_INTERNAL_ERROR:                                       \
        std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR\n";                         \
        break;                                                                 \
      default:                                                                 \
        std::cerr << "Unknown error\n";                                        \
      }                                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void kernel_gemm_0_launch(GemmData &data, const unsigned int num_runs) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // warmup
  auto kernel_func = [&]() {
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data.dim_n,
                             data.dim_m, data.dim_k, &data.alpha, data.dev_B,
                             data.dim_n, data.dev_A, data.dim_k, &data.beta,
                             data.dev_C, data.dim_n));
  };
  kernel_func();
  data.set_c_ref();

  KernelProfiler profiler;
  for (int i = 0; i < num_runs; i++) {
    profiler.start();
    kernel_func();
    profiler.stop();
  }
  CUDA_CHECK(cudaPeekAtLastError());

  std::cout << "kernel 0 (cublas) GFLOPS for size (" << data.dim_m << "x"
            << data.dim_n << "x" << data.dim_k << "): "
            << profiler.log_gemm_stats(data.dim_m, data.dim_n, data.dim_k)
            << std::endl;
}
