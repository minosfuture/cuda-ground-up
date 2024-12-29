#pragma once

#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

struct GemmData {
  GemmData(unsigned int M, unsigned int N, unsigned int K, float alpha_param,
           float beta_param)
      : dim_m(M), dim_n(N), dim_k(K), alpha(alpha_param), beta(beta_param),
        num_A(dim_m * dim_k), num_B(dim_n * dim_k), num_C(dim_m * dim_n),
        size_A(num_A * sizeof(half)), size_B(num_B * sizeof(half)),
        size_C(num_C * sizeof(half)) {}
  unsigned int dim_m;
  unsigned int dim_n;
  unsigned int dim_k;
  half alpha;
  half beta;
  unsigned int num_A;
  unsigned int num_B;
  unsigned int num_C;
  unsigned int size_A;
  unsigned int size_B;
  unsigned int size_C;

  half *A;
  half *B;
  half *C;
  half *C_ref;
  half *dev_A;
  half *dev_B;
  half *dev_C;

  void gen_host(bool random_init = false) {
    A = (half *)malloc(size_A);
    B = (half *)malloc(size_B);
    C = (half *)malloc(size_C);
    C_ref = (half *)malloc(size_C);

    if (A == nullptr || B == nullptr || C == nullptr) {
      std::cerr << "Failed to allocate host vectors!" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "initialize host input vectors..." << std::endl;
    for (int i = 0; i < num_A; ++i) {
      A[i] = (half)(rand() / (float)RAND_MAX);
    }
    for (int i = 0; i < num_B; ++i) {
      B[i] = (half)(rand() / (float)RAND_MAX);
    }
    for (int i = 0; i < num_C; ++i) {
      C[i] = (half)(rand() / (float)RAND_MAX);
    }
    if (random_init) {
      host_gemm(C_ref);
    }
  }

  void gen_dev() {
    CUDA_CHECK(cudaMalloc(&dev_A, size_A));
    CUDA_CHECK(cudaMalloc(&dev_B, size_B));
    CUDA_CHECK(cudaMalloc(&dev_C, size_C));
    std::cout << "copy inputs..." << std::endl;
    CUDA_CHECK(cudaMemcpy(dev_A, A, size_A, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_B, B, size_B, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_C, C, size_C, cudaMemcpyDefault));
  }

  void reset_c() {
    CUDA_CHECK(cudaMemcpy(dev_C, C, size_C, cudaMemcpyDefault));
  }

  bool check_out() {
    std::cout << "check outputs..." << std::endl;
    half *kernel_output_C = (half *)malloc(size_C);
    CUDA_CHECK(cudaMemcpy(kernel_output_C, dev_C, size_C, cudaMemcpyDefault));
    bool res = check_correctness(kernel_output_C, C_ref);

    delete kernel_output_C;
    return res;
  }

  void host_gemm(half *C_ref) {
    for (int i = 0; i < dim_m; ++i) {
      for (int j = 0; j < dim_n; ++j) {
        // Initialize C_ref with beta * C
        float c_value = __half2float(C[i * dim_n + j]) * __half2float(beta);

        for (int k = 0; k < dim_k; ++k) {
          // Perform the dot product in float precision
          float a_value = __half2float(A[i * dim_k + k]);
          float b_value = __half2float(B[k * dim_n + j]);
          c_value += __half2float(alpha) * (a_value * b_value);
        }

        // Store the result back as half
        C_ref[i * dim_n + j] = __float2half(c_value);
      }
    }
  }

  bool check_correctness(half *C, half *C_ref, float epsilon = 0.5) {
    for (int i = 0; i < dim_m; ++i) {
      for (int j = 0; j < dim_n; ++j) {
        float diff = std::abs(__half2float(C[i * dim_n + j]) -
                              __half2float(C_ref[i * dim_n + j]));
        if (diff > epsilon) {
          std::cerr << "Mismatch at (" << i << ", " << j << "): "
                    << "GPU result = " << __half2float(C[i * dim_n + j])
                    << ", Reference result = "
                    << __half2float(C_ref[i * dim_n + j]) << ", Diff = " << diff
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }
};

void kernel_gemm(const unsigned int M, const unsigned int N,
                 const unsigned int K, half alpha, half beta,
                 const unsigned int num_runs = 10);
void kernel_gemm_1_launch(GemmData &data, const unsigned int num_runs);
void kernel_gemm_2_launch(GemmData &data, const unsigned int num_runs);
