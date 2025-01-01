#pragma once

#include "utils.h"

#include <bits/types/struct_timeval.h>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <sys/time.h>

struct GemmData;
bool kernel_verify_launch(GemmData &data);

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
  half *dev_C_ref;

  void gen_host(bool prep_host_gemm = false) {
    A = (half *)malloc(size_A);
    B = (half *)malloc(size_B);
    C = (half *)malloc(size_C);
    C_ref = (half *)malloc(size_C);

    struct timeval time;
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    if (A == nullptr || B == nullptr || C == nullptr) {
      std::cerr << "Failed to allocate host vectors!" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "initialize host input vectors..." << std::endl;
    auto rand_gen = [](float lb = -0.1, float ub = 0.1) {
      return (half)(lb + (rand() / (float)RAND_MAX * (ub - lb)));
    };
    for (int i = 0; i < num_A; ++i) {
      A[i] = rand_gen();
    }
    for (int i = 0; i < num_B; ++i) {
      B[i] = rand_gen();
    }
    for (int i = 0; i < num_C; ++i) {
      C[i] = rand_gen();
    }
    if (prep_host_gemm) {
      host_gemm(C_ref);
    }
  }

  void gen_dev() {
    CUDA_CHECK(cudaMalloc(&dev_A, size_A));
    CUDA_CHECK(cudaMalloc(&dev_B, size_B));
    CUDA_CHECK(cudaMalloc(&dev_C, size_C));
    CUDA_CHECK(cudaMalloc(&dev_C_ref, size_C));
    std::cout << "copy inputs..." << std::endl;
    CUDA_CHECK(cudaMemcpy(dev_A, A, size_A, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_B, B, size_B, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_C, C, size_C, cudaMemcpyDefault));
  }

  void reset_c() {
    CUDA_CHECK(cudaMemcpy(dev_C, C, size_C, cudaMemcpyDefault));
  }

  bool check_out() {
    const bool kUseGpuVerify = true;
    if (kUseGpuVerify) {
      return kernel_verify_launch(*this);
    } else {
      std::cout << "check outputs..." << std::endl;
      half *kernel_output_C = (half *)malloc(size_C);
      CUDA_CHECK(cudaMemcpy(kernel_output_C, dev_C, size_C, cudaMemcpyDefault));
      bool res = check_correctness(kernel_output_C, C_ref);

      delete kernel_output_C;
      return res;
    }
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

  // to be used by standard implemention, e.g., cublas, for fast correct result
  // prepartion
  // dev_C should contain correct results
  void set_c_ref() {
    CUDA_CHECK(cudaMemcpy(dev_C_ref, dev_C, size_C, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(C_ref, dev_C, size_C, cudaMemcpyDefault));
  }

  bool check_correctness(half *C, half *C_ref, float epsilon = 0.01) {
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
