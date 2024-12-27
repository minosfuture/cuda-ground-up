#include "kernel_vector_add.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

struct VecAddData {
  VecAddData(unsigned int N)
      : num_elements(N), size_elements(sizeof(half) * num_elements) {}
  unsigned int num_elements;
  unsigned int size_elements;
  half *A;
  half *B;
  half *C;
  half *dev_A;
  half *dev_B;
  half *dev_C;

  void gen_host() {
    A = (half *)malloc(size_elements);
    B = (half *)malloc(size_elements);
    C = (half *)malloc(size_elements);

    if (A == nullptr || B == nullptr || C == nullptr) {
      std::cerr << "Failed to allocate host vectors!" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "initialize host input vectors..." << std::endl;
    for (int i = 0; i < num_elements; ++i) {
      A[i] = rand() / (half)(float)RAND_MAX;
      B[i] = rand() / (half)(float)RAND_MAX;
    }
  }

  void gen_dev() {
    CUDA_CHECK(cudaMalloc(&dev_A, size_elements));
    CUDA_CHECK(cudaMalloc(&dev_B, size_elements));
    CUDA_CHECK(cudaMalloc(&dev_C, size_elements));
    std::cout << "copy inputs..." << std::endl;
    CUDA_CHECK(cudaMemcpy(dev_A, A, size_elements, cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(dev_B, B, size_elements, cudaMemcpyDefault));
  }

  bool check_out() {
    std::cout << "check outputs..." << std::endl;
    CUDA_CHECK(cudaMemcpy(C, dev_C, size_elements, cudaMemcpyDefault));
    for (int i = 0; i < num_elements; ++i) {
      if (std::fabs((float)A[i] + (float)B[i] - (float)C[i]) > 1e-5) {
        std::cerr << "Result verification failed at element " << i << std::endl;
        return false;
      }
    }
    return true;
  }
};

void kernel_vector_add(const unsigned int N, const unsigned int num_runs) {
  VecAddData data(N);
  data.gen_host();
  data.gen_dev();

  kernel_vector_add_1_launch(data.dev_A, data.dev_B, data.dev_C, N, num_runs);
  data.check_out();
  kernel_vector_add_2_launch(data.dev_A, data.dev_B, data.dev_C, N, num_runs);
  data.check_out();
  kernel_vector_add_3_launch(data.dev_A, data.dev_B, data.dev_C, N, num_runs);
  data.check_out();
}
