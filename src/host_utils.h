#pragma once
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(status)                                                     \
  do {                                                                         \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got cuda error: " << cudaGetErrorString(error) << " at "   \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
