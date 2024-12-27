#pragma once
#include <cuda_fp16.h>

void kernel_vector_add(const unsigned int N, const unsigned int num_runs = 10);

void kernel_vector_add_1_launch(half *dev_A, half *dev_B, half *dev_C,
                                const unsigned int N,
                                const unsigned int num_runs = 10);
void kernel_vector_add_2_launch(half *dev_A, half *dev_B, half *dev_C,
                                const unsigned int N,
                                const unsigned int num_runs = 10);
void kernel_vector_add_3_launch(half *dev_A, half *dev_B, half *dev_C,
                                const unsigned int N,
                                const unsigned int num_runs = 10);
