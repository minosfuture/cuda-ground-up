#include "kernel_vector_add.h"

void kernel_vector_add(const unsigned int N, const unsigned int num_runs) {
  kernel_vector_add_1_launch(N, num_runs);
  kernel_vector_add_2_launch(N, num_runs);
  kernel_vector_add_3_launch(N, num_runs);
}
