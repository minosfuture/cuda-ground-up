#include "kernel_vector_add.h"
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {

  int N = 1;
  int num_runs = std::stoi(argv[1]);
  for (int i = 2; i < argc; i++) {
    N *= std::stoi(argv[i]);
  }

  kernel_vector_add_1_launch(N, num_runs);
  kernel_vector_add_2_launch(N, num_runs);
  kernel_vector_add_3_launch(N, num_runs);

  return EXIT_SUCCESS;
}
