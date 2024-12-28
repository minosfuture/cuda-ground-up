#include "kernel_gemm.h"
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {

  int num_runs = std::stoi(argv[1]);
  int M = std::stoi(argv[2]);
  int N = std::stoi(argv[3]);
  int K = std::stoi(argv[4]);

  kernel_gemm(M, N, K, 0.4, 0.6, num_runs);

  return EXIT_SUCCESS;
}
