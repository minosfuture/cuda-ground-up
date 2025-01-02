# cuda-ground-up

A hands-on repository for learning and experimenting with GPU programming, CUDA
kernel optimization, and model optimization techniques, built from the ground up
with a teaching-focused approach.

## Steps

Example for Turing T4 GPU:

```bash
mkdir build
cd build
cmake -DCUDAToolkit_ROOT=/usr/local/cuda ..
make

# run each implementation for 10 iterations with data size of 2048x2048x32
./vecadd_runner 10 2048 2048 32
# run gemm, each implementation for 10 iterations
./gemm_runner 10 4096 4096 4096

# run nsight compute profiling on the implementations (for one iteration)
ncu -f --set full --call-stack -o gemm ./gemm_runner 1 4096 4096 4096
# run nsight system profiling
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=100000 --gpu-metrics-set=tu10x-gfxt ./gemm_runner 1 4096 4096 4096
```

## References

* [alexarmbr/matmul-playground](https://github.com/alexarmbr/matmul-playground)
* [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
* [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
