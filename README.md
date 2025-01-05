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

### Example output

```log
./gemm_runner 1 2048 2048 2048
initialize host input vectors...
copy inputs...
kernel 0 (cublas) GFLOPS for size (2048x2048x2048): 38986.6
kernel 1 naive GFLOPS for size (2048x2048x2048): 61.7414
kernel 2 (coalesced gmem access) (blockDim(16,16)) GFLOPS for size (2048x2048x2048): 558.901
kernel 2 (coalesced gmem access) (blockDim(32,8)) GFLOPS for size (2048x2048x2048): 582.312
kernel 2 (coalesced gmem access) (blockDim(64,4)) GFLOPS for size (2048x2048x2048): 576.93
kernel 2 (coalesced gmem access) (blockDim(128,2)) GFLOPS for size (2048x2048x2048): 472.316
kernel 2 (coalesced gmem access) (blockDim(256,1)) GFLOPS for size (2048x2048x2048): 474.798
kernel 3 (shmem) (blockDim(32,32)) GFLOPS for size (2048x2048x2048): 848.646
kernel 4 (1D tiling) (blockDim(32,16)) GFLOPS for size (2048x2048x2048): 2934.53
kernel 5 (1D tiling for both A and B) (blockDim(32,16)) GFLOPS for size (2048x2048x2048): 6334.66
kernel 5 (1D tiling refactored) (blockDim(32,16)) GFLOPS for size (2048x2048x2048): 6980.94
kernel 6 (2D tiling) (blockDim(32,16)) GFLOPS for size (2048x2048x2048): 8301.54
```

## References

* [alexarmbr/matmul-playground](https://github.com/alexarmbr/matmul-playground)
* [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
* [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
