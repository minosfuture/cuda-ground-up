# cuda-ground-up
A hands-on repository for learning and experimenting with GPU programming, CUDA kernel optimization, and model optimization techniques, built from the ground up with a teaching-focused approach.

## Steps

```bash
mkdir build
cd build
cmake -DCUDAToolkit_ROOT=/usr/local/cuda ..
make
# run each implementation for 10 iterations with data size of 2048x2048x32
./runner 10 2048 2048 32
# run nsight compute profiling on the implementations (for one iteration)
ncu -f --set full --call-stack -o vector_add ./runner 1 2048 2048 32
# run nsight system profiling
nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=100000 --gpu-metrics-set=tu10x-gfxt ./runner 0 10 2048 2048 32
```
