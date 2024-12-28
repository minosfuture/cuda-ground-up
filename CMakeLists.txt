cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# Try to find CUDAToolkit (modern approach)
find_package(CUDAToolkit QUIET)

if (NOT CUDAToolkit_FOUND)
    # Fallback to FindCUDA (legacy)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARIES})
endif()

# turing arch, tesla t4
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --std=c++17")

file(GLOB SOURCES "src/kernel*.cu" "src/kernel*.cc")
add_library(kernels ${SOURCES})
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>) # print register use
#target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=cg>) # turn off L1 cache
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
#target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
#target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
#target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)

#add_library(cublas_launch src/cublas_benchmark.cu)
#target_link_libraries(cublas_launch CUDA::cublas)

add_executable(vecadd_runner src/vecadd_main.cu)
add_executable(gemm_runner src/gemm_main.cu)
target_link_libraries(vecadd_runner PUBLIC kernels) # cublas_launch)
target_link_libraries(gemm_runner PUBLIC kernels) # cublas_launch)
