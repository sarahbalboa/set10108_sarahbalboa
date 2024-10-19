project(set10108-cw1Cuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
include_directories(../contrib)
 
add_executable(main-cuda-unoptimised main-cuda-unoptimised.cu)
set_property(TARGET main-cuda-unoptimised PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(main-cuda-optimised1 main-cuda-optimised1.cu)
set_property(TARGET main-cuda-optimised1 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(main-cuda-optimised2 main-cuda-optimised2.cu)
set_property(TARGET main-cuda-optimised2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Copy the dataset directory to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/dataset DESTINATION ${CMAKE_BINARY_DIR})