cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
project(felzenszwalb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)

add_executable(felz
	"${CMAKE_SOURCE_DIR}/src/RecMST.cu")

set_property(TARGET felz PROPERTY CUDA_ARCHITECTURES 61)


add_custom_command(
        TARGET felz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/data/
                ${CMAKE_BINARY_DIR}/data/
)
