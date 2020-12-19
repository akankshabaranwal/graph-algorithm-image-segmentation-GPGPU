cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
project(felzenszwalb LANGUAGES CXX CUDA)

include(cmake/OpenCV.cmake)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)


add_executable(felz
	"${CMAKE_SOURCE_DIR}/src/RecMST.cu")

set_property(TARGET felz PROPERTY CUDA_ARCHITECTURES 61)

target_include_directories(felz PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(felz PRIVATE ${OpenCV_LIBS} -lcurand)

add_custom_command(
        TARGET felz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/data/
                ${CMAKE_BINARY_DIR}/data/
)
