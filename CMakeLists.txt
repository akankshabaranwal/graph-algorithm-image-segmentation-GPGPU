cmake_minimum_required(VERSION 3.17)

project(felzenswalb LANGUAGES CXX CUDA)

include(cmake/OpenCV.cmake)

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)

add_library(felz-gpu
        "${CMAKE_SOURCE_DIR}/src/mst.cu"
        )
set_property(TARGET felz-gpu PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET felz-gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(felz-gpu PRIVATE cudadevrt)

add_executable(felz
        "${CMAKE_SOURCE_DIR}/src/Main.cpp"
        )
set_property(TARGET felz PROPERTY CUDA_ARCHITECTURES 61)
target_include_directories(felz PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(felz PRIVATE ${OpenCV_LIBS})
target_link_libraries(felz PRIVATE felz-gpu)

if(WIN32)
    set_property(TARGET felz PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_custom_command(
        TARGET felz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/data/
                ${CMAKE_BINARY_DIR}/data/
)

add_custom_command(
        TARGET felz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${OpenCV_INSTALL_PATH}/lib
                ${CMAKE_BINARY_DIR}
)
