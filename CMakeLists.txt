cmake_minimum_required(VERSION 3.17)

project(felzenswalb LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(cmake/OpenCV.cmake)

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)

add_executable(felz
        "${CMAKE_SOURCE_DIR}/src/Main.cpp"
        "${CMAKE_SOURCE_DIR}/src/Kernel.cu"
        )

set_property(TARGET felz PROPERTY CUDA_ARCHITECTURES 61)

target_include_directories(felz PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(felz PRIVATE ${OpenCV_LIBS})

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
