cmake_minimum_required(VERSION 3.17)
project(sobel)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(sobel main.cpp)

# linking
target_link_libraries(sobel ${OpenCV_LIBS})