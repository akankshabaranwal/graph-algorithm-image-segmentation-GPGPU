cmake_minimum_required(VERSION 3.17)
project(structured_forest)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(structured_forest main.cpp)

# linking
target_link_libraries(structured_forest ${OpenCV_LIBS})