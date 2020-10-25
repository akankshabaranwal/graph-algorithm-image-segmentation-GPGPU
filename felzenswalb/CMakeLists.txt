cmake_minimum_required(VERSION 3.17)
project(felzenswalb)

set(CMAKE_CXX_STANDARD 14)

add_executable(felzenswalb main.cpp stb_image/stb_image.h stb_image/stb_image_write.h stb_image/stb_image_resize.h gaussian_filter.cpp gaussian_filter.h)