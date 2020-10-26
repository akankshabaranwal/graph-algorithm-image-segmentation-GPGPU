cmake_minimum_required(VERSION 3.17)

project(felzenswalb)

set(CMAKE_CXX_STANDARD 14)

add_executable(felz
	main.cpp
	gaussian_filter.cpp
	gaussian_filter.h

	stb_image/stb_image.h
	stb_image/stb_image_write.h
	stb_image/stb_image_resize.h
)

# Copy the data to the output directory
add_custom_command(
        TARGET felz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/data/
                ${CMAKE_CURRENT_BINARY_DIR}/data/)
