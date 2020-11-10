cmake_minimum_required(VERSION 3.17)

project(felzenswalb)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(felzenswalb
	main.cpp
	gaussian_filter.cpp
	gaussian_filter.h
	segment_image.cpp
	segment_image.h

	stb_image/stb_image.h
	stb_image/stb_image_write.h
	stb_image/stb_image_resize.h
)
target_link_libraries( felzenswalb ${OpenCV_LIBS} )

#target_link_libraries( felzenswalb ${OpenCV_LIBS} )
# Copy the data to the output directory
add_custom_command(
        TARGET felzenswalb POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/data/
                ${CMAKE_CURRENT_BINARY_DIR}/data/)
