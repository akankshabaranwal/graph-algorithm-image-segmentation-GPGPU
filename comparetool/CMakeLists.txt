cmake_minimum_required(VERSION 3.17)


project(comparetool LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

include(../cmake/OpenCV.cmake)

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)

add_executable(comparetool "${CMAKE_SOURCE_DIR}/src/comparetool.cpp")

target_include_directories(comparetool PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(comparetool PRIVATE ${OpenCV_LIBS})

if(WIN32)
    set_property(TARGET comparetool PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_custom_command(
        TARGET comparetool POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${OpenCV_INSTALL_PATH}/lib
                ${CMAKE_BINARY_DIR}
)
