# Make sure you have compiled and installed OpenCV in these locations
if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(FELZ_OPENCV_PATH "C:/Program Files/OpenCV/Debug")
    else()
        set(FELZ_OPENCV_PATH "C:/Program Files/OpenCV/Release")
    endif()
else()
     set(FELZ_OPENCV_PATH "/usr/local/opencv/debug") # TODO: change back
    #if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    set(FELZ_OPENCV_PATH "/usr/local/opencv/debug")
    #else()
    #    set(FELZ_OPENCV_PATH "/usr/local/opencv/release")
    #endif()
endif()

set(OpenCV_CUDA ON)
find_package(OpenCV REQUIRED core imgproc highgui cudev cudafilters
    PATHS "${FELZ_OPENCV_PATH}" NO_DEFAULT_PATH)

if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV with CUDA support was not found.")
    message(STATUS "See README.md for instructions on how to install OpenCV with CUDA for the Felzenswalb project.")
    return()
endif()
