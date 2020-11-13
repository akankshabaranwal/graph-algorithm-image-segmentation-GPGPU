# Template for OpenCV CUDA-Enabled Project

This branch will serve as a template for sub-projects using OpenCV (with CUDA support).

## Building and installing CUDA-enabled OpenCV on Ubuntu Linux

- Make sure that CUDA 11.1 is installed before starting. There are instructions for installing CUDA on Linux [here][2].
  Also make sure the necessary build tools are installed:

```
$ sudo apt-get install build-essential cmake ninja git
```

[2]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

- Create a workspace where you will be building OpenCV and clone the `opencv` and `opencv_contrib` repos.

```
$ mkdir -p ~/opencv_workspace && cd ~/opencv_workspace
$ git clone --depth 1 --branch 4.5.0 https://github.com/opencv/opencv.git
$ git clone --depth 1 --branch 4.5.0 https://github.com/opencv/opencv_contrib.git
```

- Perform the following commands to build and install a Debug build of OpenCV to `/usr/local/opencv/debug`.

```
$ mkdir build && cd build

$ cmake ../opencv -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
                  -DBUILD_LIST=core,imgproc,highgui,cudev,cudafilters   \
                  -DBUILD_SHARED_LIBS=ON                                \
                  -DWITH_CUDA=ON                                        \
                  -DCUDA_GENERATION=Auto                                \
                  -DCMAKE_BUILD_TYPE=Debug                              \
                  -DCMAKE_INSTALL_PREFIX="/usr/local/opencv/debug"      \
                  -GNinja

$ ninja
$ sudo ninja install
```

- Perform the following commands to build and install a Release build of OpenCV to `/usr/local/opencv/release`.

```
$ mkdir build && cd build

$ cmake ../opencv -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
                  -DBUILD_LIST=core,imgproc,highgui,cudev,cudafilters   \
                  -DBUILD_SHARED_LIBS=ON                                \
                  -DWITH_CUDA=ON                                        \
                  -DCUDA_GENERATION=Auto                                \
                  -DCMAKE_BUILD_TYPE=Release                            \
                  -DCMAKE_INSTALL_PREFIX="/usr/local/opencv/release"    \
                  -GNinja

$ ninja
$ sudo ninja install
```

The OpenCV library requires that you use a Release build when compiling your project with Release. A Release build is
important for benchmarking since it generally gives you the best performance. See below for some additional notes
regarding the CMake configuration.

## Building and installing CUDA-enabled OpenCV on Windows

If you are using Visual Studio 2019 (with Visual C++ installed), the installation contains a special shortcut for a
command-line interpreter with the `PATH` environment variable containing `cmake` and other build tools such as the
compiler and the linker. The shortcut for the "x64 Native Tools Command Prompt for VS 2019" can be found in
`C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2019\Visual Studio Tools\VC\`. When opening the
prompt, make sure to **run it as Administrator**. Also check if [Git for Windows][2] is installed and `git` is available
from your `PATH`.

[2]: https://gitforwindows.org/

- Inside the build-tools command prompt, create a workspace where you will be building OpenCV and clone the `opencv` and
  `opencv_contrib` repos. It is recommend to also add this workspace folder to your anti-virus exclusions to speed up
  the builds.

```
cd %USERPROFILE% & mkdir OpenCV_Workspace & cd OpenCV_Workspace
git clone --depth 1 --branch 4.5.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.5.0 https://github.com/opencv/opencv_contrib.git
```

- Perform the following commands to build and install a Debug build of OpenCV to `C:/Program Files/OpenCV/Debug/`.

```
mkdir build & cd build

cmake ../opencv -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules   ^
                -DBUILD_LIST=core,imgproc,highgui,cudev,cudafilters     ^
                -DBUILD_SHARED_LIBS=ON                                  ^
                -DWITH_CUDA=ON                                          ^
                -DCUDA_GENERATION=Auto                                  ^
                -DCMAKE_BUILD_TYPE=Debug                                ^
                -DCMAKE_INSTALL_PREFIX="C:/Program Files/OpenCV/Debug/" ^
                -GNinja

ninja
ninja install
```

- Perform the following commands to build and install a Release build of OpenCV to `C:/Program Files/OpenCV/Release/`.

```
cd .. & rd /s /q build & mkdir build & cd build

cmake ../opencv -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules     ^
                -DBUILD_LIST=core,imgproc,highgui,cudev,cudafilters       ^
                -DBUILD_SHARED_LIBS=ON                                    ^
                -DWITH_CUDA=ON                                            ^
                -DCUDA_GENERATION=Auto                                    ^
                -DCMAKE_BUILD_TYPE=Release                                ^
                -DCMAKE_INSTALL_PREFIX="C:/Program Files/OpenCV/Release/" ^
                -GNinja

ninja
ninja install
```

The OpenCV library requires that you use a Release build when compiling your project with Release. A Release build is
important for benchmarking since it generally gives you the best performance. See below for some additional notes
regarding the CMake configuration.

## Notes regarding the CMake configuration when building OpenCV

 - `-DBUILD_LIST` defines the list of modules to be built. OpenCV has a plethora of modules but in the project template
   we only require `cudafilters` which depends on `cudev`, and `highgui` to display the results.

 - `-DCMAKE_INSTALL_PREFIX` defines where the OpenCV libraries will be installed. You can decide to install OpenCV in a
   different path than the one we provide but you will need to set `-DFELZ_OPENCV_PATH` when configuring the project.
   (See `cmake/OpenCV.cmake` to understand how this works.)
