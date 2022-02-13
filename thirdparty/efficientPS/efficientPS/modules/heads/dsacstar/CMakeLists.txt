cmake_minimum_required(VERSION 3.13)  # CMake version check
project(dsacstar)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

set(CMAKE_PREFIX_PATH /home/gosalan/libtorch-pre-abi)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OPENCV_LIBS: ${OpenCV_LIBS}")

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES dsacstar.cpp
                 thread_rand.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(dsacstar ${SOURCE_FILES})
target_link_libraries(dsacstar ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${PYTHON_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})