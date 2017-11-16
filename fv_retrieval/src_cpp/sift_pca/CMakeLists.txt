cmake_minimum_required(VERSION 2.8)
project(pca_pro)

set(CMAKE_CXX_FLAGS "-std=c++11 -fno-operator-names")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OpenCV_DIR /home/yuanyong/lib/opencv/share/OpenCV)
#set(OpenCV_DIR /home/yuanyong/lib/opencv3-no-gpu/share/OpenCV)
find_package(OpenCV REQUIRED)

set(USE_GPU True)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB pca_pro
    "*.hpp"
    "*.cpp"
    "*.cc"
    "*.h"
)

add_executable(pca_train ${pca_pro})
target_link_libraries (pca_train ${OpenCV_LIBS})
