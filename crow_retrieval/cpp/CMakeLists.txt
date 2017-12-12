cmake_minimum_required(VERSION 2.8.8)

set(Caffe_DIR /media/disk2/caffe)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY


SET( EIGEN3_INCLUDE_DIR "/home/yuanyong/eigen-eigen-f562a193118d" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#file(GLOB utils
#    "utils.h"
#    "utils.cpp"
#)

add_executable(caffeinated_application extract_feature_bk.cpp)
target_link_libraries(caffeinated_application ${Caffe_LIBRARIES})
