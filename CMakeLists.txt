cmake_minimum_required(VERSION 3.2)
project(mxnet_wdsr)

ADD_DEFINITIONS(-std=c++11)

FIND_PACKAGE(OpenCV REQUIRED)

MESSAGE(STATUS "    OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(include include/mxnet-cpp)

LINK_DIRECTORIES(/usr/lib)
AUX_SOURCE_DIRECTORY(src DIR_SRCS)


add_executable(mxnet_wdsr ${DIR_SRCS})
TARGET_LINK_LIBRARIES(mxnet_wdsr ${OpenCV_LIBS} mxnet)
