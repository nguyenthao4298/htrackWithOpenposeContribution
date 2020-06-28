# https://github.com/IntelRealSense/librealsense/issues/62

cmake_minimum_required(VERSION 2.8)
get_filename_component(PROJECT ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${PROJECT})
add_compile_options(-std=c++11 -fPIC -lusb-1.0 -lpthread )


# libusb-1.0
find_package(Threads REQUIRED)
find_package(PkgConfig)
pkg_check_modules(PC_LIBUSB REQUIRED libusb-1.0)
include_directories( ${LIBUSB_1_INCLUDE_DIRS})


set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
include(../../cmake/ConfigureOpenCV.cmake)
include(../../cmake/ConfigureOpenGL.cmake)

find_package(OpenCV 3.3.1 EXACT PATHS ~/Programs/opencv-3.3.1/build QUIET)

# GLUT
#find_package(GLUT REQUIRED)
#include_directories(${GLUT_INCLUDE_DIRS})
#LIST(APPEND LIBRARIES ${GLUT_LIBRARY})

# GLFW
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})
LIST(APPEND LIBRARIES ${GLFW_LIBRARIES})



# libfreenect
set(freenect2_DIR ~/Programs/freenect2/lib/cmake/freenect2)
find_package(freenect2 REQUIRED)
INCLUDE_DIRECTORIES(${freenect2_INCLUDE_DIR})
list(APPEND LIBRARIES ${freenect2_LIBRARIES})


aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
