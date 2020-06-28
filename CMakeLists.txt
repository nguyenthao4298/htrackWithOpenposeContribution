cmake_minimum_required(VERSION 2.8)
project(htrack)

#--- Gets rid of annoying CMake 3 warnings
if(NOT (CMAKE_MAJOR_VERSION LESS 3))
    cmake_policy(SET CMP0043 OLD)
    cmake_policy(SET CMP0020 OLD)
    cmake_policy(SET CMP0005 OLD)
endif()

#--- Where to find the FindPackageName.cmake
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#--- Where exes/libs will be deployed
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

#--- Compiler setup
include(cmake/ConfigureCompiler.cmake)

#--- Configure the libraries
include(cmake/ConfigureQt.cmake)
include(cmake/ConfigureOpenGL.cmake)
include(cmake/ConfigureGLEW.cmake)
include(cmake/ConfigureEigen.cmake)
include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureOpenNI.cmake)
include(cmake/ConfigureRealSense.cmake)
include(cmake/ConfigureRealSense.cmake)
#include(cmake/ConfigureLibRealSense.cmake)
include(cmake/ConfigureSoftKinetic.cmake)
include(cmake/ConfigureCUDA.cmake)
include(cmake/ConfigureAntTweakBar.cmake)
include(cmake/ConfigureKinect.cmake)
#--- sub-projects
add_subdirectory(util)
add_subdirectory(cudax)
add_subdirectory(tracker)

#--- Choose which sensor to use
include(cmake/ConfigureSensor.cmake)
add_definitions(${DEFINITION_SENSOR})

add_subdirectory(apps/htrack_atb)
add_subdirectory(apps/htrack_qt)
add_subdirectory(apps/colorpick)
#add_subdirectory(apps/synthgen) #<<< NOT READY



### DOWNLOAD MODELS
# Download the models if flag is set
message(STATUS "Download the models.")

# URL to the models
set(OPENPOSE_URL "http://posefs1.perception.cs.cmu.edu/OpenPose/models/")

download_model("BODY_25" ${DOWNLOAD_BODY_25_MODEL} pose/body_25/pose_iter_584000.caffemodel
  78287B57CF85FA89C03F1393D368E5B7) # Body (BODY_25)
download_model("body (COCO)" ${DOWNLOAD_BODY_COCO_MODEL} pose/coco/pose_iter_440000.caffemodel
  5156d31f670511fce9b4e28b403f2939) # Body (COCO)
download_model("body (MPI)" ${DOWNLOAD_BODY_MPI_MODEL} pose/mpi/pose_iter_160000.caffemodel
  2ca0990c7562bd7ae03f3f54afa96e00) # Body (MPI)
download_model("face" ${DOWNLOAD_FACE_MODEL} face/pose_iter_116000.caffemodel
  e747180d728fa4e4418c465828384333) # Face
download_model("hand" ${DOWNLOAD_HAND_MODEL} hand/pose_iter_102000.caffemodel
  a82cfc3fea7c62f159e11bd3674c1531) # Hand

message(STATUS "Models Downloaded.")

