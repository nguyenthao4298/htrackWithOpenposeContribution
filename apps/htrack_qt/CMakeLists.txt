#--- This project needs this extra library
include(../../cmake/ConfigureQGLViewer.cmake)

INCLUDE_DIRECTORIES(../../) #< #include "tracker/..."
qt5_wrap_cpp(MOCS_HDRS Main_window.h OpenGL_viewer.h)
find_package(OpenPose REQUIRED)
  find_package(GFlags) # For Caffe and OpenPose
  find_package(Glog) # For Caffe
file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
add_executable(htrack_qt ${HDR} ${SRC} ${MOCS_HDRS})
target_link_libraries(htrack_qt htracksdk ${LIBRARIES} ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${OpenCV_LIBS})

