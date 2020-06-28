INCLUDE_DIRECTORIES(../../) #< #include "tracker/..."
find_package(OpenPose REQUIRED)
  find_package(GFlags) # For Caffe and OpenPose
  find_package(Glog) # For Caffe
file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
add_executable(htrack_atb ${HDR} ${SRC})
target_link_libraries(htrack_atb htracksdk ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${OpenCV_LIBS})





