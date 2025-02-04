#--- Choose a sensor

#SET(SENSOR_DEPTHSENSEGRABBER 0 CACHE BOOL "Use SoftKinetic with DepthSenseGrabber")
#SET(SENSOR_SOFTKIN 0 CACHE BOOL "Use SoftKinetic with SoftKin")
#SET(SENSOR_REALSENSE 0 CACHE BOOL "Use Intel RealSense")
#SET(SENSOR_LIBREALSENSE 0 CACHE BOOL "Use Intel LibRealSense")
#SET(SENSOR_OPENNI 0 CACHE BOOL "Use PrimeSense Carmine or ASUS Xtion with OpenNI")
#SET(SENSOR_KINECT 0 CACHE BOOL "Use Kinect")

SET(SENSOR "librealsense" CACHE STRING "Define your own sensor while configuring CMAKE")
SET(SENSOR_LIST depthsensegrabber libfreenect librealsense openni realsense softkin)
SET_PROPERTY(CACHE SENSOR PROPERTY STRINGS ${SENSOR_LIST})

if (";${SENSOR_LIST};" MATCHES ";${SENSOR};")
    message(STATUS "Sensor is in Sensor List")
else (";${SENSOR_LIST};" MATCHES ";${SENSOR};")
    message(STATUS "${SENSOR} is not a valid option (${SENSOR_LIST}). Setting the SENSOR value to librealsense")
    SET(SENSOR librealsense)
endif (";${SENSOR_LIST};" MATCHES ";${SENSOR};")
message(STATUS "SENSOR ${SENSOR}")

SET(SENSOR_COUNT 0)

#macro(SENSOR_RESET SENSOR_1 SENSOR_2)
#    UNSET(SENSOR_DEPTHSENSEGRABBER CACHE)
#    UNSET(SENSOR_SOFTKIN CACHE)
#    UNSET(SENSOR_REALSENSE CACHE)
#    UNSET(SENSOR_LIBREALSENSE CACHE)
#    UNSET(SENSOR_OPENNI CACHE)
#    message(FATAL_ERROR "Sensor conflict: both ${SENSOR_1} and ${SENSOR_2} are set to enabled. Resetting cache, please run cmake and choose a sensor again.")
#endmacro(SENSOR_RESET)

if (SENSOR STREQUAL "depthsensegrabber")
    set (SENSOR_TEST DEPTHSENSEGRABBER)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "depthsensegrabber")

if (SENSOR STREQUAL "libfreenect")
    set (SENSOR_TEST LIBFREENECT)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "libfreenect")

if (SENSOR STREQUAL "softkin")
    set (SENSOR_TEST SOFTKIN)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "softkin")

if (SENSOR STREQUAL "realsense")
    set (SENSOR_TEST REALSENSE)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "realsense")

if (SENSOR STREQUAL "librealsense")
    set (SENSOR_TEST LIBREALSENSE)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "librealsense")

if (SENSOR STREQUAL "openni")
    set (SENSOR_TEST OPENNI)
    if (SENSOR_COUNT EQUAL 0)
        set(SENSOR_COUNT 1)
        set(SENSOR_TYPE ${SENSOR_TEST})
        message(STATUS "Enabling ${SENSOR_TEST}")
    else (SENSOR_COUNT EQUAL 0)
        SENSOR_RESET(${SENSOR_TYPE} ${SENSOR_TEST})
    endif (SENSOR_COUNT EQUAL 0)
endif (SENSOR STREQUAL "openni")


if (NOT SENSOR_COUNT)
    message(FATAL_ERROR "Please choose a sensor, example: cmake -DSENSOR=librealsense (valid options: ${SENSOR_LIST}), or use cmake-gui. realsense works on Windows with RealSense SDK while librealsense works on Linux with librealsense SDK")
endif (NOT SENSOR_COUNT)

set(DEFINITION_SENSOR "-D${SENSOR_TYPE}")
