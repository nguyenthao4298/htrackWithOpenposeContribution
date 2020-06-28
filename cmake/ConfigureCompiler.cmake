#--- Build type
# set(CMAKE_BUILD_TYPE "Release") #<<< FORCE SET
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")

if(APPLE)
    message(STATUS "Compiler(Apple): clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
    set(CMAKE_CC_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1") # max number of compiler errors 
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall") # enable warnings
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # only for clangomp
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -mno-avx -DNDEBUG")
elseif(UNIX)
    message(STATUS "Compiler(Linux): g++")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -DNDEBUG")
    #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto -fwhole-program")

    #The following are needed if you are using librealsense
    #add_compile_options(-std=c++11 -fPIC -lusb-1.0 -lpthread )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lusb-1.0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")
elseif(WIN32)
    message(STATUS "Compiler(Windows): Visual Studio 12 2013 Win64")
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "--> NOTE: enabled windows performance flags")
        add_definitions(-DNDEBUG)
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /Ot /fp:fast /GS- /GL")
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    endif()
endif()
