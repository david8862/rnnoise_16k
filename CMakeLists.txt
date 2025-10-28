# project
cmake_minimum_required(VERSION 3.5)
project(rnnoise_16k LANGUAGES C CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

#set(CMAKE_C_STANDARD 17)
include(GNUInstallDirs)

# ---[ RNNoise version
set(RNNOISE_TARGET_VERSION   "1.0.0"  CACHE STRING "RNNoise logical version")
set(RNNOISE_TARGET_SOVERSION "1"      CACHE STRING "RNNoise soname version")
add_definitions(-DRNNOISE_VERSION=${RNNOISE_TARGET_VERSION})

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# https://www.zhihu.com/question/27090458
# https://blog.csdn.net/m0_37870649/article/details/109139204
option(RNNOISE_BUILD_FAST "build lib & binary with gcc -Ofast optimization" OFF)
option(RNNOISE_BUILD_SHARED_LIBS "build shared or static lib" OFF)
option(RNNOISE_BUILD_TEST_COVERAGE "build lib & binary for test coverage" OFF)
option(RNNOISE_BUILD_ASAN "build lib & binary for asan analysis" OFF)
option(RNNOISE_BUILD_FOR_GPROF "build lib & binary for gprof analysis" OFF)

# --- [ Add some build optimization
if (RNNOISE_BUILD_FAST)
    message(STATUS "Build with gcc -Ofast optimization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -g")
endif()

if (RNNOISE_BUILD_TEST_COVERAGE)
    message(STATUS "Add flags in build for test coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage -fPIC")
endif()

if (RNNOISE_BUILD_ASAN)
    message(STATUS "Add flags in build for asan analysis")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address -fno-var-tracking -fno-omit-frame-pointer -fno-stack-protector")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize-recover=address -fno-var-tracking -fno-omit-frame-pointer -fno-stack-protector")
endif()

if(RNNOISE_BUILD_FOR_GPROF)
    message(STATUS "Add '-pg' in build for gprof analysis")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

set(LIBRNNOISE rnnoise)

file(GLOB_RECURSE LIBRNNOISE_SRCS    ${CMAKE_HOME_DIRECTORY}/src/*.c)

if(RNNOISE_BUILD_SHARED_LIBS)
    add_library(${LIBRNNOISE} SHARED ${LIBRNNOISE_SRCS})
else()
    add_library(${LIBRNNOISE} STATIC ${LIBRNNOISE_SRCS})
endif()
target_include_directories(${LIBRNNOISE} PUBLIC ${CMAKE_HOME_DIRECTORY}/include/)

# --- [ Build samples, including .c & .cpp
file (GLOB C_SAMPLE_SRCS   ${CMAKE_HOME_DIRECTORY}/samples/*.c)
file (GLOB CPP_SAMPLE_SRCS   ${CMAKE_HOME_DIRECTORY}/samples/*.cpp)
set(SAMPLE_SRCS ${C_SAMPLE_SRCS} ${CPP_SAMPLE_SRCS})
foreach (source ${SAMPLE_SRCS})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${CMAKE_HOME_DIRECTORY}/include/)
  target_link_libraries(${name} PUBLIC ${LIBRNNOISE} -lm)
endforeach ()

# --- [ Install

# ---[ Install options
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE)
endif()

# get header files
file(GLOB LIBRNNOISE_HDRS   ${CMAKE_HOME_DIRECTORY}/include/rnnoise.h)

set_target_properties(${LIBRNNOISE} PROPERTIES
        VERSION   ${RNNOISE_TARGET_VERSION}
        SOVERSION ${RNNOISE_TARGET_SOVERSION})
install(TARGETS ${LIBRNNOISE} EXPORT RNNoiseTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBRNNOISE_HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(RNNOISE_SAMPLES rnnoise_sample rnnvad_sample rnnvad_postprocess_sample rnnvad_postprocess_stream_sample)
install(TARGETS ${RNNOISE_SAMPLES} DESTINATION ${CMAKE_INSTALL_BINDIR}/samples)

