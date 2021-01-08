cmake_minimum_required(VERSION 2.8)

project(test_tino CXX)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native")

option(USE_OPENMP "Use OpenMP" OFF)
option(USE_OPENBLAS "Use OpenBlas" OFF)
option(USE_INTEL_TBB "Use Intel TBB" OFF)

set(LIBRARY_INCLUDE_PATH "")

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(TINO_USE_OPENMP)
  endif()
endif()

if(USE_OPENBLAS)
  # set(BLAS_DIR "/opt/OpenBLAS/latest/openmp")
  find_package(BLAS)
  if(BLAS_FOUND)
    if(EXISTS "/usr/include/x86_64-linux-gnu/openblas-openmp/cblas.h")
      set(LIBRARY_INCLUDE_PATH
          "${LIBRARY_INCLUDE_PATH} -I /usr/include/x86_64-linux-gnu/openblas-openmp/cblas.h"
      )
    elseif(EXISTS "/usr/include/cblas.h")
      set(LIBRARY_INCLUDE_PATH
          "${LIBRARY_INCLUDE_PATH} -I /usr/include/cblas.h")
    elseif(EXISTS "${OPENBLAS_LIBRARY_ROOT}/include/cblas.h")
      set(LIBRARY_INCLUDE_PATH
          "${LIBRARY_INCLUDE_PATH} -I ${OPENBLAS_LIBRARY_ROOT}/include/cblas.h")
    else()
      message(FATAL_ERROR "OpenBLAS not found in your system.")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LIBRARY_INCLUDE_PATH} -l openblas")
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${LIBRARY_INCLUDE_PATH} -l openblas")
    add_compile_definitions(TINO_USE_OPENBLAS)

  elseif(EXISTS "${OPENBLAS_LIBRARY_ROOT}/lib/libopenblas.so"
         AND EXISTS "${OPENBLAS_LIBRARY_ROOT}/include/cblas.h")
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -I ${OPENBLAS_LIBRARY_ROOT}/include -L ${OPENBLAS_LIBRARY_ROOT}/lib/ -l openblas"
    )
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -I ${OPENBLAS_LIBRARY_ROOT}/include -L ${OPENBLAS_LIBRARY_ROOT}/lib/ -l openblas"
    )
    add_compile_definitions(TINO_USE_OPENBLAS)
    message(STATUS "Found OpenBLAS at ${OPENBLAS_LIBRARY_ROOT}")
  else()
    message(FATAL_ERROR "OpenBLAS not found in your system.")
  endif()
endif()

message(STATUS "${CMAKE_CXX_FLAGS}")

add_executable(test_naive.out test_naive.cpp)
target_include_directories(test_naive.out PRIVATE ../)

add_executable(test_openblas.out test_openblas.cpp)
target_include_directories(test_openblas.out PRIVATE ../)
