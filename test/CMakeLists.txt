cmake_minimum_required(VERSION 2.8)

project(test_tino CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}../cmake/modules")

option(USE_OPENMP "Use OpenMP" OFF)
option(USE_OPENBLAS "Use OpenBlas" OFF)
option(USE_INTEL_TBB "Use Intel TBB" OFF)

set(OPENBLAS_INCLUDE_DIR "")

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(TINO_USE_OPENMP)
  endif()
endif()

if(USE_OPENBLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    if(EXISTS "/usr/include/x86_64-linux-gnu/openblas-openmp/cblas.h")
      message(
        STATUS
          "Found OpenBLAS header: /usr/include/x86_64-linux-gnu/openblas-openmp/cblas.h"
      )
    elseif(EXISTS "/usr/include/cblas.h")
      message(STATUS "Found OpenBLAS header: /usr/include/cblas.h")
    elseif(EXISTS "${OPENBLAS_LIBRARY_ROOT}/include/cblas.h")
      message(
        STATUS "Found OpenBLAS header: ${OPENBLAS_LIBRARY_ROOT}/include/cblas.h"
      )
      set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_LIBRARY_ROOT}/include")
    else()
      message(FATAL_ERROR "OpenBLAS not found in your system.")
    endif()
    set(OPENBLAS_LIBRARY "-lopenblas")
    add_compile_definitions(TINO_USE_OPENBLAS)

    message(STATUS "${BLAS_LIBRARIES}")
    message(STATUS "${BLAS_LINKER_FLAGS}")

  elseif(EXISTS "${OPENBLAS_LIBRARY_ROOT}/lib/libopenblas.so"
         AND EXISTS "${OPENBLAS_LIBRARY_ROOT}/include/cblas.h")
    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -L ${OPENBLAS_LIBRARY_ROOT}/lib/ -l
    # openblas") set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L
    # ${OPENBLAS_LIBRARY_ROOT}/lib/ -l openblas")
    set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_LIBRARY_ROOT}/include")

    link_directories(${OPENBLAS_LIBRARY_ROOT}/lib)
    set(OPENBLAS_LIBRARY "-lopenblas")

    add_compile_definitions(TINO_USE_OPENBLAS)
    message(STATUS "Found OpenBLAS at ${OPENBLAS_LIBRARY_ROOT}")
  else()
    message(FATAL_ERROR "OpenBLAS not found in your system.")
  endif()
endif()

if(USE_INTEL_TBB)
  find_package(TBB REQUIRED)
  if(TBB_FOUND)
    message(STATUS "Found Intel TBB")
    message(STATUS "Intel TBB Version: ${TBB_VERSION}")
    add_compile_definitions(TINO_USE_INTEL_TBB)
  endif()
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -s -DNDEBUG -march=native ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_MINSIZEREL
    "-Os -s -DNDEBUG -march=native ${CMAKE_CXX_FLAGS}")

add_executable(test_all_in_one.out test_all_in_one.cpp)
target_include_directories(test_all_in_one.out PRIVATE ../
                                                       ${OPENBLAS_INCLUDE_DIR})
target_link_libraries(test_all_in_one.out tbb ${OPENBLAS_LIBRARY})
