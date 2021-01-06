cmake_minimum_required(VERSION 2.8)

project(test_tino CXX)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og -pg")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -s -DNDEBUG -march=native")

option(USE_OPENMP "Use OpenMP" OFF)

find_package(OpenMP)
if(OpenMP_FOUND AND USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(TINO_USE_OPENMP)
endif()

add_executable(test_naive.out test_naive.cpp)
target_include_directories(test_naive.out PRIVATE ../)

add_executable(test_openblas.out test_openblas.cpp)
target_include_directories(test_openblas.out PRIVATE ../)