cmake_minimum_required(VERSION 3.25)
project(caksm LANGUAGES CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Enforce ISO C++, disables GNU extensions (-std=c++23 not -std=gnu++23)

# Compiler Warnings
add_compile_options(
        -Wall
        -Wextra
        -Wconversion
        -Wpedantic
)

include(FetchContent)

# Eigen (header-only)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.0
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# caksm interface library (header-only)
add_library(caksm INTERFACE)
target_include_directories(caksm INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(caksm INTERFACE Eigen3::Eigen)

# Main Executable
add_executable(pricer src/main.cpp)
target_link_libraries(pricer PRIVATE caksm)

# Profiler Executable (KSM-EI only: memory, timing, roofline)
add_executable(profiler src/profiler.cpp)
target_link_libraries(profiler PRIVATE caksm)

# FMA Loop (AVX-512 FP64 peak benchmark, only built on x86 with AVX-512F support)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx512f CXX_HAS_AVX512F)
if(CXX_HAS_AVX512F)
    add_executable(fma-loop src/fma_loop.cpp)
    target_compile_options(fma-loop PRIVATE -march=native -O3)
else()
    message(STATUS "fma-loop skipped: compiler does not support -mavx512f")
endif()

# Catch2
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Test Executable
enable_testing()  # Activates CTest

add_executable(arnoldi_tests tests/test_arnoldi.cpp)
target_link_libraries(arnoldi_tests
        PRIVATE
        caksm
        Catch2::Catch2WithMain  # Catch2 provides its own main(), you don't write one
)

# Auto-registers each TEST_CASE as a separate CTest test
include(CTest)
include(Catch)
catch_discover_tests(arnoldi_tests)

# STREAM v5.10 Memory Bandwidth Benchmark (OpenMP multi-threaded)
#
# Array size: 100M doubles = 800 MiB/array, 2.4 GiB total across three arrays.
# Rule: each array must be >= 4x the aggregate LLC. With ~48 MiB L3/socket on
# the 8358 host, 100M is safely above that ceiling even across two sockets.
# -mcmodel=medium is required because static data exceeds the 2 GiB small-model
# limit (~85M elements).
#
# Run: ./scripts/stream_sweep.sh for the thread-count sweep (find the bandwidth saturation knee):
find_package(OpenMP REQUIRED)
add_executable(stream src/stream.c)
target_compile_options(stream PRIVATE
    -O3
    -march=native
    -mcmodel=medium
)
target_compile_definitions(stream PRIVATE
    STREAM_ARRAY_SIZE=100000000
    NTIMES=20
)
target_link_libraries(stream PRIVATE OpenMP::OpenMP_C m)
