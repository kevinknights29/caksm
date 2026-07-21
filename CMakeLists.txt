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
target_compile_options(pricer PRIVATE -march=native -O3)
target_compile_definitions(pricer PRIVATE EIGEN_ENABLE_AVX512)

# Profiler Executable (KSM-EI only: memory, timing, roofline)
add_executable(profiler src/profiler.cpp)
target_link_libraries(profiler PRIVATE caksm)
target_compile_options(profiler PRIVATE -march=native -O3)
target_compile_definitions(profiler PRIVATE EIGEN_ENABLE_AVX512)

# Scaling Executable (OpenMP strong/weak-scaling harness for KSM-EI)
#
# Per-kernel instrumented KSM-EI at a fixed Krylov dimension, with a hand-rolled
# CSR row-partitioned SpMV and parallel Gram-Schmidt, so SpMV's bandwidth
# plateau and Modified Gram-Schmidt's synchronization roll-off appear on the same axis.
# Thread count P comes from OMP_NUM_THREADS, placement from OMP_PLACES=cores and
# OMP_PROC_BIND=close (set by scripts/scaling/scaling_strong.sh and scaling_weak.sh).
find_package(OpenMP REQUIRED)
add_executable(scaling src/scaling.cpp)
target_link_libraries(scaling PRIVATE caksm OpenMP::OpenMP_CXX)
target_compile_options(scaling PRIVATE -march=native -O3)
target_compile_definitions(scaling PRIVATE EIGEN_ENABLE_AVX512)

# Regime Control (numerics gate: Laplacian-scaffold controls)
#
# Validates the synthetic instrument's a-priori predictions about arithmetic before any
# claim about communication is attempted: the analytic Kronecker-sum spectrum, the
# permutation-invariance that makes pure-axis motion possible, the Hochbruck-Lubich bound
# on the Krylov dimension, the Vandermonde prediction of matrix-powers basis conditioning,
# and the soundness of the u^(-1/2) CholeskyQR certificate.
#
# Exits non-zero on failure; do not proceed to the scattered-pattern operator until it passes.
add_executable(regime-control src/regime_control.cpp)
target_link_libraries(regime-control PRIVATE caksm)
target_compile_options(regime-control PRIVATE -march=native -O3)
target_compile_definitions(regime-control PRIVATE EIGEN_ENABLE_AVX512)

# Regime Sweep (the vertical crossover, theta_v)
#
# Times baseline SpMV (m applications) against the cache-blocked matrix-powers kernel
# (include/akx.hpp) building the same Krylov basis in blocks of the certified width, across
# an R_v sweep. The claim under test is a speedup that switches on above R_v = 1, the
# vertical mechanism paying once the working set spills aggregate cache. Single core,
# single cache level. Includes the arithmetic-intensity gate, so a kernel that misses its
# modelled reuse is flagged as a bug, not a finding.
add_executable(regime-sweep src/regime_sweep.cpp)
target_link_libraries(regime-sweep PRIVATE caksm)
target_compile_options(regime-sweep PRIVATE -march=native -O3)
target_compile_definitions(regime-sweep PRIVATE EIGEN_ENABLE_AVX512)

# Alpha Calibration (the reduction-latency machine constant)
#
# Measures alpha, the per-tree-level latency of one global reduction, by timing a bare
# scalar all-reduce over a bound OpenMP team at each CCX-aligned P and fitting
# t_reduce(P) = alpha * log2(P) through the origin.
#
# Deliberately links neither caksm nor Eigen: alpha is an a-priori predictor, so it must be
# knowable without reference to the method under study. If this binary could see the solver,
# R_h would risk becoming a measured fraction of the runtime it is supposed to predict.
#
# Run: ./scripts/regime/calibrate_alpha.sh, then record the fitted value in include/machine.hpp.
add_executable(calibrate-alpha src/calibrate_alpha.cpp)
target_include_directories(calibrate-alpha PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(calibrate-alpha PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(calibrate-alpha PRIVATE -march=native -O3)

# FMA Loop (peak FP64 throughput benchmark)
# Requires AVX2+FMA (AMD/Intel Haswell+).
# Uses -march=native so the compiler selects the right ISA path automatically.
# On arm64 / unsupported ISA the check fails silently and the target is skipped.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx2    CXX_HAS_AVX2)
if(CXX_HAS_AVX512F OR CXX_HAS_AVX2)
    add_executable(fma-loop src/fma_loop.cpp)
    target_compile_options(fma-loop PRIVATE -march=native -O3)
else()
    message(STATUS "fma-loop skipped: requires AVX2+FMA (x86-64 only)")
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

# Regime-instrument tests: the invariants the map rests on (permutation similarity,
# R_v's aggregate accounting frame, MGS/s-step reduction counts, CholQR breakdown).
#
# Split by subject across several translation units.
add_executable(regime_tests
        tests/test_regime_coordinates.cpp
        tests/test_regime_synthetic.cpp
        tests/test_regime_krylov.cpp
        tests/test_regime_nonnormal.cpp
        tests/test_regime_kernels.cpp
)
target_link_libraries(regime_tests PRIVATE caksm Catch2::Catch2WithMain)
target_compile_options(regime_tests PRIVATE -O2)

# Auto-registers each test case as a separate CTest test
include(CTest)
include(Catch)
catch_discover_tests(arnoldi_tests)
# Prefix the regime-instrument tests so `ctest -R regime` selects them: ctest filters on
# test name, and Catch2 registers each test case under its description, not its [regime]
# tag or target name.
catch_discover_tests(regime_tests TEST_PREFIX "regime.")

# STREAM v5.10 Memory Bandwidth Benchmark (OpenMP multi-threaded)
#
# Array size: 100M doubles = 800 MiB/array, 2.4 GiB across three arrays. Each array must be
# >= 4x the aggregate LLC; puffin's aggregate L3 is 128 MiB (8 CCX x 16 MiB), so the rule
# demands >= 512 MiB/array and 800 MiB clears it. -mcmodel=medium is required because
# static data exceeds the 2 GiB small-model limit (~85M elements).
#
# Run: ./scripts/stream/stream_sweep.sh for the thread-count sweep.
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
