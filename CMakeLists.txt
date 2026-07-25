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

# Regime GPU Place (the GPU map's predictor)
#
# Places points on the GPU (R_v, R_h) plane, evaluates the roofline gate that must pass
# before either coordinate means anything, and regenerates docs/regime_gpu_phase0.md's
# tables from whatever constants the calibration has recorded.
#
# Host-only and CUDA-free by design: it links neither a solver nor a timer, so every
# number it prints is an a-priori coordinate. It builds and runs on a login node or a
# laptop, which is what lets the predictions be published before the hardware is touched.
add_executable(regime-gpu-place src/regime_gpu_place.cpp)
target_link_libraries(regime-gpu-place PRIVATE caksm)
target_compile_options(regime-gpu-place PRIVATE -O2)

# GPU instruments (optional: built only where a CUDA toolkit is present)
#
# The GPU analogues of calibrate-alpha and stream. Everything they measure is an offline
# machine constant, recorded into include/gpu_machine.hpp by hand, exactly as the CPU
# reduction constants are: no binary writes a preset, so a calibration is always a
# deliberate, reviewable edit.
#
#   calibrate-gpu-reduction  warp / block / grid rungs + kernel-launch latency (CUDA only)
#   gpu-stream               achieved HBM and L2 roofs + the occupancy sweep (CUDA only)
#   gpu-fma-loop             measured FP64/FP32 peak: the roof the gate divides by (CUDA only)
#   calibrate-gpu-p2p        device-to-device and node rungs (CUDA + NCCL; MPI optional)
#
# The p2p target is separated so a missing NCCL cannot block Phase A: the first three
# calibrate with nothing but a CUDA toolkit and a single device, which is the whole of what
# the negative arm needs.
#
# Host compiler: use the CUDAHOSTCXX environment variable, not -DCMAKE_CUDA_HOST_COMPILER.
# nvcc refuses host compilers newer than it knows about (CUDA 12.8 caps GCC at 14), and both
# target machines default to a newer GCC than that. Passing -DCMAKE_CUDA_HOST_COMPILER on the
# command line does not work, because CMake's compiler-identification step runs before that
# cache variable is available and invokes bare `gcc` from PATH. CUDAHOSTCXX is read early
# enough to apply:
#
#   CUDAHOSTCXX=/usr/bin/g++ cmake -B build \
#       -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
#
# check_language() is a silent probe: it try_compiles a scratch CUDA project and reports only
# NOTFOUND on failure, discarding the compiler's complaint. That is the right default, since a
# machine with no toolkit must still configure the CPU targets, but it is useless when nvcc
# plainly works from the shell and the probe still fails.
#
# The escape hatch is to pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc explicitly. check_language()
# short-circuits when the variable is already defined, so enable_language(CUDA) runs the real
# detection and fails with the underlying error. Configure then aborts, taking the CPU targets
# with it, which is deliberate: at that point the error message is what you want.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)          # not 23: nvcc's host pass lags the C++ targets here
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # sm_70 = V100 (synge), sm_86 = RTX 3090 (puffin). Both are compiled so one build serves
    # the controlled pair.
    #
    # Override per host with -DCMAKE_CUDA_ARCHITECTURES=70 on synge, 86 on puffin. The
    # default is the pair, not a promise that both compile everywhere.
    #
    # Pin the toolkit to 12.8 on both. CUDA 13.x removed offline compilation for Volta, so
    # synge has no choice and puffin meets synge rather than the reverse. The reason is
    # control, not convenience: the two cards are a controlled pair differing only in FP64
    # rate, and two different nvcc versions would put code generation and default FMA
    # contraction into that comparison, both of which land on the measured FP64 peak the
    # roofline gate divides by. See docs/regime_gpu_phase0.md.
    #   cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc ...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 86)
    endif()

    # NVML (CUDA::nvml) backs the contention check in include/gpu_contention.cuh, which needs
    # to enumerate resident PIDs and exclude our own: cudaMemGetInfo counts this process's
    # ~265 MiB context and so flagged an idle 3090 as contended. CUDAToolkit links the stub at
    # build and the driver's libnvidia-ml at runtime. Part of every CUDA install, so this does
    # not add an external dependency the way NCCL does.
    find_package(CUDAToolkit REQUIRED)

    # The include path only, not the caksm interface target: these binaries must not see Eigen
    # or the solver. They pick up include/gpu_contention.cuh and nothing else, which keeps them
    # offline instruments in the same sense calibrate-alpha is.
    add_executable(calibrate-gpu-reduction src/calibrate_gpu_reduction.cu)
    set_target_properties(calibrate-gpu-reduction PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)  # required for cooperative-groups grid.sync()
    target_include_directories(calibrate-gpu-reduction PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(calibrate-gpu-reduction PRIVATE CUDA::nvml)
    target_compile_options(calibrate-gpu-reduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    add_executable(gpu-stream src/gpu_stream.cu)
    target_include_directories(gpu-stream PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(gpu-stream PRIVATE CUDA::nvml)
    target_compile_options(gpu-stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # No -use_fast_math and no -ffast-math analogue: the FMA chain must stay exactly the
    # dependent sequence written, or what is measured is whatever ptxas contracted it into.
    add_executable(gpu-fma-loop src/gpu_fma_loop.cu)
    target_include_directories(gpu-fma-loop PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(gpu-fma-loop PRIVATE CUDA::nvml)
    target_compile_options(gpu-fma-loop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # Times baseline SpMV to test the vertical mechanism's precondition.
    add_executable(regime-gpu-spmv src/regime_gpu_spmv.cu)
    target_include_directories(regime-gpu-spmv PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(regime-gpu-spmv PRIVATE CUDA::nvml CUDA::cusparse)
    target_compile_options(regime-gpu-spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # Timed MGS: the reduction-bound half of the cycle, and R_h's denominator.
    # cuBLAS supplies the dot / axpy / norm kernels a real implementation calls.
    add_executable(regime-gpu-mgs src/regime_gpu_mgs.cu)
    target_include_directories(regime-gpu-mgs PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(regime-gpu-mgs PRIVATE CUDA::nvml CUDA::cublas)
    target_compile_options(regime-gpu-mgs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # Times s-step orthogonalization: CholQR runs on cuBLAS (Gram, triangular solve)
    # and cuSOLVER (Cholesky). Compares against MGS to locate the horizontal crossover,
    # and reports the Gram block's FP64 rate for the two-card contrast.
    add_executable(regime-gpu-sstep src/regime_gpu_sstep.cu)
    target_include_directories(regime-gpu-sstep PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(regime-gpu-sstep PRIVATE CUDA::nvml CUDA::cublas CUDA::cusolver)
    target_compile_options(regime-gpu-sstep PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # The interconnect rungs need a real collectives library: the reduction timed must be the
    # one an implementation would actually call, or R_h's calibrated constant is the slope of
    # code nobody runs.
    #
    # NCCL is required for this target, MPI is optional. Both of synge's V100s sit on one
    # node, so ncclCommInitAll drives the DEVICE_P2P rung from a single process with no
    # launcher, and that is the rung Phase 0 predicts carries the production operator into
    # Upper-Right. MPI buys only the NODE rung, so a site without it loses the fabric
    # measurement rather than the headline one.
    #
    # NAMES carries libnccl.so.2 explicitly: a pip-installed NCCL (nvidia-nccl-cu12) ships the
    # versioned soname with no unversioned symlink, and plain `nccl` would miss it.
    find_package(MPI COMPONENTS CXX)
    find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2
            HINTS ENV NCCL_HOME ENV NCCL_ROOT ENV CUDA_HOME
            PATH_SUFFIXES lib lib64 nvidia/nccl/lib)
    find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS ENV NCCL_HOME ENV NCCL_ROOT ENV CUDA_HOME
            PATH_SUFFIXES include nvidia/nccl/include)
    if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
        add_executable(calibrate-gpu-p2p src/calibrate_gpu_p2p.cu)
        target_include_directories(calibrate-gpu-p2p
                PRIVATE ${NCCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(calibrate-gpu-p2p PRIVATE ${NCCL_LIBRARY} CUDA::nvml)
        target_compile_options(calibrate-gpu-p2p PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        if(MPI_CXX_FOUND)
            target_link_libraries(calibrate-gpu-p2p PRIVATE MPI::MPI_CXX)
            target_compile_definitions(calibrate-gpu-p2p PRIVATE CAKSM_HAVE_MPI)
            message(STATUS "calibrate-gpu-p2p: NCCL + MPI, both DEVICE_P2P and NODE rungs")
        else()
            message(STATUS
                "calibrate-gpu-p2p: NCCL only, no MPI. The DEVICE_P2P rung is reachable "
                "(single process, both local GPUs), the NODE rung is not. reduction_calibrated "
                "must stay false until the fabric rung lands.")
        endif()
    else()
        message(STATUS
            "calibrate-gpu-p2p skipped: NCCL not found. Set NCCL_HOME to a directory holding "
            "include/nccl.h and lib/libnccl.so*. Both interconnect rungs stay uncalibrated, so "
            "reduction_calibrated must remain false and no horizontal verdict may be published.")
    endif()
else()
    # Distinguish "no toolkit" from "toolkit present but the compiler check failed": they need
    # different fixes and the bare CMake message conflates them. nvcc pins a maximum supported
    # host GCC, and both synge and puffin ship a newer default compiler than their toolkit
    # accepts, so check_language()'s try_compile fails with nvcc plainly on PATH.
    find_program(CAKSM_NVCC nvcc)
    if(CAKSM_NVCC)
        message(WARNING
            "GPU instruments skipped, but nvcc is on PATH at ${CAKSM_NVCC}.\n"
            "check_language(CUDA) compiled nothing and discarded the reason. The likely cause "
            "is that nvcc rejected this build's default host compiler "
            "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}) as too new: CUDA 12.8 "
            "caps GCC at 14, and CMake's compiler-identification step invokes bare `gcc` from "
            "PATH regardless of -DCMAKE_CUDA_HOST_COMPILER.\n"
            "Set the environment variable instead, which is read early enough to apply, and "
            "force the compiler so the silent probe is bypassed and any remaining failure is "
            "reported in full:\n"
            "  CUDAHOSTCXX=/usr/bin/g++ cmake -B build -DCMAKE_CUDA_COMPILER=${CAKSM_NVCC} "
            "-DCMAKE_CUDA_ARCHITECTURES=<86 for RTX 3090, 70 for V100>\n"
            "The .cu sources are C++17, so an older host compiler costs nothing; only the "
            "C++23 targets need a recent GCC, and they keep using CMAKE_CXX_COMPILER "
            "(${CMAKE_CXX_COMPILER}).")
    else()
        message(STATUS "GPU instruments skipped: no CUDA compiler found. "
                       "regime-gpu-place still builds; it is host-only by design.")
    endif()
endif()

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
        tests/test_gpu_regime.cpp
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
