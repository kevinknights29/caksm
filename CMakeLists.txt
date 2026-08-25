cmake_minimum_required(VERSION 3.25)
project(caksm LANGUAGES CXX C)

# CMake can ask the linker to emit a dependency file, but Synge's system ld does not
# implement --dependency-file. Dependency tracking through the generated build graph is
# sufficient here and keeps a CMake 4.x configuration compatible with that linker.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

option(CAKSM_REQUIRE_MPI
       "Fail configuration when the multi-node GPU executables cannot use MPI"
       OFF)

# The projected exponential uses four statically allocated m-by-m binary64
# arrays. CUDA limits static shared allocations to 48 KiB on both target GPUs,
# so m=39 is the largest common value the current kernel can compile and launch:
# four arrays plus its two shared integers use 48,680 bytes. Keep the value in
# the cache because it changes CUDA code generation and therefore belongs to a
# build configuration, not to a runtime configuration file.
set(CAKSM_GPU_CA_MAX_M 39 CACHE STRING
    "Compile-time maximum Krylov dimension for the GPU CA integrators (1..39)")
if(NOT CAKSM_GPU_CA_MAX_M MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "CAKSM_GPU_CA_MAX_M must be a positive integer; got '${CAKSM_GPU_CA_MAX_M}'")
endif()
if(CAKSM_GPU_CA_MAX_M GREATER 39)
    message(FATAL_ERROR
        "CAKSM_GPU_CA_MAX_M=${CAKSM_GPU_CA_MAX_M} exceeds the current static-shared "
        "kernel limit of 39. A larger value requires a dynamic-shared-memory refactor.")
endif()
message(STATUS
    "GPU CA-Arnoldi compile-time maximum m: ${CAKSM_GPU_CA_MAX_M}")

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
#
# SYSTEM makes Eigen's headers arrive as -isystem, so the -W flags above apply to
# this project's code and not to a dependency nobody here is going to fix. A
# find_package import would get this for free; a FetchContent build tree does not,
# because its targets are ordinary targets in this project. Needs CMake 3.25.
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.0
        SYSTEM
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# caksm interface library (header-only)
add_library(caksm INTERFACE)
target_include_directories(caksm INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(caksm INTERFACE Eigen3::Eigen)
target_compile_definitions(caksm INTERFACE
        CAKSM_GPU_CA_MAX_M=${CAKSM_GPU_CA_MAX_M})

# Source revision stamped into every referee sidecar, so a stored checksum can
# be traced back to the build that produced it.
set(CAKSM_GIT_REVISION "unknown")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE CAKSM_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE CAKSM_GIT_REVISION_RESULT)
    if(NOT CAKSM_GIT_REVISION_RESULT STREQUAL "0"
       OR CAKSM_GIT_REVISION STREQUAL "")
        set(CAKSM_GIT_REVISION "unknown")
    endif()
endif()

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

# Endpoint defect calibration against dense exponential actions on small pricing systems.
add_executable(krylov-indicator-calibration src/krylov_indicator_calibration.cpp)
target_link_libraries(krylov-indicator-calibration PRIVATE caksm)
target_compile_options(krylov-indicator-calibration PRIVATE -O2)

# Financial Reference (the independent terminal-distribution references)
#
# The Johnson formula for the Rainbow price, randomized quasi-Monte Carlo for the
# Basket price and for both payoffs' Delta and diagonal Gamma, and the uncertainty
# each of them carries.
#
# Deliberately does not link caksm: include/financial_reference.hpp reaches no PDE
# assembly, boundary closure, interpolation, Krylov, or ADI code, and pulls in no
# Eigen. A reference that shared the discretization would not be independent of the
# method it is used to judge, and the linker is the cheapest place to enforce that.
# The one shared header is config.hpp, which carries the model parameters and no
# numerics, so both sides price the same contract by construction.
add_executable(financial-reference src/financial_reference.cpp)
target_include_directories(financial-reference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(financial-reference PRIVATE -O2)

# Financial Validation (the PDE-side refinement, Greeks, and equal-accuracy table)
#
# Domain, spatial, temporal, and bump refinement for both payoffs, prices and Greeks
# read through trilinear interpolation from one solved field per configuration, and
# the equal-accuracy comparison of CPU KSM-EI against smoothed CPU HV-ADI. Reads the
# accepted reference row and refuses to run against a mismatched contract.
add_executable(financial-validation src/financial_validation.cpp)
target_link_libraries(financial-validation PRIVATE caksm)
target_compile_options(financial-validation PRIVATE -O2)

# Regime Sweep (the vertical crossover, theta_v)
#
# Times baseline SpMV (m applications) against the cache-blocked matrix-powers kernel
# (include/akx.hpp) building the same Krylov basis in blocks of the certified width, across
# an R_v sweep. The claim under test is a speedup that switches on above R_v = 1, the
# vertical mechanism paying once the working set spills aggregate cache. Single core,
# single cache level. Includes the arithmetic-intensity gate, so a kernel that misses its
# modeled reuse is flagged as a bug, not a finding.
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
# before either coordinate means anything, and regenerates the predictor tables from the
# recorded calibration constants.
#
# Host-only and CUDA-free by design: it links neither a solver nor a timer, so every
# number it prints is an a-priori coordinate. It builds and runs on a login node or a
# laptop, which is what lets the predictions be published before the hardware is touched.
add_executable(regime-gpu-place src/regime_gpu_place.cpp)
target_link_libraries(regime-gpu-place PRIVATE caksm)
target_compile_options(regime-gpu-place PRIVATE -O2)

# Sweeps the participant and node topology instead of the rung, and writes the canonical CSV
# behind the poster's trajectory figure. Host-only for the same reason as its sibling above:
# it reads the accepted placement table and the calibrated preset and adds only the
# predeclared byte model, so no timer is in the loop.
add_executable(regime-gpu-trajectory src/regime_gpu_trajectory.cpp)
target_link_libraries(regime-gpu-trajectory PRIVATE caksm)
target_compile_options(regime-gpu-trajectory PRIVATE -O2)

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
    # roofline gate divides by.
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

    # Static copy/compute-overlap capabilities from cudaGetDeviceProperties. Unlike the
    # calibrated roofs, these require no timed kernel and are safe to collect in gpu_probe.sh.
    add_executable(gpu-device-probe src/gpu_device_probe.cu)
    target_compile_options(gpu-device-probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)

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

    # Split-K Gram kernel (include/gram_splitk.cuh) validated and timed against cuBLAS: the
    # tall-skinny G = B^T B that cuBLAS runs latency-bound, parallelized over the k = n dimension
    # so it reaches the ~s/4 roofline. Correctness vs a cuBLAS gemm, rate vs cuBLAS dsyrk.
    add_executable(gram-splitk-test src/gram_splitk_test.cu)
    target_include_directories(gram-splitk-test PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(gram-splitk-test PRIVATE CUDA::nvml CUDA::cublas)
    target_compile_options(gram-splitk-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # Tall-skinny triangular solve (include/trsm_tallskinny.cuh): the B <- B R^-1 step of CholQR,
    # which cuBLAS runs compute-bound on this shape. One thread per row (independent solves) makes
    # it memory-bound. Correctness vs cuBLAS dtrsm, rate against the DRAM roof.
    add_executable(trsm-test src/trsm_test.cu)
    target_include_directories(trsm-test PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(trsm-test PRIVATE CUDA::nvml CUDA::cublas)
    target_compile_options(trsm-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # The first solver component: a matrix-free pricing stencil that builds a block of
    # powers in one shared-memory tile. The harness validates every vector against the
    # assembled CPU operator before reporting traffic.
    #
    # Built on every default, so this one binary is the baseline of both kernel
    # families at once: full-volume on the production tile, and plane-streamed on
    # the streamed header's default tile. The generated variant list therefore
    # emits neither baseline again.
    add_executable(ca-matrix-powers src/ca_matrix_powers.cu)
    target_include_directories(ca-matrix-powers PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(ca-matrix-powers PRIVATE caksm CUDA::nvml)
    target_compile_definitions(ca-matrix-powers PRIVATE
            CAKSM_MPK_CONFIGURATION="baseline"
            CAKSM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
    target_compile_options(ca-matrix-powers PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>)

    # One function builds every swept matrix-powers configuration, so the sweep
    # widens by regenerating a table rather than by hand-writing targets. The
    # tile extents, thread count and layout are compile-time constants in the
    # kernel, which is why a configuration is a binary and not a flag.
    #
    #   NAME              suffix for ca-matrix-powers-<name>, and the string the
    #                     result row carries as its configuration
    #   FAMILY            full-volume or plane-streamed
    #   TILE_X/Y/Z        interior extents; TILE_Z is unused when streamed
    #   THREADS           threads per block, swept independently of the tile
    #   STREAM_HEIGHT     compile-time default z segment; the runner overrides it
    #   SHARED_PAD_X      shared-memory row padding, in doubles
    #   PITCH_ALIGN       physical row stride, in doubles; 1 is the packed layout
    #   REGISTER_CAP      --maxrregcount value, 0 for none
    #   CACHE_POLICY      ca or cg for -Xptxas=-dlcm, empty for the default
    #   FAST_MATH         adds --use_fast_math; a numerics arm, never a default
    #   VECTORIZATION     adds --extra-device-vectorization
    #   ELIDE_BARRIER     drops the barrier after the basis write
    #   USE_RESTRICT      qualifies the kernel pointers
    #   SOLVER            also build ca-integrator-<name> on the same settings
    function(caksm_add_matrix_powers_variant)
        set(options FAST_MATH VECTORIZATION ELIDE_BARRIER USE_RESTRICT SOLVER)
        set(values NAME FAMILY TILE_X TILE_Y TILE_Z THREADS STREAM_HEIGHT
                SHARED_PAD_X PITCH_ALIGN REGISTER_CAP CACHE_POLICY)
        cmake_parse_arguments(MPKV "${options}" "${values}" "" ${ARGN})
        if(NOT MPKV_NAME)
            message(FATAL_ERROR "caksm_add_matrix_powers_variant needs a NAME")
        endif()
        if(NOT MPKV_FAMILY)
            set(MPKV_FAMILY "full-volume")
        endif()

        set(definitions CAKSM_MPK_CONFIGURATION="${MPKV_NAME}")
        if(MPKV_FAMILY STREQUAL "plane-streamed")
            if(MPKV_TILE_X)
                list(APPEND definitions CAKSM_MPK_STREAM_TILE_X=${MPKV_TILE_X})
            endif()
            if(MPKV_TILE_Y)
                list(APPEND definitions CAKSM_MPK_STREAM_TILE_Y=${MPKV_TILE_Y})
            endif()
            if(MPKV_STREAM_HEIGHT)
                list(APPEND definitions
                        CAKSM_MPK_STREAM_HEIGHT=${MPKV_STREAM_HEIGHT})
            endif()
            if(MPKV_THREADS)
                list(APPEND definitions CAKSM_MPK_STREAM_THREADS=${MPKV_THREADS})
            endif()
            if(MPKV_SHARED_PAD_X)
                list(APPEND definitions
                        CAKSM_MPK_STREAM_SHARED_PAD_X=${MPKV_SHARED_PAD_X})
            endif()
        else()
            if(MPKV_TILE_X)
                list(APPEND definitions CAKSM_MPK_BLOCK_X=${MPKV_TILE_X})
            endif()
            if(MPKV_TILE_Y)
                list(APPEND definitions CAKSM_MPK_BLOCK_Y=${MPKV_TILE_Y})
            endif()
            if(MPKV_TILE_Z)
                list(APPEND definitions CAKSM_MPK_BLOCK_Z=${MPKV_TILE_Z})
            endif()
            if(MPKV_THREADS)
                list(APPEND definitions CAKSM_MPK_THREADS=${MPKV_THREADS})
            endif()
            if(MPKV_SHARED_PAD_X)
                list(APPEND definitions
                        CAKSM_MPK_SHARED_PAD_X=${MPKV_SHARED_PAD_X})
            endif()
        endif()
        if(MPKV_PITCH_ALIGN)
            list(APPEND definitions CAKSM_MPK_PITCH_ALIGN=${MPKV_PITCH_ALIGN})
        endif()
        if(MPKV_ELIDE_BARRIER)
            list(APPEND definitions CAKSM_MPK_ELIDE_BASIS_BARRIER)
        endif()
        if(MPKV_USE_RESTRICT)
            list(APPEND definitions CAKSM_MPK_USE_RESTRICT)
        endif()

        # --resource-usage is what makes registers, spills and code size part of
        # the record instead of something the profiler has to be asked for later.
        set(flags $<$<COMPILE_LANGUAGE:CUDA>:-O3>
                  $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage>)
        if(MPKV_REGISTER_CAP)
            list(APPEND flags
                    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${MPKV_REGISTER_CAP}>)
            list(APPEND definitions
                    CAKSM_MPK_REGISTER_CAP=${MPKV_REGISTER_CAP})
        endif()
        if(MPKV_CACHE_POLICY)
            list(APPEND flags
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-dlcm=${MPKV_CACHE_POLICY}>)
        endif()
        if(MPKV_VECTORIZATION)
            list(APPEND flags
                    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>)
        endif()
        if(MPKV_FAST_MATH)
            list(APPEND flags $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        endif()

        set_property(GLOBAL APPEND PROPERTY CAKSM_MPK_VARIANTS "${MPKV_NAME}")
        set(harness ca-matrix-powers-${MPKV_NAME})
        # Sweep and diagnostic binaries remain reproducible by explicit target
        # name, but failed or superseded experiments must not inflate the
        # production `all` build after the tuning study closes.
        add_executable(${harness} EXCLUDE_FROM_ALL src/ca_matrix_powers.cu)
        target_include_directories(${harness}
                PRIVATE ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(${harness} PRIVATE caksm CUDA::nvml)
        target_compile_definitions(${harness} PRIVATE
                ${definitions}
                CAKSM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
        target_compile_options(${harness} PRIVATE ${flags})

        # A kernel-level win has to be re-reported end to end, and the tile is a
        # compile-time extent, so the solver needs rebuilding on the same
        # settings rather than being told about them at run time. Only requested
        # configurations get one: the kernel screen is much wider than the set
        # worth carrying through a full solver run.
        if(MPKV_SOLVER)
            set(solver ca-integrator-${MPKV_NAME})
            add_executable(${solver} EXCLUDE_FROM_ALL src/ca_arnoldi_gpu.cu)
            target_include_directories(${solver}
                    PRIVATE ${CMAKE_SOURCE_DIR}/include)
            target_link_libraries(${solver}
                    PRIVATE caksm CUDA::nvml CUDA::cublas CUDA::cusolver)
            target_compile_definitions(${solver} PRIVATE
                    ${definitions}
                    CAKSM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
            target_compile_options(${solver} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        endif()
    endfunction()

    # Compile-time experiment arms. Pointer qualification, shared-row padding
    # and the barrier elision cannot be switched at run time, so each keeps its
    # own executable and passes the same assembled-operator gate as the baseline.
    caksm_add_matrix_powers_variant(NAME restrict USE_RESTRICT SOLVER)
    caksm_add_matrix_powers_variant(NAME padded SHARED_PAD_X 1)
    caksm_add_matrix_powers_variant(NAME elide-barrier ELIDE_BARRIER SOLVER)

    # The pitched-layout arm. Kernel-level only: the solver indexes the state
    # logically in several places and refuses to build padded until this arm has
    # cleared its profiling gate, which is deliberate and asserted in the source.
    foreach(pitch_align 4 8 16)
        caksm_add_matrix_powers_variant(
                NAME pitch${pitch_align} PITCH_ALIGN ${pitch_align})
    endforeach()

    # Compilation-flag arms, screened on the baseline geometry so a timing
    # difference is attributable to the flag and not to a shape. Each is built
    # unconditionally and screened by device-code comparison before it is timed:
    # a flag that leaves the cubin unchanged is a recorded no-op, not a result.
    #
    # --use_fast_math is a diagnostic arm only. It relaxes FP64 semantics this
    # stencil depends on, so it is never a promotion candidate and is kept here
    # to answer whether it changes device code at all.
    #
    # Absent on purpose: a global --restrict, because the Chebyshev chunk
    # transition passes one address as both input and output; --allow-expensive
    # -optimizations, already implied at -O3; and -fmad=false, which is a
    # numerics control rather than an optimization. Device LTO is absent because
    # each executable here is a single translation unit over header-only code,
    # so there is no cross-file device optimization for it to perform.
    caksm_add_matrix_powers_variant(NAME vectorized VECTORIZATION)
    caksm_add_matrix_powers_variant(NAME cache-ca CACHE_POLICY ca)
    caksm_add_matrix_powers_variant(NAME cache-cg CACHE_POLICY cg)
    caksm_add_matrix_powers_variant(NAME fastmath FAST_MATH)
    foreach(register_cap 96 80 72 64)
        caksm_add_matrix_powers_variant(
                NAME regcap${register_cap} REGISTER_CAP ${register_cap})
    endforeach()

    # The swept geometry and thread configurations. Generated with
    #   python3 scripts/regime/ca_matrix_powers_configurations.py
    # which also writes data/ca-matrix-powers-tuning/configurations.csv, where
    # every rejected shape keeps the reason it was excluded.
    #
    # The generated list is a configure-time input, so it is registered as a
    # configure dependency: regenerating it re-runs cmake rather than leaving a
    # build directory quietly describing the previous sweep.
    set(CAKSM_MPK_VARIANT_FILE
            ${CMAKE_SOURCE_DIR}/cmake/matrix_powers_variants.cmake)
    if(EXISTS ${CAKSM_MPK_VARIANT_FILE})
        set_property(DIRECTORY APPEND PROPERTY
                CMAKE_CONFIGURE_DEPENDS ${CAKSM_MPK_VARIANT_FILE})
        include(${CAKSM_MPK_VARIANT_FILE})
        get_property(CAKSM_MPK_VARIANTS GLOBAL PROPERTY CAKSM_MPK_VARIANTS)
        list(LENGTH CAKSM_MPK_VARIANTS CAKSM_MPK_VARIANT_COUNT)
        message(STATUS
            "matrix-powers tuning: ${CAKSM_MPK_VARIANT_COUNT} swept variants")
    else()
        # A warning rather than a status line. Without this file the sweep
        # targets do not exist, and the failure surfaces much later as a missing
        # binary in a runner, by which point the cause is several steps back.
        # It is generated rather than committed because the same script writes
        # the configuration table beside it, under a gitignored data directory,
        # and the two have to be produced together to agree.
        message(WARNING
            "No matrix-powers variant list at ${CAKSM_MPK_VARIANT_FILE}. "
            "Only the hand-declared arms will be built, and the tuning and gate "
            "runners will report missing binaries. Generate it with:\n"
            "  python3 scripts/regime/ca_matrix_powers_configurations.py\n"
            "then configure again.")
    endif()

    # The three geometries the earlier tile sweep recorded results under also
    # need a solver binary, because that sweep's kernel win has to be re-reported
    # end to end. They are built by the generated list above; this adds only the
    # solver side, which the wider screen does not need.
    foreach(tile "16x8x8" "8x8x8" "16x8x4")
        if(TARGET ca-matrix-powers-${tile} AND NOT TARGET ca-integrator-${tile})
            string(REPLACE "x" ";" tile_extents "${tile}")
            list(GET tile_extents 0 tile_x)
            list(GET tile_extents 1 tile_y)
            list(GET tile_extents 2 tile_z)
            add_executable(
                    ca-integrator-${tile} EXCLUDE_FROM_ALL
                    src/ca_arnoldi_gpu.cu)
            target_include_directories(ca-integrator-${tile}
                    PRIVATE ${CMAKE_SOURCE_DIR}/include)
            target_link_libraries(ca-integrator-${tile}
                    PRIVATE caksm CUDA::nvml CUDA::cublas CUDA::cusolver)
            target_compile_definitions(ca-integrator-${tile} PRIVATE
                    CAKSM_MPK_BLOCK_X=${tile_x}
                    CAKSM_MPK_BLOCK_Y=${tile_y}
                    CAKSM_MPK_BLOCK_Z=${tile_z}
                    CAKSM_MPK_CONFIGURATION="${tile}"
                    CAKSM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
            target_compile_options(ca-integrator-${tile}
                    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        endif()
    endforeach()

    # One-GPU CA-Arnoldi: the numerical gate at --steps 0, where block Gram-Schmidt
    # and CholQR2 use the measured kernels, H is assembled from the block factors,
    # and exp(H)e1 stays on-device; the single-device integrator at --steps K. One
    # program, one name, so the manifest carries one content hash for it.
    add_executable(ca-integrator src/ca_arnoldi_gpu.cu)
    target_include_directories(ca-integrator PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(ca-integrator
            PRIVATE caksm CUDA::nvml CUDA::cublas CUDA::cusolver)
    target_compile_definitions(ca-integrator PRIVATE
            CAKSM_MPK_CONFIGURATION="baseline"
            CAKSM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
    target_compile_options(ca-integrator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    # ca-integrator-restrict is built by caksm_add_matrix_powers_variant above,
    # from the same arm definition as its kernel harness.

    # Independent full-accuracy referee: the assembled pricing operator is uploaded as CSR
    # and applied with cuSPARSE, keeping this path separate from the production stencil.
    add_executable(ca-referee-gpu src/ca_referee_gpu.cu)
    target_include_directories(ca-referee-gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(ca-referee-gpu
            PRIVATE caksm CUDA::nvml CUDA::cusparse)
    # Synge's nvcc host toolchain is GCC 8-era: std::filesystem is implemented
    # in the separate libstdc++fs compatibility archive there.  Newer GNU
    # toolchains still provide the archive, so linking it explicitly on Linux
    # keeps the CUDA target portable without changing the C++ ABI or numerics.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(ca-referee-gpu PRIVATE stdc++fs)
    endif()
    target_compile_options(ca-referee-gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

    target_compile_definitions(ca-referee-gpu PRIVATE
            CAKSM_GIT_REVISION="${CAKSM_GIT_REVISION}")

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
    # The CUDA sources use MPI's C API only.  Asking FindMPI for the C++ component needlessly
    # depends on the deprecated MPI C++ bindings and can miss an otherwise usable installation.
    # CAKSM_REQUIRE_MPI is intended for multi-node experiment builds: it turns the otherwise
    # useful single-node/NCCL-only fallback into an early, unambiguous configure error.
    find_package(MPI COMPONENTS C)
    if(CAKSM_REQUIRE_MPI AND NOT MPI_C_FOUND)
        message(FATAL_ERROR
            "CAKSM_REQUIRE_MPI=ON, but no MPI C compiler/runtime was found. "
            "Load the site's MPI module, verify `mpicc` and `mpirun` are on PATH, "
            "then configure a fresh build directory or pass -DMPI_C_COMPILER=/path/to/mpicc.")
    endif()
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

        # Two-GPU measured horizontal crossover: MGS vs s-step with real inter-GPU all-reduces.
        # DEVICE_P2P rung only (single process, both local GPUs), so it only needs NCCL.
        # cuBLAS and cuSOLVER supply the same kernels the single-GPU s-step harness calls.
        add_executable(regime-gpu-sstep-2gpu src/regime_gpu_sstep_2gpu.cu)
        target_include_directories(regime-gpu-sstep-2gpu
                PRIVATE ${NCCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(regime-gpu-sstep-2gpu
                PRIVATE ${NCCL_LIBRARY} CUDA::nvml CUDA::cublas CUDA::cusolver)
        target_compile_options(regime-gpu-sstep-2gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

        add_executable(ca-matrix-powers-2gpu src/ca_matrix_powers_2gpu.cu)
        target_include_directories(ca-matrix-powers-2gpu
                PRIVATE ${NCCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(ca-matrix-powers-2gpu
                PRIVATE caksm ${NCCL_LIBRARY} CUDA::nvml)
        target_compile_options(ca-matrix-powers-2gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

        add_executable(ca-integrator-2gpu src/ca_integrator_2gpu.cu)
        target_include_directories(ca-integrator-2gpu
                PRIVATE ${NCCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(ca-integrator-2gpu
                PRIVATE caksm ${NCCL_LIBRARY} CUDA::nvml CUDA::cublas
                        CUDA::cusolver CUDA::nvToolsExt)
        target_compile_options(ca-integrator-2gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        if(MPI_C_FOUND)
            target_link_libraries(ca-integrator-2gpu PRIVATE MPI::MPI_C)
            target_compile_definitions(ca-integrator-2gpu PRIVATE CAKSM_HAVE_MPI)
            message(STATUS
                "ca-integrator-2gpu: MPI bootstrap enabled for the multi-node NCCL path")
        else()
            message(STATUS
                "ca-integrator-2gpu: NCCL only, no MPI; the multi-node path is disabled")
        endif()

        # Retained only to reproduce the closed pointer-qualification arm.
        add_executable(
                ca-integrator-2gpu-restrict EXCLUDE_FROM_ALL
                src/ca_integrator_2gpu.cu)
        target_include_directories(ca-integrator-2gpu-restrict
                PRIVATE ${NCCL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(ca-integrator-2gpu-restrict
                PRIVATE caksm ${NCCL_LIBRARY} CUDA::nvml CUDA::cublas
                        CUDA::cusolver CUDA::nvToolsExt)
        target_compile_definitions(ca-integrator-2gpu-restrict
                PRIVATE CAKSM_MPK_USE_RESTRICT)
        target_compile_options(ca-integrator-2gpu-restrict
                PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
        if(MPI_C_FOUND)
            target_link_libraries(ca-integrator-2gpu-restrict PRIVATE MPI::MPI_C)
            target_compile_definitions(ca-integrator-2gpu-restrict
                    PRIVATE CAKSM_HAVE_MPI)
        endif()

        if(MPI_C_FOUND)
            target_link_libraries(calibrate-gpu-p2p PRIVATE MPI::MPI_C)
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
# SYSTEM for the same reason as Eigen above: Catch2's headers are not this
# project's warnings to answer for.
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
        SYSTEM
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

add_executable(ca_referee_tests tests/test_ca_referee_scaled.cpp)
target_link_libraries(
        ca_referee_tests PRIVATE caksm Catch2::Catch2WithMain)

# Financial-validation tests: the analytical Rainbow reference, the QMC limiting
# cases, trilinear log-grid interpolation, and the HV start-up smoothing.
add_executable(financial_validation_tests tests/test_financial_validation.cpp)
target_link_libraries(
        financial_validation_tests PRIVATE caksm Catch2::Catch2WithMain)
target_compile_options(financial_validation_tests PRIVATE -O2)

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
        tests/test_ca_integrator_memory.cpp
)
target_link_libraries(regime_tests PRIVATE caksm Catch2::Catch2WithMain)
target_compile_options(regime_tests PRIVATE -O2)

# Auto-registers each test case as a separate CTest test
include(CTest)
include(Catch)
catch_discover_tests(arnoldi_tests)
catch_discover_tests(ca_referee_tests TEST_PREFIX "ca-referee.")
catch_discover_tests(financial_validation_tests TEST_PREFIX "financial.")
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
