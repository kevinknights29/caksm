cmake_minimum_required(VERSION 3.25)
project(parallel_krylov_subspace LANGUAGES CXX)

# ── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Enforce ISO C++, disables GNU extensions (-std=c++23 not -std=gnu++23)

# ── Compiler Warnings ─────────────────────────────────────────────────────────
# Replicate your Makefile flags, but scoped to targets (best practice)
add_compile_options(
        -Wall
        -Wextra
        -Wconversion
        -Wpedantic
)

# ── Eigen ─────────────────────────────────────────────────────────────────────
# Since you vendor Eigen locally (no system install), tell CMake where it lives.
# This mirrors your -I flag from the Makefile.
add_library(Eigen3 INTERFACE)  # Interface = header-only, no compiled artifact
target_include_directories(Eigen3 INTERFACE
        ${CMAKE_SOURCE_DIR}/eigen  # adjust if your eigen/ folder sits elsewhere
)

# ── Arnoldi as an INTERFACE library ──────────────────────────────────────────
# This lets both main.cpp and the tests link against the algorithm
# without compiling it twice into a .a/.so (it's header-only).
add_library(arnoldi_lib INTERFACE)
target_include_directories(arnoldi_lib INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(arnoldi_lib INTERFACE Eigen3)

# ── Main Executable ───────────────────────────────────────────────────────────
add_executable(arnoldi src/main.cpp
        src/arnoldi.cc)
target_link_libraries(arnoldi PRIVATE arnoldi_lib)

# ── Catch2 via FetchContent ───────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)  # Downloads + configures Catch2 automatically

# ── Test Executable ───────────────────────────────────────────────────────────
enable_testing()  # Activates CTest

add_executable(arnoldi_tests tests/test_arnoldi.cpp)
target_link_libraries(arnoldi_tests
        PRIVATE
        arnoldi_lib
        Catch2::Catch2WithMain  # Catch2 provides its own main(), you don't write one
)

# Auto-registers each TEST_CASE as a separate CTest test
include(CTest)
include(Catch)
catch_discover_tests(arnoldi_tests)
