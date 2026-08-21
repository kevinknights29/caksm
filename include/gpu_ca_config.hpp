/**
 * @file gpu_ca_config.hpp
 * @brief Build-time resource limits shared by GPU CA-Arnoldi code and its CPU calibration.
 */
#pragma once

#include <cstddef>

// Direct compilation outside CMake retains the production default. CMake
// supplies this macro from its CAKSM_GPU_CA_MAX_M cache variable.
#ifndef CAKSM_GPU_CA_MAX_M
#define CAKSM_GPU_CA_MAX_M 39
#endif

inline constexpr int kGpuCaMaxM = CAKSM_GPU_CA_MAX_M;

// ca_expm_candidates has the larger of the two projected-exponential static
// footprints: four m-by-m binary64 arrays and two shared integers.
inline constexpr std::size_t kGpuCaExpmStaticSharedLimitBytes = 48U * 1024U;
inline constexpr std::size_t kGpuCaExpmStaticSharedBytes =
    4U * static_cast<std::size_t>(kGpuCaMaxM)
       * static_cast<std::size_t>(kGpuCaMaxM) * sizeof(double)
    + 2U * sizeof(int);

static_assert(kGpuCaMaxM >= 1, "GPU CA maximum m must be positive");
static_assert(
    kGpuCaExpmStaticSharedBytes <= kGpuCaExpmStaticSharedLimitBytes,
    "GPU CA maximum m exceeds the 48 KiB static shared-memory limit");
