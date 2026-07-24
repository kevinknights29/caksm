/**
 * @file gpu_contention.cuh
 * @brief Detect co-tenant processes on a GPU, and withhold recordable constants when one is
 *        present.
 *
 * Every GPU calibration here is a timing, and a co-tenant moves all of them at once: it holds
 * device memory that displaces the working set, occupies SMs, and competes for the bandwidth
 * gpu-stream is measuring. The result is wrong rather than noisy, and it looks plausible. It
 * also shares the single 6 MiB L2 that R_v is measured against, with no way to account for it.
 *
 * Detection is process-based, via NVML. The obvious test, cudaMemGetInfo total minus free, is
 * sampled after cudaSetDevice and so counts this process's own context (~265 MiB on an RTX
 * 3090), which flagged a genuinely idle card as contended. NVML enumerates resident PIDs, so
 * ours is excluded and only foreign memory counts. Without NVML the code falls back to the
 * coarse test with a context-aware threshold and reports the verdict as provisional.
 *
 * A contended run still proves the kernels compile, launch and complete, so it is allowed: the
 * binary runs, prints its numbers, withholds the "To record, set these..." block, and stamps
 * contended=1 into the CSV.
 *
 * @author Kevin Knights
 * @date 2026-07-22
 */
#pragma once

#include <cuda_runtime.h>
#include <nvml.h>

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <vector>

/// Foreign occupancy of the device, sampled before this process did any work.
struct DeviceContention {
    std::size_t other_bytes = 0;    ///< device memory held by processes other than ours
    int         other_procs = 0;    ///< count of resident PIDs other than ours
    std::size_t total_bytes = 0;
    bool        contended   = false;
    bool        via_nvml    = false;///< true = process-accurate; false = coarse cudaMemGetInfo
};

namespace detail {

/// The current CUDA device's NVML handle, matched by PCI bus id rather than by index, so a
/// CUDA_VISIBLE_DEVICES remapping cannot point us at the wrong physical card.
[[nodiscard]] inline bool nvml_handle_for_current_device(nvmlDevice_t& out)
{
    int ordinal = 0;
    if (cudaGetDevice(&ordinal) != cudaSuccess) return false;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return false;

    char pci[32];
    std::snprintf(pci, sizeof(pci), "%04X:%02X:%02X.0",
                  prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    return nvmlDeviceGetHandleByPciBusId(pci, &out) == NVML_SUCCESS;
}

/// Sum the memory of resident processes whose PID is not ours, over one NVML query function.
/// Appends to the running totals. Returns false only when the query is unsupported, so the
/// caller can tell an empty device from an unanswerable question.
template <typename QueryFn>
[[nodiscard]] inline bool accumulate_foreign(QueryFn query, nvmlDevice_t dev, unsigned int me,
                                             std::size_t& bytes, int& procs)
{
    unsigned int count = 0;
    const nvmlReturn_t probe = query(dev, &count, nullptr);
    if (probe == NVML_SUCCESS) return true;                 // genuinely zero processes
    if (probe != NVML_ERROR_INSUFFICIENT_SIZE) return false;// unsupported / no permission

    std::vector<nvmlProcessInfo_t> infos(count);
    if (query(dev, &count, infos.data()) != NVML_SUCCESS) return false;

    for (unsigned int i = 0; i < count; ++i) {
        if (infos[i].pid == me) continue;                   // our own context is not contention
        ++procs;
        // usedGpuMemory reads NVML_VALUE_NOT_AVAILABLE when the driver will not disclose a
        // foreign process's footprint. A co-tenant we cannot size is still a co-tenant, so
        // its presence, counted above, is what trips the verdict.
        if (infos[i].usedGpuMemory != static_cast<unsigned long long>(-1))
            bytes += static_cast<std::size_t>(infos[i].usedGpuMemory);
    }
    return true;
}

}  // namespace detail

/**
 * @brief Sample foreign device occupancy. Call after cudaSetDevice, before any cudaMalloc.
 *
 * @param threshold_bytes foreign memory above which the device counts as contended. Small by
 *        design: 64 MiB clears rounding while still catching any real tenant, and the cluster
 *        is shared with other researchers, so an idle card is never guaranteed. Foreign memory
 *        only, so our own ~265 MiB context does not enter.
 */
[[nodiscard]] inline DeviceContention check_device_contention(
    std::size_t threshold_bytes = 64u << 20)
{
    DeviceContention c;

    // Preferred path: NVML process enumeration, which can exclude our own PID.
    if (nvmlInit_v2() == NVML_SUCCESS) {
        nvmlDevice_t dev{};
        if (detail::nvml_handle_for_current_device(dev)) {
            nvmlMemory_t mem{};
            if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS)
                c.total_bytes = static_cast<std::size_t>(mem.total);

            const unsigned int me = static_cast<unsigned int>(getpid());
            const bool ok_c = detail::accumulate_foreign(
                nvmlDeviceGetComputeRunningProcesses, dev, me, c.other_bytes, c.other_procs);
            const bool ok_g = detail::accumulate_foreign(
                nvmlDeviceGetGraphicsRunningProcesses, dev, me, c.other_bytes, c.other_procs);

            if (ok_c && ok_g) {
                c.via_nvml = true;
                // Contended when a foreign process holds memory over the threshold, or when
                // one is present whose footprint the driver would not disclose. A small
                // sub-threshold display server is tolerated.
                c.contended = c.other_bytes >= threshold_bytes
                           || (c.other_procs > 0 && c.other_bytes == 0);
                nvmlShutdown();
                return c;
            }
        }
        nvmlShutdown();
    }

    // Fallback: cudaMemGetInfo cannot exclude our own context, so the threshold must clear it.
    // Too coarse to see a small co-tenant, and flagged via_nvml=false so the caller reports
    // the verdict as provisional.
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        c.total_bytes = total_b;
        const std::size_t used = total_b - free_b;
        constexpr std::size_t kContextSlack = 700u << 20;   // clears a bare context + margin
        c.other_bytes = used > kContextSlack ? used - kContextSlack : 0;
        c.contended   = used > kContextSlack;
    }
    return c;
}

/// Print the verdict. Loud when contended, because the numbers that follow will look fine.
inline void report_contention(const DeviceContention& c)
{
    const double other_mib = static_cast<double>(c.other_bytes) / (1024.0 * 1024.0);
    const double tot_mib   = static_cast<double>(c.total_bytes) / (1024.0 * 1024.0);
    const char* method = c.via_nvml ? "NVML, foreign processes only"
                                    : "cudaMemGetInfo fallback (coarse; excludes our context "
                                      "by a fixed margin, cannot see a small co-tenant)";

    if (!c.contended) {
        std::printf("  device is idle (%d foreign process(es), %.0f MiB; %s): "
                    "results are recordable\n\n", c.other_procs, other_mib, method);
        return;
    }
    std::printf("\n");
    std::printf("  ########################################################################\n");
    std::printf("  #  CONTENDED DEVICE: %d other process(es) hold %.0f MiB of %.0f MiB.\n",
                c.other_procs, other_mib, tot_mib);
    std::printf("  #  detection: %s\n", method);
    std::printf("  #\n");
    std::printf("  #  The numbers below are not recordable. A co-tenant displaces the\n");
    std::printf("  #  working set, occupies SMs, and competes for the bandwidth being\n");
    std::printf("  #  measured, so what comes out is wrong rather than noisy, and it will\n");
    std::printf("  #  not look wrong. A shared L2 is not the cache geometry the preset\n");
    std::printf("  #  describes.\n");
    std::printf("  #\n");
    std::printf("  #  Running anyway is still worth something as a smoke test: it proves\n");
    std::printf("  #  the kernels compile, launch and complete. Treat it as nothing more,\n");
    std::printf("  #  and re-run on an idle device before transcribing any constant.\n");
    std::printf("  #  Check with: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv\n");
    std::printf("  ########################################################################\n\n");
}

/// The suppression itself: true when the caller may print constants for transcription.
[[nodiscard]] inline bool may_record(const DeviceContention& c) noexcept
{
    return !c.contended;
}

/**
 * @brief Which toolkit built this binary, plus the runtime and driver it found.
 *
 * Not the same question as `nvcc --version`: puffin carries seven toolkits and the build is
 * pinned with -DCMAKE_CUDA_COMPILER, so the nvcc first on PATH is often not the one that
 * compiled anything. Toolkit version is load-bearing, since code generation and default FMA
 * contraction both move between major CUDA versions and land on the measured FP64 peak the
 * roofline gate divides by. __CUDACC_VER_* are baked in at compile time, so PATH cannot fool
 * this. See docs/regime_gpu_phase0.md.
 */
inline void report_toolkit()
{
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::printf("  built with nvcc %d.%d | runtime %d.%d | driver supports up to %d.%d\n",
                __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
                rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);
}

/// Printed in place of the "To record" block on a contended device.
inline void report_suppressed()
{
    std::printf("Recording block suppressed: the device was contended when this ran, so these\n"
                "constants must not reach include/gpu_machine.hpp. Re-run on an idle GPU.\n");
}
