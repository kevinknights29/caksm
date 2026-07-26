/**
 * @file gpu_fma_loop.cu
 * @brief Measured FP64 and FP32 peak on a GPU: the compute roof the roofline gate divides by.
 *
 * The analogue of src/fma_loop.cpp. Everything else in this study measures its constants: the
 * CPU's peak_gflops_core is 68.12 because fma-loop measured it, not because a datasheet said
 * so. The GPU compute roof was the one figure still transcribed, and it is the wrong place to
 * trust a datasheet, because the negative arm of the two-card contrast is a claim about the
 * FP64 rate:
 *
 *   - roofline_gate() divides by fp64_flops_peak to get the ridge.
 *   - The 3090's whole role in the study is that its ridge is ~13x lower than the V100's.
 *   - The margins on consumer silicon are thin. MGS clears the 3090's ridge by under 2x, so
 *     an FP64 rate wrong by a factor of two flips a gate verdict, and with it the headline
 *     claim about where the map stops applying.
 *
 * The FP64:FP32 ratio is the reading to take: 1:2 on a datacenter part with dedicated FP64
 * units, 1:64 on GA102 where FP64 exists only for compatibility. The kernel exposes that ratio
 * directly, which turns "the 3090 is throttled 1:64" into a measurement.
 *
 * Each thread runs kUnroll independent FMA chains so instruction-level parallelism covers the
 * pipeline latency; a single dependent chain would measure latency rather than throughput and
 * under-report peak by the pipeline depth. The accumulators are written out unconditionally so
 * nothing is dead code, and the multiplicands come from the thread index so ptxas cannot
 * constant-fold the chain. Cost is read as the slope against iteration count, so the kernel's
 * prologue is not charged to the arithmetic.
 *
 * Usage:
 *   ./gpu-fma-loop [--machine rtx-3090] [--device 0] [--iters N] [--repeats K] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-22
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpu_contention.cuh"

namespace {

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d -- %s\n",                    \
                         cudaGetErrorName(err_), __FILE__, __LINE__,                  \
                         cudaGetErrorString(err_));                                   \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

/// Independent FMA chains per thread. Enough to cover the pipeline depth on both GV100 and
/// GA102; a dependent chain would measure FMA latency instead of throughput.
constexpr int kUnroll = 16;

__global__ void k_fma64(double* out, long long iters)
{
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    double acc[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u)
        acc[u] = static_cast<double>(tid + u) * 1e-8;

    // Thread-derived, so the chain cannot be constant-folded.
    const double b = 1.0 + static_cast<double>(tid) * 1e-12;
    const double c = 1e-9 * static_cast<double>(tid & 31);

    for (long long i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            acc[u] = fma(acc[u], b, c);
    }

    double s = 0.0;
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) s += acc[u];
    out[tid] = s;   // unconditional: nothing here is dead code
}

__global__ void k_fma32(float* out, long long iters)
{
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    float acc[kUnroll];
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u)
        acc[u] = static_cast<float>(tid + u) * 1e-4f;

    const float b = 1.0f + static_cast<float>(tid) * 1e-7f;
    const float c = 1e-5f * static_cast<float>(tid & 31);

    for (long long i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            acc[u] = fmaf(acc[u], b, c);
    }

    float s = 0.0f;
    #pragma unroll
    for (int u = 0; u < kUnroll; ++u) s += acc[u];
    out[tid] = s;
}

template <typename F>
[[nodiscard]] double time_it(F&& f)
{
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return static_cast<double>(ms) * 1e-3;
}

[[nodiscard]] double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/**
 * @brief Peak FLOP/s as the slope of total time against iteration count.
 *
 * Differencing two iteration counts cancels the kernel prologue (launch, register allocation,
 * accumulator initialization), which a single division would charge to the arithmetic and so
 * under-report peak.
 */
template <typename Launch>
[[nodiscard]] double peak_flops(Launch&& launch, long long threads, long long lo, long long hi,
                                int repeats)
{
    (void)time_it([&] { launch(hi); });   // warm-up: clocks ramp, and the first launch is slow
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> f;
    f.reserve(static_cast<std::size_t>(repeats));
    for (int k = 0; k < repeats; ++k) {
        const double t_lo = time_it([&] { launch(lo); });
        const double t_hi = time_it([&] { launch(hi); });
        const double dt = t_hi - t_lo;
        // 2 FLOP per FMA, kUnroll chains per thread per iteration.
        const double flops = 2.0 * static_cast<double>(kUnroll)
                           * static_cast<double>(hi - lo) * static_cast<double>(threads);
        f.push_back(dt > 0.0 ? flops / dt : 0.0);
    }
    return median(f);
}

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    long long   iters   = 4000;
    int         repeats = 7;
    std::string csv_path;
};

[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                               std::exit(EXIT_FAILURE); }
            return argv[i];
        };
        if      (arg == "--machine") a.machine = next();
        else if (arg == "--device")  a.device  = std::stoi(next());
        else if (arg == "--iters")   a.iters   = std::stoll(next());
        else if (arg == "--repeats") a.repeats = std::stoi(next());
        else if (arg == "--csv")     a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./gpu-fma-loop [--machine KEY] [--device D] [--iters N]\n"
                "                      [--repeats K] [--csv PATH]\n\n"
                "  Measures FP64 and FP32 peak, and the ratio between them. The ratio is the\n"
                "  reading that matters: 1:2 on a datacenter part with dedicated FP64 units,\n"
                "  1:64 on consumer silicon. The entire negative arm of the GPU study is a\n"
                "  claim about that number, so it is measured rather than transcribed.\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.iters < 100) { std::fprintf(stderr, "--iters must be >= 100\n"); std::exit(1); }
    return a;
}

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(a.device));

    // Before any cudaMalloc: once this process has allocated, its own footprint is
    // indistinguishable from a tenant's.
    const DeviceContention contention = check_device_contention();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));

    // Full occupancy in the sense that matters here: enough warps resident to keep the FMA
    // pipes fed. The occupancy calculator sizes it rather than a guessed multiplier.
    const int threads = 256;
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, reinterpret_cast<const void*>(k_fma64), threads, 0));
    const int blocks = std::max(1, blocks_per_sm * prop.multiProcessorCount);
    const long long total_threads = static_cast<long long>(blocks) * threads;

    std::printf("GPU FMA peak -- the compute roof the roofline gate divides by\n");
    std::printf("  machine=%s  device=%d (%s)  CC=%d.%d\n",
                a.machine.c_str(), a.device, prop.name, prop.major, prop.minor);
    std::printf("  SMs=%d  blocks=%d x %d threads = %lld threads  unroll=%d\n",
                prop.multiProcessorCount, blocks, threads, total_threads, kUnroll);
    std::printf("  clocks: sm=%d MHz  mem=%d MHz\n",
                prop.clockRate / 1000, prop.memoryClockRate / 1000);
    report_toolkit();
    report_contention(contention);

    double* d_out64 = nullptr;
    float*  d_out32 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out64, static_cast<std::size_t>(total_threads) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out32, static_cast<std::size_t>(total_threads) * sizeof(float)));

    const long long lo = a.iters / 10;
    const long long hi = a.iters;

    const double f64 = peak_flops(
        [&](long long it) { k_fma64<<<blocks, threads>>>(d_out64, it);
                            CUDA_CHECK(cudaGetLastError()); },
        total_threads, lo, hi, a.repeats);

    const double f32 = peak_flops(
        [&](long long it) { k_fma32<<<blocks, threads>>>(d_out32, it);
                            CUDA_CHECK(cudaGetLastError()); },
        total_threads, lo, hi, a.repeats);

    const double ratio = f64 > 0.0 ? f32 / f64 : 0.0;

    std::printf("  FP64 peak : %8.3f TFLOP/s\n", f64 * 1e-12);
    std::printf("  FP32 peak : %8.3f TFLOP/s\n", f32 * 1e-12);
    std::printf("  FP32:FP64 : %8.1f : 1\n\n", ratio);

    // A part near 2:1 has dedicated FP64 units and the map's communication premise holds
    // comfortably; a part near 64:1 is the designed negative arm.
    if (ratio < 4.0)
        std::printf("  READING: dedicated FP64 (~1:2). FP64 is first-class here, so the map's\n"
                    "           communication premise holds and the kernels sit well below the\n"
                    "           ridge. This is the on-map arm.\n");
    else if (ratio > 32.0)
        std::printf("  READING: FP64 throttled (~1:%.0f). This is the negative arm. Note what it\n"
                    "           does and does not imply: Arnoldi's SpMV (AI 0.135) and MGS\n"
                    "           (0.375) are still below even this ridge, so the baseline stays\n"
                    "           on-map. What crosses is the tall-skinny Gram matrix that s-step\n"
                    "           adds (AI ~ s/4). Check gram_ridge_s() against the certified\n"
                    "           s_max before claiming the arm fires.\n", ratio);
    else
        std::printf("  READING: an intermediate ratio (1:%.0f), matching neither the 1:2\n"
                    "           datacenter class nor the 1:64 consumer one. Do not round it to\n"
                    "           whichever is nearer; record the measurement and re-derive the\n"
                    "           ridge from it.\n", ratio);
    std::printf("\n");

    if (may_record(contention)) {
        std::printf("To record, set these on the %s preset in include/gpu_machine.hpp:\n",
                    a.machine.c_str());
        std::printf("  fp64_flops_peak = %.4e\n", f64);
        std::printf("  fp32_flops_peak = %.4e\n", f32);
        std::printf("\nThese replace the datasheet figures. The ridge, and therefore every gate\n"
                    "verdict, is computed from them, which is why they are measured rather than\n"
                    "transcribed, exactly as the CPU side's peak_gflops_core is.\n");
    } else {
        report_suppressed();
    }

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device,device_name,cc_major,cc_minor,sm_count,blocks,threads,"
                        "unroll,iters_lo,iters_hi,repeats,fp64_flops,fp32_flops,fp32_fp64_ratio,"
                        "contended,device_used_bytes\n");
        std::fprintf(f, "%s,%d,\"%s\",%d,%d,%d,%d,%d,%d,%lld,%lld,%d,%.6e,%.6e,%.4f,%d,%zu\n",
                     a.machine.c_str(), a.device, prop.name, prop.major, prop.minor,
                     prop.multiProcessorCount, blocks, threads, kUnroll, lo, hi, a.repeats,
                     f64, f32, ratio, contention.contended ? 1 : 0, contention.other_bytes);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }

    CUDA_CHECK(cudaFree(d_out64));
    CUDA_CHECK(cudaFree(d_out32));
    return EXIT_SUCCESS;
}
