/**
 * @file gpu_stream.cu
 * @brief Achieved memory roofs on a GPU, and the occupancy sweep the CPU study had no
 *        counterpart for.
 *
 * The analogue of src/stream.c, and it supplies two things the map cannot be read without:
 *
 *   1. The achieved bandwidth the roofline gate divides by. A gate evaluated against a
 *      theoretical roof over-states the memory side and would wave a compute-bound point onto
 *      a map that cannot chart it, always in the unsafe direction. Until this binary has run,
 *      `roofline_gated` stays false and every gate verdict is provisional.
 *   2. The L2 roof, separately from the HBM roof. The CPU model priced kernels against L3 or
 *      DRAM depending on residency; the GPU needs the same two-roof treatment, and 6 MiB is
 *      small enough that the distinction bites at grid sizes the application uses.
 *
 * The size sweep separates them: the triad runs from well inside 6 MiB to well past it, and
 * the two plateaus are read off directly rather than inferred. An L2-resident point is only
 * meaningful once the first pass has faulted the data in, so the resident sizes are looped
 * many times and the warm-up discarded; otherwise what is measured is HBM with extra steps.
 *
 * The occupancy sweep asks whether the kernel is bandwidth-bound or occupancy-bound, the GPU
 * analogue of the CPU study's SSE-vs-AVX2 finding. It matters because shared-memory tiling for
 * matrix-powers trades against occupancy, and that tradeoff has to be characterized before a
 * tiling depth can be chosen. If throughput proves independent of occupancy above some
 * threshold the map stays two-dimensional; if not, occupancy is a third axis.
 *
 * Byte accounting matches src/stream.c: 24 B per element for the triad (read b, read c,
 * write a), so the GPU and CPU roofs are directly comparable.
 *
 * Usage:
 *   ./gpu-stream [--machine v100-pcie-16gb] [--device 0] [--repeats K]
 *                [--max-mib M] [--csv PATH] [--occupancy-csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-21
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

/// Bytes charged per element, matching src/stream.c's Triad so the two roofs compare.
constexpr double kTriadBytesPerElem = 24.0;

/**
 * @brief `reps` triads inside one launch, rotating the three buffers between repetitions.
 *
 * Two things this shape defends against.
 *
 * Launch cost. An L2-resident size moves so few bytes (3 MiB at ~1 TB/s is ~3 us) that a
 * ~2.5 us kernel launch is a comparable term, so repeating by re-launching measures launch
 * latency and reports it as bandwidth. Looping inside the kernel amortizes the launch to
 * nothing.
 *
 * Dead-store elimination. With `a` marked __restrict__ and never read, a compiler may keep
 * only the last repetition's store and discard the rest, turning `reps` into 1. Rotating the
 * buffers makes each repetition's output the next one's input, so the chain is a real
 * dependency and nothing can be dropped.
 *
 * @note Values may saturate to +/-inf over many repetitions. That is harmless: NVIDIA GPUs
 *       process IEEE specials at full rate, and this kernel's byte traffic is independent of
 *       the values moved.
 */
__global__ void k_triad(double* p0, double* p1, double* p2, double scalar, long long n,
                        int reps)
{
    double* arr[3] = { p0, p1, p2 };
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    const long long base   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (int r = 0; r < reps; ++r) {
        // No __restrict__ here. The rotation means the array written this repetition is read
        // the next one, so a restrict qualifier would be a false promise and would license the
        // compiler to reorder or elide the traffic being measured.
        double*       a = arr[r % 3];
        const double* b = arr[(r + 1) % 3];
        const double* c = arr[(r + 2) % 3];
        for (long long i = base; i < n; i += stride)
            a[i] = b[i] + scalar * c[i];

        // Compiler barrier, zero runtime cost. At the small sizes each thread owns roughly one
        // element, and without this the compiler unrolls the rotation, keeps all three values
        // in registers, and the repetitions move no memory at all, reporting register
        // bandwidth as cache bandwidth.
        asm volatile("" ::: "memory");
    }
}

__global__ void k_fill(double* p, double v, long long n)
{
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        p[i] = v;
}

template <typename F>
[[nodiscard]] double time_it(F&& f)
{
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    f();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return static_cast<double>(ms) * 1e-3;
}

[[nodiscard]] double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/**
 * @brief Best achieved GB/s for `n` elements, at a given launch geometry.
 *
 * `inner` repetitions happen inside one launch (see k_triad), so a resident working set stays
 * resident for the whole measurement and launch cost is amortized rather than charged to
 * bandwidth. `inner` scales down as the arrays grow, since a 2 GiB array has nothing to keep
 * warm.
 *
 * The grid is capped at what the problem needs. A grid sized for 2 GiB launches ~672k threads;
 * at 131k elements most exit immediately having done nothing, so the launch is paid in full
 * for a fraction of the work.
 */
[[nodiscard]] double triad_gbs(double* a, double* b, double* c, long long n,
                               int blocks, int threads, int repeats, int inner)
{
    const double scalar = 3.0;   // STREAM convention, so the roofs compare with src/stream.c
    const long long needed = (n + threads - 1) / threads;
    const int eff_blocks = static_cast<int>(
        std::min(static_cast<long long>(blocks), std::max(1LL, needed)));

    auto run = [&] { k_triad<<<eff_blocks, threads>>>(a, b, c, scalar, n, inner); };
    (void)time_it(run);   // warm-up: first touch, and the pass that makes L2 residency real
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> gbs;
    gbs.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const double t = time_it(run);
        const double bytes = kTriadBytesPerElem * static_cast<double>(n)
                           * static_cast<double>(inner);
        gbs.push_back(t > 0.0 ? bytes / t * 1e-9 : 0.0);
    }
    return median(gbs);
}

struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    int         repeats = 7;
    long long   max_mib = 2048;   ///< largest array, per array
    std::string csv_path;
    std::string occ_csv_path;
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
        if      (arg == "--machine")       a.machine = next();
        else if (arg == "--device")        a.device  = std::stoi(next());
        else if (arg == "--repeats")       a.repeats = std::stoi(next());
        else if (arg == "--max-mib")       a.max_mib = std::stoll(next());
        else if (arg == "--csv")           a.csv_path = next();
        else if (arg == "--occupancy-csv") a.occ_csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./gpu-stream [--machine KEY] [--device D] [--repeats K]\n"
                "                    [--max-mib M] [--csv PATH] [--occupancy-csv PATH]\n\n"
                "  --max-mib M     largest array in MiB, per array, three arrays allocated\n"
                "                  (default 2048 = 6 GiB total). Must clear L2 by >= 4x for\n"
                "                  the HBM plateau to be real, the same rule src/stream.c uses.\n"
                "  --csv PATH      size sweep: one row per array size\n"
                "  --occupancy-csv PATH  occupancy sweep: one row per (blocks/SM, threads)\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(a.device));

    // Before any cudaMalloc. This binary is the most contention-sensitive of the three: a
    // tenant competes for the exact resource being measured.
    const DeviceContention contention = check_device_contention();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));

    const double l2_mib = static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0);
    std::printf("GPU STREAM triad -- the achieved roofs the roofline gate divides by\n");
    std::printf("  machine=%s  device=%d (%s)\n", a.machine.c_str(), a.device, prop.name);
    std::printf("  SMs=%d  L2=%.1f MiB  memory=%.1f GiB  theoretical BW=%.0f GB/s\n",
                prop.multiProcessorCount, l2_mib,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                2.0 * static_cast<double>(prop.memoryClockRate) * 1e3
                    * (static_cast<double>(prop.memoryBusWidth) / 8.0) * 1e-9);
    std::printf("  charging %.0f B/element, matching src/stream.c's Triad\n",
                kTriadBytesPerElem);
    report_toolkit();
    report_contention(contention);

    const long long n_max = a.max_mib * 1024 * 1024 / 8;
    double *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;
    CUDA_CHECK(cudaMalloc(&a_d, static_cast<std::size_t>(n_max) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b_d, static_cast<std::size_t>(n_max) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c_d, static_cast<std::size_t>(n_max) * sizeof(double)));

    const int threads_default = 256;
    const int blocks_default  = prop.multiProcessorCount * 32;
    // All three, not just b and c: k_triad rotates the buffers, so every array is read as an
    // input on some repetition and an uninitialized one would feed NaNs through the chain.
    k_fill<<<blocks_default, threads_default>>>(a_d, 0.5, n_max);
    k_fill<<<blocks_default, threads_default>>>(b_d, 1.0, n_max);
    k_fill<<<blocks_default, threads_default>>>(c_d, 2.0, n_max);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Three tiers, not two, because aggregate L1 exceeds L2 on this hardware class: on GA102
    // that is 82 SMs x 100 KiB = 8.2 MB of L1 against a 6 MiB L2, so a footprint that has
    // spilled L2 may still sit entirely in L1 and read several TB/s. src/stream.c's rule of
    // clearing the last level by 4x therefore applies to the whole hierarchy, not to L2 alone.
    // Judging on L2 alone put the DRAM roof at 3299 GB/s, roughly 5x the true figure, and
    // every roofline verdict downstream inherits that.
    const long long l1_total =
        static_cast<long long>(prop.multiProcessorCount) * prop.sharedMemPerMultiprocessor;
    const long long cache_total = prop.l2CacheSize + l1_total;

    std::printf("  cache hierarchy: L2 %.1f MiB + aggregate L1 %.1f MiB = %.1f MiB\n",
                static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0),
                static_cast<double>(l1_total) / (1024.0 * 1024.0),
                static_cast<double>(cache_total) / (1024.0 * 1024.0));
    std::printf("  a size is DRAM-clean only past 4x that, i.e. a %.0f MiB footprint\n\n",
                4.0 * static_cast<double>(cache_total) / (1024.0 * 1024.0));

    enum class Tier { RESIDENT, PARTIAL, SPILLED };
    auto tier_of = [&](long long mib) {
        const long long footprint = 3 * mib * 1024 * 1024;
        if (footprint <= prop.l2CacheSize)   return Tier::RESIDENT;
        if (footprint <= 4 * cache_total)    return Tier::PARTIAL;
        return Tier::SPILLED;
    };
    auto tier_name = [](Tier t) {
        return t == Tier::RESIDENT ? "L2-resident" : (t == Tier::PARTIAL ? "partial" : "DRAM");
    };

    std::printf("  %14s %12s %14s %14s\n", "array (MiB)", "elements", "GB/s", "served by");
    std::printf("  %s\n", std::string(60, '-').c_str());

    struct Row { double mib; long long n; double gbs; Tier tier; };
    std::vector<Row> rows;

    for (long long mib = 1; mib <= a.max_mib; mib *= 2) {
        const long long n = mib * 1024 * 1024 / 8;
        const Tier t = tier_of(mib);
        // A cached set needs many passes to be genuinely warm; a 2 GiB set has nothing to
        // keep warm and would only waste wall-clock.
        const int inner = t == Tier::RESIDENT ? 200 : (mib < 64 ? 20 : 4);
        const double gbs = triad_gbs(a_d, b_d, c_d, n, blocks_default, threads_default,
                                     a.repeats, inner);
        rows.push_back({static_cast<double>(mib), n, gbs, t});
        std::printf("  %14.0f %12lld %14.1f %14s\n", static_cast<double>(mib), n, gbs,
                    tier_name(t));
    }
    std::printf("  %s\n\n", std::string(60, '-').c_str());

    double l2_peak = 0.0;
    std::vector<double> dram;
    for (const Row& r : rows) {
        if (r.tier == Tier::RESIDENT) l2_peak = std::max(l2_peak, r.gbs);
        if (r.tier == Tier::SPILLED)  dram.push_back(r.gbs);
    }
    // Median of the DRAM-clean sizes, not the maximum. A roof used to price a kernel should be
    // what the device sustains, and the max over a handful of samples is biased upward by
    // whichever one got the quietest slice of a shared card. The max is printed alongside so
    // the spread stays visible.
    if (dram.empty()) {
        std::printf("  WARNING: no size cleared the cache hierarchy by 4x. Raise --max-mib to\n"
                    "           at least %.0f; the DRAM roof below is NOT a DRAM roof.\n",
                    12.0 * static_cast<double>(cache_total) / (1024.0 * 1024.0));
        for (const Row& r : rows) dram.push_back(r.gbs);
    }
    const double hbm_clean = median(dram);
    double hbm_max = 0.0;
    for (const double g : dram) hbm_max = std::max(hbm_max, g);

    std::printf("  cache-resident peak : %8.1f GB/s   (L2 and aggregate L1 together)\n",
                l2_peak);
    std::printf("  DRAM roof (sustained): %8.1f GB/s   (median of DRAM-clean sizes; "
                "best single %.1f)\n", hbm_clean, hbm_max);
    if (l2_peak > 0.0 && hbm_clean > 0.0)
        std::printf("  cache/DRAM ratio    : %8.2fx  -- the gap the vertical mechanism "
                    "converts\n", l2_peak / hbm_clean);
    std::printf("\n");

    // Occupancy sweep: is the kernel bandwidth-bound or occupancy-bound?
    const long long n_occ = std::min(n_max, 512LL * 1024 * 1024 / 8);   // 512 MiB: well spilled
    std::printf("  occupancy sweep at %lld MiB (spilled), blocks as a multiple of SM count\n",
                n_occ * 8 / (1024 * 1024));
    std::printf("  %10s %10s %14s %10s\n", "threads", "blocks/SM", "GB/s", "% of peak");
    std::printf("  %s\n", std::string(50, '-').c_str());

    struct OccRow { int threads; int bps; double gbs; };
    std::vector<OccRow> occ;
    for (const int threads : {64, 128, 256, 512, 1024}) {
        for (const int bps : {1, 2, 4, 8, 16, 32}) {
            const int blocks = prop.multiProcessorCount * bps;
            const double gbs = triad_gbs(a_d, b_d, c_d, n_occ, blocks, threads, a.repeats, 4);
            occ.push_back({threads, bps, gbs});
            std::printf("  %10d %10d %14.1f %9.0f%%\n", threads, bps, gbs,
                        hbm_clean > 0.0 ? gbs / hbm_clean * 100.0 : 0.0);
        }
    }
    std::printf("  %s\n\n", std::string(50, '-').c_str());

    // Two spreads, because they answer different questions. The full sweep includes the
    // starved corner (64 threads x 1 block/SM = 5,248 threads, far too few to cover memory
    // latency), which no kernel would be launched with. Whether occupancy is a third map axis
    // turns on how much throughput varies across geometries a real kernel might choose, so
    // that range is reported separately and is the one to read.
    double best = 0.0, worst = 1e30, u_best = 0.0, u_worst = 1e30;
    for (const OccRow& r : occ) {
        best  = std::max(best, r.gbs);
        worst = std::min(worst, r.gbs);
        if (r.threads >= 128 && r.bps >= 2) {
            u_best  = std::max(u_best, r.gbs);
            u_worst = std::min(u_worst, r.gbs);
        }
    }
    const double spread   = worst > 0.0 ? best / worst : 0.0;
    const double u_spread = u_worst > 0.0 && u_worst < 1e29 ? u_best / u_worst : 0.0;

    std::printf("  occupancy spread, whole sweep      : %.2fx  (includes the starved corner)\n",
                spread);
    std::printf("  occupancy spread, usable geometries: %.2fx  (>=128 threads, >=2 blocks/SM)\n",
                u_spread);
    if (u_spread > 0.0 && u_spread < 1.25)
        std::printf("  READING: flat wherever a real kernel would live -- bandwidth-bound, and\n"
                    "           the map stays two-dimensional. Shared-memory tiling may spend\n"
                    "           occupancy freely (the direct analogue of the CPU study's\n"
                    "           SSE=AVX2 finding). The starved corner is a launch-configuration\n"
                    "           artifact, not a third axis.\n");
    else
        std::printf("  READING: throughput varies by %.2fx even across usable geometries.\n"
                    "           Occupancy is not free, the tiling depth trades against it, and\n"
                    "           the third-axis risk is live. State it; do not hide it by fixing\n"
                    "           occupancy at one value.\n", u_spread);
    std::printf("\n");

    if (may_record(contention)) {
        std::printf("To record, set these on the %s preset in include/gpu_machine.hpp:\n",
                    a.machine.c_str());
        std::printf("  hbm_bw_gbs_achieved = %.1f\n", hbm_clean);
        std::printf("  l2_bw_gbs_achieved  = %.1f\n", l2_peak);
        std::printf("  roofline_gated      = true\n");
    } else {
        report_suppressed();
        std::printf("This binary is the one contention hurts most: a co-tenant competes for\n"
                    "exactly the bandwidth being measured, so an under-read here would lower\n"
                    "the achieved roof, raise the computed ridge, and wrongly declare kernels\n"
                    "memory-bound, waving compute-bound points onto a map that cannot chart\n"
                    "them. That is the unsafe direction.\n");
    }

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device,device_name,sm_count,l2_bytes,array_mib,elements,"
                        "tier,gbs,cache_peak_gbs,dram_roof_gbs,dram_best_gbs,"
                        "contended,device_used_bytes\n");
        for (const Row& r : rows)
            std::fprintf(f, "%s,%d,\"%s\",%d,%d,%.0f,%lld,%s,%.4f,%.4f,%.4f,%.4f,%d,%zu\n",
                         a.machine.c_str(), a.device, prop.name, prop.multiProcessorCount,
                         prop.l2CacheSize, r.mib, r.n, tier_name(r.tier), r.gbs,
                         l2_peak, hbm_clean, hbm_max, contention.contended ? 1 : 0,
                         contention.other_bytes);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }

    if (!a.occ_csv_path.empty()) {
        std::FILE* f = std::fopen(a.occ_csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.occ_csv_path.c_str()); return 1; }
        std::fprintf(f, "machine,device,device_name,sm_count,elements,threads,blocks_per_sm,"
                        "blocks,gbs,dram_roof_gbs,spread_all,spread_usable,"
                        "contended,device_used_bytes\n");
        for (const OccRow& r : occ)
            std::fprintf(f, "%s,%d,\"%s\",%d,%lld,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%d,%zu\n",
                         a.machine.c_str(), a.device, prop.name, prop.multiProcessorCount,
                         n_occ, r.threads, r.bps, prop.multiProcessorCount * r.bps,
                         r.gbs, hbm_clean, spread, u_spread, contention.contended ? 1 : 0,
                         contention.other_bytes);
        std::fclose(f);
        std::printf("  [Wrote %s]\n", a.occ_csv_path.c_str());
    }

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));
    return EXIT_SUCCESS;
}
