/**
 * @file calibrate_gpu_reduction.cu
 * @brief Offline measurement of the GPU reduction ladder: the numerator of R_h, and the gate
 *        on every horizontal-axis verdict on a GPU.
 *
 * The analogue of src/calibrate_alpha.cpp, and it inherits that binary's discipline: the
 * solver is not linked, no application payload is reduced, and the numbers are known before
 * any run they place. If this binary could see the method, R_h would risk becoming a measured
 * fraction of the runtime it is supposed to predict.
 *
 * What it measures, each as its own term rather than a fitted constant:
 *
 *   warp    __shfl_down_sync, register-to-register. The floor.
 *   block   + shared memory and __syncthreads across the warps of a block.
 *   grid    + a device-wide combine, measured with cooperative-groups grid.sync() so the
 *           on-die cost is isolated from the launch cost.
 *   launch  an empty kernel launch and sync. A fixed cost per reduction, independent of tree
 *           depth and with no CPU analogue. Reported separately because if it swamps the tiers
 *           then R_h's ladder collapses to a constant and the horizontal mechanism becomes
 *           uninteresting, which is a result in itself and invisible if launch is folded into
 *           the grid tier.
 *
 * The two-kernel form of the grid reduction is timed as well, since it is what an
 * implementation without cooperative launch actually calls. Reporting both is what lets the
 * study say whether the tier structure survives contact with the launch cost.
 *
 * The device-to-device and node tiers live in src/calibrate_gpu_p2p.cu, which needs NCCL and
 * MPI; this binary needs only CUDA, so the on-device rungs can be calibrated on any node.
 *
 * Each reduction is chained into the next contribution, scaled to 1e-18 as in
 * calibrate_alpha.cpp, so the dependency is real and neither ptxas nor the hardware can hoist
 * the combine out of the loop. Costs are read as the slope against iteration count rather than
 * a single division, which would charge the kernel's fixed prologue to the reduction.
 *
 * Usage:
 *   ./calibrate-gpu-reduction [--machine v100-pcie-16gb] [--device 0]
 *                             [--iters N] [--repeats K] [--block-threads T] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "gpu_contention.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int kWarpSize = 32;

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

// The reduction primitives, one per rung
/// Warp rung: a butterfly over the 32 lanes of one warp. No memory traffic at all.
__device__ __forceinline__ double warp_reduce(double v)
{
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return __shfl_sync(0xffffffffu, v, 0);   // broadcast: an all-reduce, as the model assumes
}

/// Block rung: warp butterflies, then one shared-memory pass over the warp leaders.
__device__ __forceinline__ double block_reduce(double v, double* smem)
{
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;

    v = warp_reduce(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    double acc = 0.0;
    // Every thread scans the warp partials, so the result is broadcast without a second
    // barrier. `warps` is at most 32, so this is cheaper than a tree plus a broadcast.
    for (int w = 0; w < warps; ++w) acc += smem[w];
    __syncthreads();   // ordering for the next iteration's write to smem
    return acc;
}

// The timed kernels. Each runs `iters` chained reductions inside one launch, so launch cost is
// amortized to nothing and appears only in the term that measures it deliberately.
__global__ void k_warp(double* out, long long iters)
{
    double contrib = 1.0 + static_cast<double>(threadIdx.x);
    for (long long i = 0; i < iters; ++i) {
        const double acc = warp_reduce(contrib);
        contrib = 1.0 + static_cast<double>(threadIdx.x) + acc * 1e-18;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = contrib;
}

__global__ void k_block(double* out, long long iters)
{
    extern __shared__ double smem[];
    double contrib = 1.0 + static_cast<double>(threadIdx.x);
    for (long long i = 0; i < iters; ++i) {
        const double acc = block_reduce(contrib, smem);
        contrib = 1.0 + static_cast<double>(threadIdx.x) + acc * 1e-18;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = contrib;
}

/**
 * Grid rung, cooperative form. Block partials land in `scratch`, grid.sync() orders them, then
 * every block scans. This is the on-die grid combine with no launch cost, which is the term
 * the model wants separated.
 *
 * Requires a cooperative launch, so the whole grid must be co-resident. main() sizes the grid
 * with cudaOccupancyMaxActiveBlocksPerMultiprocessor and refuses to launch otherwise: an
 * over-subscribed cooperative grid deadlocks rather than degrading.
 */
__global__ void k_grid_coop(double* out, double* scratch, long long iters)
{
    extern __shared__ double smem[];
    const cg::grid_group grid = cg::this_grid();
    const int nblocks = static_cast<int>(gridDim.x);

    double contrib = 1.0 + static_cast<double>(threadIdx.x);
    for (long long i = 0; i < iters; ++i) {
        const double b = block_reduce(contrib, smem);
        if (threadIdx.x == 0) scratch[blockIdx.x] = b;
        grid.sync();

        // Strided across the block, then one block reduce. Having every thread scan all
        // gridDim.x partials costs O(nblocks) global loads per thread, which at 492 blocks x
        // 125,952 threads is 62M loads per reduction and swamps the grid.sync() this rung
        // exists to measure: it read 297 us/reduction against 13 us for the two-kernel form,
        // so the scan was the measurement.
        double part = 0.0;
        for (int k = static_cast<int>(threadIdx.x); k < nblocks;
             k += static_cast<int>(blockDim.x))
            part += scratch[k];
        const double acc = block_reduce(part, smem);

        grid.sync();   // nobody may overwrite scratch until every block has read it
        contrib = 1.0 + static_cast<double>(threadIdx.x) + acc * 1e-18;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = contrib;
}

/// Two-kernel form, pass 1: block partials to global memory.
__global__ void k_partials(const double* in, double* scratch, long long n_unused)
{
    (void)n_unused;
    extern __shared__ double smem[];
    const double contrib = in ? in[blockIdx.x] : 1.0 + static_cast<double>(threadIdx.x);
    const double b = block_reduce(contrib, smem);
    if (threadIdx.x == 0) scratch[blockIdx.x] = b;
}

/// Two-kernel form, pass 2: one block folds the partials into the result.
__global__ void k_finalise(const double* scratch, double* out, int nblocks)
{
    extern __shared__ double smem[];
    double v = 0.0;
    for (int k = static_cast<int>(threadIdx.x); k < nblocks; k += static_cast<int>(blockDim.x))
        v += scratch[k];
    const double acc = block_reduce(v, smem);
    if (threadIdx.x == 0) *out = acc;
}

/// The launch-cost probe: does nothing, so what is timed is the launch and the sync.
__global__ void k_empty() {}

// Timing helpers
/// Wall time of a callable, in seconds, with the device synchronized on both sides.
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

struct Stat { double median = 0.0; double q1 = 0.0; double q3 = 0.0; };

[[nodiscard]] double quantile(std::vector<double> v, double q)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

[[nodiscard]] Stat summarize(const std::vector<double>& v)
{
    return { quantile(v, 0.5), quantile(v, 0.25), quantile(v, 0.75) };
}

/**
 * @brief Per-reduction cost as the slope of total time against iteration count.
 *
 * Two iteration counts, not one division. A kernel pays a prologue (launch, register
 * allocation, the first touch of shared memory) that a single division would charge to the
 * reduction, inflating the cheap rungs the ladder's shape depends on. Differencing two runs
 * cancels anything that does not scale with the loop.
 */
template <typename Launch>
[[nodiscard]] Stat slope_cost(Launch&& launch, long long iters_lo, long long iters_hi,
                              int repeats)
{
    (void)time_it([&] { launch(iters_hi); });   // warm-up, discarded

    std::vector<double> per;
    per.reserve(static_cast<std::size_t>(repeats));
    for (int k = 0; k < repeats; ++k) {
        const double t_lo = time_it([&] { launch(iters_lo); });
        const double t_hi = time_it([&] { launch(iters_hi); });
        per.push_back((t_hi - t_lo) / static_cast<double>(iters_hi - iters_lo));
    }
    return summarize(per);
}

// CLI
struct Args {
    std::string machine = "v100-pcie-16gb";
    int         device  = 0;
    long long   iters   = 20000;    ///< high arm; the low arm is iters/10
    int         repeats = 7;
    int         block_threads = 256;
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
        if      (arg == "--machine")       a.machine = next();
        else if (arg == "--device")        a.device  = std::stoi(next());
        else if (arg == "--iters")         a.iters   = std::stoll(next());
        else if (arg == "--repeats")       a.repeats = std::stoi(next());
        else if (arg == "--block-threads") a.block_threads = std::stoi(next());
        else if (arg == "--csv")           a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./calibrate-gpu-reduction [--machine KEY] [--device D]\n"
                "                                 [--iters N] [--repeats K]\n"
                "                                 [--block-threads T] [--csv PATH]\n\n"
                "  --machine KEY   preset name to print alongside the numbers, so a result is\n"
                "                  never recorded against the wrong card. Does NOT change what\n"
                "                  is measured: every figure here comes from the device.\n"
                "  --device D      CUDA device ordinal (default 0)\n"
                "  --iters N       chained reductions in the high arm (default 20000). The low\n"
                "                  arm is N/10; the cost is the slope between them.\n"
                "  --repeats K     timed repeats after one warm-up (default 7)\n"
                "  --block-threads T  threads per block (default 256). Report the value you\n"
                "                  used: the block rung's cost depends on the warps per block.\n"
                "  --csv PATH      write one row per rung (file truncated and re-headed)\n");
            std::exit(0);
        }
        else { std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str()); std::exit(EXIT_FAILURE); }
    }
    if (a.iters < 100)   { std::fprintf(stderr, "--iters must be >= 100\n"); std::exit(1); }
    if (a.repeats < 1)   { std::fprintf(stderr, "--repeats must be >= 1\n"); std::exit(1); }
    if (a.block_threads % kWarpSize != 0 || a.block_threads > 1024) {
        std::fprintf(stderr, "--block-threads must be a multiple of 32 and <= 1024\n");
        std::exit(1);
    }
    return a;
}

const char* kCsvHeader =
    "machine,device,device_name,sm_count,block_threads,blocks,coop_supported,"
    "iters_lo,iters_hi,repeats,"
    "tier,t_cumulative_s,t_cumulative_q1,t_cumulative_q3,t_incremental_s,multiplier,"
    "contended,device_used_bytes\n";

}  // namespace

int main(int argc, char** argv)
{
    const Args a = parse_args(argc, argv);

    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (a.device >= n_dev) {
        std::fprintf(stderr, "Device %d requested but only %d visible\n", a.device, n_dev);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(a.device));

    // Before any cudaMalloc: once this process has allocated, its own footprint is
    // indistinguishable from a tenant's.
    const DeviceContention contention = check_device_contention();

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, a.device));

    std::printf("GPU reduction calibration -- the tiers of R_h's numerator\n");
    std::printf("  machine=%s  device=%d (%s)\n", a.machine.c_str(), a.device, prop.name);
    std::printf("  SMs=%d  CC=%d.%d  L2=%.1f MiB  sharedPerSM=%.0f KiB  visible devices=%d\n",
                prop.multiProcessorCount, prop.major, prop.minor,
                static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0),
                static_cast<double>(prop.sharedMemPerMultiprocessor) / 1024.0, n_dev);
    std::printf("  block_threads=%d  iters=%lld (low arm %lld)  repeats=%d\n",
                a.block_threads, a.iters, a.iters / 10, a.repeats);
    report_toolkit();
    report_contention(contention);

    // Cooperative launch needs the whole grid co-resident, so the grid is sized to what the
    // occupancy calculator says fits. Over-subscribing a cooperative grid deadlocks.
    const std::size_t smem_bytes =
        static_cast<std::size_t>(a.block_threads / kWarpSize) * sizeof(double);

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, reinterpret_cast<const void*>(k_grid_coop), a.block_threads, smem_bytes));
    const int blocks = std::max(1, blocks_per_sm * prop.multiProcessorCount);

    const bool coop_supported = prop.cooperativeLaunch != 0 && blocks_per_sm > 0;
    std::printf("  co-resident blocks: %d/SM x %d SMs = %d   cooperative launch: %s\n\n",
                blocks_per_sm, prop.multiProcessorCount, blocks,
                coop_supported ? "supported" : "NOT SUPPORTED (grid rung will be skipped)");

    double* d_out = nullptr;
    double* d_scratch = nullptr;
    double* d_single = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(blocks) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<std::size_t>(blocks) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_single, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_scratch, 0, static_cast<std::size_t>(blocks) * sizeof(double)));

    const long long lo = a.iters / 10;
    const long long hi = a.iters;

    // Rung 1: warp
    const Stat t_warp = slope_cost(
        [&](long long it) {
            k_warp<<<blocks, a.block_threads>>>(d_out, it);
            CUDA_CHECK(cudaGetLastError());
        }, lo, hi, a.repeats);

    // Rung 2: block
    const Stat t_block = slope_cost(
        [&](long long it) {
            k_block<<<blocks, a.block_threads, smem_bytes>>>(d_out, it);
            CUDA_CHECK(cudaGetLastError());
        }, lo, hi, a.repeats);

    // Rung 3: grid, on-die, launch-free
    Stat t_grid{};
    if (coop_supported) {
        t_grid = slope_cost(
            [&](long long it) {
                void* args[] = { &d_out, &d_scratch, &it };
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    reinterpret_cast<const void*>(k_grid_coop),
                    dim3(static_cast<unsigned>(blocks)),
                    dim3(static_cast<unsigned>(a.block_threads)),
                    args, smem_bytes, nullptr));
            }, lo, hi, a.repeats);
    }

    // The launch term: its own cost, not part of any rung.
    const Stat t_launch = slope_cost(
        [&](long long it) {
            for (long long i = 0; i < it; ++i) { k_empty<<<blocks, a.block_threads>>>(); }
            CUDA_CHECK(cudaGetLastError());
        }, lo / 20, hi / 20, a.repeats);

    // The two-kernel grid reduction: what an implementation without cooperative launch calls.
    const Stat t_two_kernel = slope_cost(
        [&](long long it) {
            for (long long i = 0; i < it; ++i) {
                k_partials<<<blocks, a.block_threads, smem_bytes>>>(nullptr, d_scratch, 0);
                k_finalise<<<1, a.block_threads, smem_bytes>>>(d_scratch, d_single, blocks);
            }
            CUDA_CHECK(cudaGetLastError());
        }, lo / 20, hi / 20, a.repeats);

    // Report
    const double inc_warp  = t_warp.median;
    const double inc_block = t_block.median - t_warp.median;
    const double inc_grid  = coop_supported ? t_grid.median - t_block.median : 0.0;

    std::printf("  %-12s %14s %14s %14s %14s\n",
                "tier", "cumulative", "q1", "q3", "incremental");
    std::printf("  %s\n", std::string(72, '-').c_str());
    std::printf("  %-12s %11.1f ns %11.1f ns %11.1f ns %11.1f ns\n", "warp",
                t_warp.median * 1e9, t_warp.q1 * 1e9, t_warp.q3 * 1e9, inc_warp * 1e9);
    std::printf("  %-12s %11.1f ns %11.1f ns %11.1f ns %11.1f ns\n", "block",
                t_block.median * 1e9, t_block.q1 * 1e9, t_block.q3 * 1e9, inc_block * 1e9);
    if (coop_supported)
        std::printf("  %-12s %11.1f ns %11.1f ns %11.1f ns %11.1f ns\n", "grid (coop)",
                    t_grid.median * 1e9, t_grid.q1 * 1e9, t_grid.q3 * 1e9, inc_grid * 1e9);
    else
        std::printf("  %-12s %s\n", "grid (coop)", "SKIPPED: cooperative launch unsupported");
    std::printf("  %s\n", std::string(72, '-').c_str());
    std::printf("  %-12s %11.1f ns   (its own term; no CPU analogue)\n",
                "launch", t_launch.median * 1e9);
    std::printf("  %-12s %11.1f ns   (partials + finalise, launches included)\n",
                "grid (2-kern)", t_two_kernel.median * 1e9);
    std::printf("\n");

    // The two readings the tier model lives or dies by.
    if (coop_supported && inc_block > 0.0)
        std::printf("  crossing multipliers:  block/warp = %.1fx   grid/block = %.1fx\n",
                    t_warp.median > 0.0 ? t_block.median / t_warp.median : 0.0,
                    t_block.median > 0.0 ? t_grid.median / t_block.median : 0.0);
    if (t_two_kernel.median > 0.0) {
        const double share = 2.0 * t_launch.median / t_two_kernel.median;
        std::printf("  launch share of the two-kernel grid reduction: %.0f%%\n", share * 100.0);
        if (share > 0.75)
            std::printf("  WARNING: launch dominates. R_h's ladder collapses toward a constant\n"
                        "           from the grid rung upward. Record this as a finding, not as\n"
                        "           a calibration nuisance.\n");
    }
    /*
     * Which grid form the model should carry.
     *
     * Two ways to reduce across the grid, and they are not close:
     *
     *   cooperative  grid.sync() inside one long-lived kernel. No launch cost, because the
     *                kernel is launched once and reduces many times inside it.
     *   two-kernel   partials kernel + finalize kernel. Pays two launches per reduction.
     *
     * The cooperative form sounds cheaper, and was introduced here to isolate the on-die
     * combine from launch overhead, but measurement says otherwise: two grid-wide barriers
     * plus a second combine phase cost more than two kernel launches. A real implementation
     * would therefore call the two-kernel form, and R_h's non-circularity rests on the
     * calibrated constant being the slope of code that actually runs. The recommendation below
     * comes from whichever form is cheaper, with the cooperative figure kept in the CSV as the
     * measurement that justified the choice.
     *
     * The subtraction reproduces the model: reduction_cost_s(GRID) accumulates warp + block +
     * grid and adds t_kernel_launch_s once, so t_reduce_s[GRID] is set to whatever makes that
     * sum equal the measured best-available grid reduction. Launch stays its own term rather
     * than being folded in, thus the question being it swamps the ladder stays answerable.
     */
    const double grid_coop_cost = coop_supported ? t_grid.median
                                                 : std::numeric_limits<double>::infinity();
    const bool   two_kernel_wins = t_two_kernel.median < grid_coop_cost;
    const double grid_best = std::min(grid_coop_cost, t_two_kernel.median);
    // Model form: warp + block + GRID + launch == grid_best.
    const double inc_grid_model =
        std::max(0.0, grid_best - t_block.median - t_launch.median);

    if (coop_supported)
        std::printf("  grid form: two-kernel %.1f us vs cooperative %.1f us -- %s is cheaper,\n"
                    "             so the model carries %s.\n",
                    t_two_kernel.median * 1e6, grid_coop_cost * 1e6,
                    two_kernel_wins ? "two-kernel" : "cooperative",
                    two_kernel_wins ? "it" : "the cooperative form");
    std::printf("\n");

    if (may_record(contention)) {
        std::printf("To record, set these on the %s preset in include/gpu_machine.hpp:\n",
                    a.machine.c_str());
        std::printf("  t_reduce_s[WARP]   = %.4e\n", inc_warp);
        std::printf("  t_reduce_s[BLOCK]  = %.4e\n", inc_block);
        if (coop_supported || t_two_kernel.median > 0.0) {
            std::printf("  t_reduce_s[GRID]   = %.4e\n", inc_grid_model);
            std::printf("      (from the %s form: %.4e total, minus block %.4e,\n"
                        "       minus the launch term %.4e, which the model adds back)\n",
                        two_kernel_wins ? "two-kernel" : "cooperative",
                        grid_best, t_block.median, t_launch.median);
        } else {
            std::printf("  t_reduce_s[GRID]   = (unmeasured; leave 0, tier_calibrated false)\n");
        }
        std::printf("  t_kernel_launch_s  = %.4e\n", t_launch.median);
        std::printf("  tier_calibrated[WARP|BLOCK%s] = true\n", coop_supported ? "|GRID" : "");
        std::printf("\nOn a multi-GPU host the DEVICE_P2P and NODE rungs come from "
                    "./calibrate-gpu-p2p, and\nreduction_calibrated stays false until they "
                    "land. On a single-GPU host those rungs are\nunreachable, so these three "
                    "are the whole ladder and the flag may go true here.\n");
    } else {
        report_suppressed();
    }

    if (!a.csv_path.empty()) {
        std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return 1; }
        std::fprintf(f, "%s", kCsvHeader);
        auto row = [&](const char* tier, const Stat& s, double inc, double mult) {
            std::fprintf(f,
                         "%s,%d,\"%s\",%d,%d,%d,%d,%lld,%lld,%d,%s,%.9e,%.9e,%.9e,%.9e,%.4f,"
                         "%d,%zu\n",
                         a.machine.c_str(), a.device, prop.name, prop.multiProcessorCount,
                         a.block_threads, blocks, coop_supported ? 1 : 0, lo, hi, a.repeats,
                         tier, s.median, s.q1, s.q3, inc, mult,
                         contention.contended ? 1 : 0, contention.other_bytes);
        };
        row("warp",  t_warp,  inc_warp,  0.0);
        row("block", t_block, inc_block,
            t_warp.median > 0.0 ? t_block.median / t_warp.median : 0.0);
        if (coop_supported)
            row("grid", t_grid, inc_grid,
                t_block.median > 0.0 ? t_grid.median / t_block.median : 0.0);
        row("launch", t_launch, t_launch.median, 0.0);
        row("grid_two_kernel", t_two_kernel, t_two_kernel.median, 0.0);
        // The value the preset should carry, and which form it came from. The cooperative and
        // two-kernel rows above are the evidence; this row is the verdict.
        row(two_kernel_wins ? "grid_model_two_kernel" : "grid_model_coop",
            Stat{grid_best, grid_best, grid_best}, inc_grid_model,
            t_block.median > 0.0 ? grid_best / t_block.median : 0.0);
        std::fclose(f);
        std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_single));
    return EXIT_SUCCESS;
}
