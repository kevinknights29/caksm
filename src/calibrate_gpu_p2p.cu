/**
 * @file calibrate_gpu_p2p.cu
 * @brief The top two rungs of the reduction ladder: device-to-device and node-to-node.
 *
 * The companion to src/calibrate_gpu_reduction.cu, split off because these rungs need NCCL
 * while the on-device rungs need only CUDA. The on-device ladder can therefore be calibrated
 * on any node, and a missing NCCL does not block Phase A.
 *
 * These two rungs are the reason synge is in the study. A single V100's reduction is on-die
 * and near-free, so a single GPU can test theta_v but cannot test the horizontal mechanism or
 * the Upper-Right corner at all. Holding N and the device fixed and changing only which rung
 * the reduction crosses traverses the horizontal axis directly, which turns theta_h from a
 * shaded prediction into a measured crossing.
 *
 * Two modes, following synge's actual layout:
 *
 *   single-process, multi-device (DEVICE_P2P).  Both V100s sit on one node, so
 *       `ncclCommInitAll` drives both from one process with no MPI involved. Phase 0 predicts
 *       this is the rung that carries the operator at production resolution into Upper-Right,
 *       and it is reachable with nothing but CUDA and NCCL in a `-N 1` allocation.
 *
 *   MPI, one rank per node (NODE).  Compiled only when MPI is found (CAKSM_HAVE_MPI). The
 *       fabric rung needs a launcher, and it is the one rung whose absence does not block the
 *       headline result.
 *
 * NCCL, not a hand-rolled ring, in either mode: the reduction timed must be the one a real
 * implementation would call. A bespoke P2P reduction would measure a primitive nobody uses,
 * and R_h's non-circularity rests on the calibrated constant being the slope of code that
 * actually runs.
 *
 * On synge the intra-node link is `SYS`, PCIe plus a cross-socket UPI hop with no NVLink, so
 * the DEVICE_P2P rung is a genuine cross-socket crossing rather than a token one.
 *
 * Usage:
 *   # DEVICE_P2P: no launcher, no MPI, one node
 *   ./calibrate-gpu-p2p --tier device-p2p --csv ...
 *   # NODE: one rank per node (needs an MPI build)
 *   srun -N 2 -n 2 --gpus-per-node=1 ./calibrate-gpu-p2p --tier node --csv ...
 *
 * @author Kevin Knights
 * @date 2026-07-21
 */

#include <cuda_runtime.h>
#include <nccl.h>

#ifdef CAKSM_HAVE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gpu_contention.cuh"

namespace {

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

[[noreturn]] void die(const char* what)
{
    std::fprintf(stderr, "%s\n", what);
#ifdef CAKSM_HAVE_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d -- %s\n",                    \
                         cudaGetErrorName(err_), __FILE__, __LINE__,                  \
                         cudaGetErrorString(err_));                                   \
            die("aborting");                                                          \
        }                                                                             \
    } while (0)

#define NCCL_CHECK(call)                                                              \
    do {                                                                              \
        const ncclResult_t res_ = (call);                                             \
        if (res_ != ncclSuccess) {                                                    \
            std::fprintf(stderr, "NCCL error at %s:%d -- %s\n", __FILE__, __LINE__,   \
                         ncclGetErrorString(res_));                                    \
            die("aborting");                                                          \
        }                                                                             \
    } while (0)

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

struct Args {
    std::string machine = "v100-pcie-16gb";
    std::string tier    = "auto";
    long long   iters   = 2000;
    long long   halo_iters = 200;
    int         repeats = 7;
    long long   bw_bytes = 64LL << 20;
    bool        production = false;
    std::vector<int> devices{0};
    std::string csv_path;
};

[[nodiscard]] std::vector<int> parse_devices(const std::string& text)
{
    std::vector<int> devices;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string token =
            text.substr(begin, end == std::string::npos
                ? std::string::npos : end - begin);
        if (token.empty()) die("--devices contains an empty device index");
        const int device = std::stoi(token);
        if (device < 0) die("--devices indices must be non-negative");
        if (std::find(devices.begin(), devices.end(), device) != devices.end())
            die("--devices must not contain duplicates");
        devices.push_back(device);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (devices.empty()) die("--devices requires at least one device");
    return devices;
}

[[nodiscard]] std::string device_list(const std::vector<int>& devices)
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) stream << '+';
        stream << devices[i];
    }
    return stream.str();
}

[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) die(("Missing value for " + arg).c_str());
            return argv[i];
        };
        if      (arg == "--machine")  a.machine  = next();
        else if (arg == "--tier")     a.tier     = next();
        else if (arg == "--iters")    a.iters    = std::stoll(next());
        else if (arg == "--halo-iters") a.halo_iters = std::stoll(next());
        else if (arg == "--repeats")  a.repeats  = std::stoi(next());
        else if (arg == "--bw-bytes") a.bw_bytes = std::stoll(next());
        else if (arg == "--devices")  a.devices  = parse_devices(next());
        else if (arg == "--production") a.production = true;
        else if (arg == "--csv")      a.csv_path = next();
        else if (arg == "--help") {
            std::printf(
                "Usage: ./calibrate-gpu-p2p [--machine KEY] [--tier auto|device-p2p|node]\n"
                "                           [--iters N] [--halo-iters N] [--repeats K]\n"
                "                           [--bw-bytes B] [--production]\n"
                "                           [--devices ID[,ID]]\n"
                "                           [--csv PATH]\n\n"
                "  DEVICE_P2P needs no launcher and no MPI: both V100s are on one node, so\n"
                "  ncclCommInitAll drives them from a single process. Just run the binary.\n"
                "  NODE needs an MPI build and one rank per node, under srun.\n\n"
                "  --production measures the CA integrator's norm, projection, Gram, and\n"
                "               nearest-neighbor halo payloads. Launch exactly one MPI process\n"
                "               per node and use --devices 0 for two participants or\n"
                "               --devices 0,1 for four. This preserves the solver's host-side\n"
                "               NCCL submission topology.\n"
                "  --tier T     which rung you expect to measure. The binary detects the truth\n"
                "               from the launch shape and errors if they disagree, so a\n"
                "               mis-specified geometry cannot record a node-tier latency\n"
                "               against the intra-node slot.\n"
                "  --iters N    chained all-reduces per timed repeat (default 2000). The\n"
                "               latency arm reduces one double: R_h's numerator is a latency,\n"
                "               not a bandwidth, so the payload must not hide the fixed cost.\n"
                "  --halo-iters N grouped halo exchanges per timed repeat in production mode\n"
                "               (default 200).\n"
                "  --bw-bytes B payload for the separate bandwidth arm (default 64 MiB), which\n"
                "               feeds interconnect_bw_gbs and R_v's multi-device extension.\n");
            std::exit(0);
        }
        else die(("Unknown flag: " + arg).c_str());
    }
    if (a.tier != "auto" && a.tier != "device-p2p" && a.tier != "node")
        die("--tier must be auto, device-p2p, or node");
    if (a.iters < 1 || a.halo_iters < 1)
        die("--iters and --halo-iters must be positive");
    if (a.repeats < 1) die("--repeats must be positive");
    if (a.bw_bytes < static_cast<long long>(sizeof(double))
        || a.bw_bytes % static_cast<long long>(sizeof(double)) != 0)
        die("--bw-bytes must be a positive multiple of sizeof(double)");
    if (a.production && a.devices.size() > 2)
        die("production calibration supports one or two local GPUs per node");
    return a;
}

struct Distribution {
    double min = 0.0;
    double q1 = 0.0;
    double median = 0.0;
    double q3 = 0.0;
    double max = 0.0;
};

[[nodiscard]] Distribution summarize(const std::vector<double>& samples)
{
    Distribution result;
    if (samples.empty()) return result;
    const auto extrema = std::minmax_element(samples.begin(), samples.end());
    result.min = *extrema.first;
    result.q1 = quantile(samples, 0.25);
    result.median = quantile(samples, 0.5);
    result.q3 = quantile(samples, 0.75);
    result.max = *extrema.second;
    return result;
}

struct ProductionMeasurement {
    std::string operation;
    std::string label;
    int n = 0;
    int s = 0;
    long long payload_doubles = 0;
    long long iterations = 0;
    std::vector<Distribution> local;
    Distribution global;
    double achieved_gbs = 0.0;
};

struct ProductionResult {
    int participants = 0;
    int processes = 0;
    int hosts = 0;
    int local_gpus = 0;
    std::string device_name;
    std::vector<std::string> process_hosts;
    std::vector<ProductionMeasurement> measurements;
};

/// Latency and bus bandwidth of one all-reduce at whichever rung was measured.
struct Result {
    double lat_s = 0.0;
    double lat_q1 = 0.0;
    double lat_q3 = 0.0;
    double bw_gbs = 0.0;
    int    participants = 0;
    int    hosts = 1;
};

/// Ring-all-reduce bus bandwidth: each participant moves 2(n-1)/n of the payload.
[[nodiscard]] double bus_gbs(double payload_bytes, int n, double seconds)
{
    if (!(seconds > 0.0) || n < 2) return 0.0;
    const double factor = 2.0 * static_cast<double>(n - 1) / static_cast<double>(n);
    return payload_bytes * factor / seconds * 1e-9;
}

void report(const Args& a, const Result& r, const char* rung, const char* device_name,
            bool recordable)
{
    std::printf("GPU interconnect calibration -- the top rungs of R_h's ladder\n");
    std::printf("  machine=%s  participants=%d  hosts=%d  rung=%s\n",
                a.machine.c_str(), r.participants, r.hosts, rung);
    std::printf("  device=%s  iters=%lld  repeats=%d\n\n", device_name, a.iters, a.repeats);
    std::printf("  all-reduce latency (1 double): %10.2f us   [q1 %.2f  q3 %.2f]\n",
                r.lat_s * 1e6, r.lat_q1 * 1e6, r.lat_q3 * 1e6);
    std::printf("  all-reduce bus bandwidth (%.0f MiB): %8.1f GB/s\n\n",
                static_cast<double>(a.bw_bytes) / (1024.0 * 1024.0), r.bw_gbs);

    if (!recordable) { report_suppressed(); return; }

    std::printf("To record, set these on the %s preset in include/gpu_machine.hpp:\n",
                a.machine.c_str());
    if (std::strcmp(rung, "node") == 0) {
        std::printf("  t_reduce_s[NODE]  = %.4e minus reduction_cost_s(gm, DEVICE_P2P)\n",
                    r.lat_s);
        std::printf("  tier_calibrated[NODE] = true\n");
    } else {
        std::printf("  t_reduce_s[DEVICE_P2P] = %.4e minus reduction_cost_s(gm, GRID)\n",
                    r.lat_s);
        std::printf("  tier_calibrated[DEVICE_P2P] = true\n");
    }
    std::printf("  interconnect_bw_gbs = %.1f\n", r.bw_gbs);
    std::printf("\nThe entries are increments: reduction_cost_s() accumulates the rungs, "
                "because a\ncross-link all-reduce necessarily performs the on-device combines "
                "first. Subtract the\nrung below before recording, or the lower rungs are "
                "double-counted.\n");
}

void write_csv(const Args& a, const Result& r, const char* rung, const char* device_name,
               bool contended)
{
    if (a.csv_path.empty()) return;
    std::FILE* f = std::fopen(a.csv_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str()); return; }
    std::fprintf(f, "machine,device_name,participants,hosts,tier,iters,repeats,"
                    "t_allreduce_s,t_q1,t_q3,bw_payload_bytes,bw_gbs,contended\n");
    std::fprintf(f, "%s,\"%s\",%d,%d,%s,%lld,%d,%.9e,%.9e,%.9e,%lld,%.4f,%d\n",
                 a.machine.c_str(), device_name, r.participants, r.hosts, rung,
                 a.iters, a.repeats, r.lat_s, r.lat_q1, r.lat_q3, a.bw_bytes, r.bw_gbs,
                 contended ? 1 : 0);
    std::fclose(f);
    std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
}

void report_production(const Args& a, const ProductionResult& result)
{
    int runtime = 0;
    int driver = 0;
    int nccl_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    NCCL_CHECK(ncclGetVersion(&nccl_version));

    std::printf("CA participant-count calibration\n");
    std::printf(
        "  machine=%s | participants=%d | MPI processes=%d | hosts=%d | local GPUs/process=%d\n",
        a.machine.c_str(), result.participants, result.processes,
        result.hosts, result.local_gpus);
    std::printf(
        "  devices=%s | device=%s | repeats=%d | collective iters=%lld | halo iters=%lld\n",
        device_list(a.devices).c_str(), result.device_name.c_str(),
        a.repeats, a.iters, a.halo_iters);
    std::printf(
        "  CUDA toolkit=%d runtime=%d driver=%d | NCCL=%d\n\n",
        CUDART_VERSION, runtime, driver, nccl_version);

    std::printf(
        "  %-10s %-21s %10s %12s %23s %10s\n",
        "operation", "payload", "doubles", "global med",
        "global [q1, q3]", "GB/s");
    for (const ProductionMeasurement& measurement : result.measurements) {
        std::printf(
            "  %-10s %-21s %10lld %9.2f us [%8.2f, %8.2f] %10.3f\n",
            measurement.operation.c_str(), measurement.label.c_str(),
            measurement.payload_doubles,
            measurement.global.median * 1e6,
            measurement.global.q1 * 1e6,
            measurement.global.q3 * 1e6,
            measurement.achieved_gbs);
    }

    std::printf("\n  MPI-process-local medians (microseconds):\n");
    for (const ProductionMeasurement& measurement : result.measurements) {
        std::printf("    %-10s %-21s", measurement.operation.c_str(),
                    measurement.label.c_str());
        for (std::size_t rank = 0; rank < measurement.local.size(); ++rank)
            std::printf(
                " rank%zu@%s=%.2f",
                rank, result.process_hosts[rank].c_str(),
                measurement.local[rank].median * 1e6);
        std::printf("\n");
    }
    std::printf(
        "\n  All reported times use the slowest MPI process for each repeat.\n"
        "  All-reduce GB/s is NCCL ring bus bandwidth. Halo GB/s is aggregate\n"
        "  bidirectional payload across the participant chain.\n");
}

void write_production_csv(const Args& a, const ProductionResult& result)
{
    if (a.csv_path.empty()) return;
    std::FILE* file = std::fopen(a.csv_path.c_str(), "w");
    if (!file) {
        std::fprintf(stderr, "Cannot open CSV: %s\n", a.csv_path.c_str());
        return;
    }
    std::fprintf(
        file,
        "machine,device_name,participants,mpi_processes,hosts,local_gpus,"
        "devices,tier,operation,label,n,s,payload_doubles,payload_bytes,"
        "iterations,repeats,mpi_rank,host,local_min_s,local_q1_s,"
        "local_median_s,local_q3_s,local_max_s,global_min_s,global_q1_s,"
        "global_median_s,global_q3_s,global_max_s,achieved_gbs,contended\n");
    for (const ProductionMeasurement& measurement : result.measurements) {
        for (std::size_t rank = 0; rank < measurement.local.size(); ++rank) {
            const Distribution& local = measurement.local[rank];
            std::fprintf(
                file,
                "%s,\"%s\",%d,%d,%d,%d,%s,node,%s,%s,%d,%d,%lld,%lld,"
                "%lld,%d,%zu,\"%s\",%.9e,%.9e,%.9e,%.9e,%.9e,"
                "%.9e,%.9e,%.9e,%.9e,%.9e,%.6f,0\n",
                a.machine.c_str(), result.device_name.c_str(),
                result.participants, result.processes, result.hosts,
                result.local_gpus, device_list(a.devices).c_str(),
                measurement.operation.c_str(), measurement.label.c_str(),
                measurement.n, measurement.s, measurement.payload_doubles,
                measurement.payload_doubles
                    * static_cast<long long>(sizeof(double)),
                measurement.iterations, a.repeats, rank,
                result.process_hosts[rank].c_str(),
                local.min, local.q1, local.median, local.q3, local.max,
                measurement.global.min, measurement.global.q1,
                measurement.global.median, measurement.global.q3,
                measurement.global.max, measurement.achieved_gbs);
        }
    }
    std::fclose(file);
    std::printf("\n  [Wrote %s]\n", a.csv_path.c_str());
}

// Single-process, multi-device: the DEVICE_P2P rung
/**
 * @brief Drive every local GPU from one process with ncclCommInitAll.
 *
 * No launcher and no MPI. On synge both V100s are on one node, so this is the whole of the
 * DEVICE_P2P rung, which Phase 0 predicts carries the production operator into Upper-Right.
 * Keeping MPI off its critical path means the headline measurement depends on CUDA and NCCL
 * only.
 *
 * The all-reduces are issued inside ncclGroupStart/End so the devices participate in one
 * collective rather than deadlocking on each other, and every stream is synchronized before
 * the clock stops: a collective costs what its slowest participant costs, not the first to
 * return.
 */
[[nodiscard]] Result run_single_process(const Args& a, int ndev, char* device_name,
                                        std::size_t device_name_len)
{
    std::vector<int> devs(static_cast<std::size_t>(ndev));
    std::iota(devs.begin(), devs.end(), 0);

    std::vector<ncclComm_t>  comms(static_cast<std::size_t>(ndev));
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(ndev));
    std::vector<double*>      lat(static_cast<std::size_t>(ndev), nullptr);
    std::vector<double*>      bw(static_cast<std::size_t>(ndev), nullptr);

    const long long bw_elems = a.bw_bytes / 8;
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&streams[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaMalloc(&lat[static_cast<std::size_t>(d)], sizeof(double)));
        CUDA_CHECK(cudaMalloc(&bw[static_cast<std::size_t>(d)],
                              static_cast<std::size_t>(bw_elems) * sizeof(double)));
        CUDA_CHECK(cudaMemset(lat[static_cast<std::size_t>(d)], 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(bw[static_cast<std::size_t>(d)], 0,
                              static_cast<std::size_t>(bw_elems) * sizeof(double)));
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::snprintf(device_name, device_name_len, "%s", prop.name);

    NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devs.data()));

    auto timed = [&](long long count, long long iters) -> double {
        for (int d = 0; d < ndev; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(d)]));
        }
        const Clock::time_point t0 = Clock::now();
        for (long long i = 0; i < iters; ++i) {
            NCCL_CHECK(ncclGroupStart());
            for (int d = 0; d < ndev; ++d) {
                double* buf = count == 1 ? lat[static_cast<std::size_t>(d)]
                                         : bw[static_cast<std::size_t>(d)];
                NCCL_CHECK(ncclAllReduce(buf, buf, static_cast<std::size_t>(count),
                                         ncclDouble, ncclSum,
                                         comms[static_cast<std::size_t>(d)],
                                         streams[static_cast<std::size_t>(d)]));
            }
            NCCL_CHECK(ncclGroupEnd());
        }
        for (int d = 0; d < ndev; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaStreamSynchronize(streams[static_cast<std::size_t>(d)]));
        }
        return Sec(Clock::now() - t0).count() / static_cast<double>(iters);
    };

    (void)timed(1, std::min(a.iters, 200LL));   // warm-up: NCCL rings, buffers, routes

    std::vector<double> lat_s;
    lat_s.reserve(static_cast<std::size_t>(a.repeats));
    for (int k = 0; k < a.repeats; ++k) lat_s.push_back(timed(1, a.iters));

    const long long bw_iters = std::max(10LL, a.iters / 100);
    (void)timed(bw_elems, 5);
    std::vector<double> bw_s;
    bw_s.reserve(static_cast<std::size_t>(a.repeats));
    for (int k = 0; k < a.repeats; ++k) bw_s.push_back(timed(bw_elems, bw_iters));

    Result r;
    r.lat_s  = quantile(lat_s, 0.5);
    r.lat_q1 = quantile(lat_s, 0.25);
    r.lat_q3 = quantile(lat_s, 0.75);
    r.bw_gbs = bus_gbs(static_cast<double>(a.bw_bytes), ndev, quantile(bw_s, 0.5));
    r.participants = ndev;
    r.hosts = 1;

    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(lat[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaFree(bw[static_cast<std::size_t>(d)]));
        CUDA_CHECK(cudaStreamDestroy(streams[static_cast<std::size_t>(d)]));
        NCCL_CHECK(ncclCommDestroy(comms[static_cast<std::size_t>(d)]));
    }
    return r;
}

}  // namespace

// MPI path: the NODE rung
#ifdef CAKSM_HAVE_MPI
namespace {

/// One rank per GPU, bootstrapped over MPI. Only needed once the crossing leaves the node.
[[nodiscard]] Result run_mpi(const Args& a, int rank, int nranks, char* device_name,
                             std::size_t device_name_len, int& distinct_hosts,
                             bool& contended)
{
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    std::vector<char> all_hosts(static_cast<std::size_t>(nranks) * MPI_MAX_PROCESSOR_NAME, 0);
    MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

    auto host_at = [&](int i) -> const char* {
        return &all_hosts[static_cast<std::size_t>(i) * MPI_MAX_PROCESSOR_NAME];
    };

    distinct_hosts = 0;
    for (int i = 0; i < nranks; ++i) {
        bool seen = false;
        for (int j = 0; j < i; ++j)
            if (std::strcmp(host_at(i), host_at(j)) == 0) { seen = true; break; }
        if (!seen) ++distinct_hosts;
    }

    // Local rank picks the device, so two ranks on one node take different GPUs.
    int local_rank = 0;
    for (int i = 0; i < rank; ++i)
        if (std::strcmp(host_at(i), host) == 0) ++local_rank;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) die("MPI rank has no visible CUDA device");
    const int device = local_rank % ndev;
    CUDA_CHECK(cudaSetDevice(device));

    const DeviceContention contention = check_device_contention();
    const int local_contended = contention.contended ? 1 : 0;
    int any_contended = 0;
    MPI_Allreduce(&local_contended, &any_contended, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    contended = any_contended != 0;
    if (contention.contended) {
        std::printf("  rank %d device %d:", rank, device);
        report_contention(contention);
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::snprintf(device_name, device_name_len, "%s", prop.name);

    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const long long bw_elems = a.bw_bytes / 8;
    double *d_lat = nullptr, *d_bw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lat, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bw, static_cast<std::size_t>(bw_elems) * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_lat, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bw, 0, static_cast<std::size_t>(bw_elems) * sizeof(double)));

    auto timed = [&](long long count, long long iters) -> double {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        for (long long i = 0; i < iters; ++i) {
            double* buf = count == 1 ? d_lat : d_bw;
            NCCL_CHECK(ncclAllReduce(buf, buf, static_cast<std::size_t>(count), ncclDouble,
                                     ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return (MPI_Wtime() - t0) / static_cast<double>(iters);
    };

    (void)timed(1, std::min(a.iters, 200LL));

    std::vector<double> lat_s;
    lat_s.reserve(static_cast<std::size_t>(a.repeats));
    for (int k = 0; k < a.repeats; ++k) lat_s.push_back(timed(1, a.iters));

    const long long bw_iters = std::max(10LL, a.iters / 100);
    (void)timed(bw_elems, 5);
    std::vector<double> bw_s;
    bw_s.reserve(static_cast<std::size_t>(a.repeats));
    for (int k = 0; k < a.repeats; ++k) bw_s.push_back(timed(bw_elems, bw_iters));

    // Each sample costs what its slowest participant costs.
    std::vector<double> collective_lat_s(lat_s.size());
    std::vector<double> collective_bw_s(bw_s.size());
    MPI_Allreduce(lat_s.data(), collective_lat_s.data(), a.repeats,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(bw_s.data(), collective_bw_s.data(), a.repeats,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double bw_t = quantile(collective_bw_s, 0.5);

    Result r;
    r.lat_s  = quantile(collective_lat_s, 0.5);
    r.lat_q1 = quantile(collective_lat_s, 0.25);
    r.lat_q3 = quantile(collective_lat_s, 0.75);
    r.bw_gbs = bus_gbs(static_cast<double>(a.bw_bytes), nranks, bw_t);
    r.participants = nranks;
    r.hosts = distinct_hosts;

    CUDA_CHECK(cudaFree(d_lat));
    CUDA_CHECK(cudaFree(d_bw));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    return r;
}

/**
 * @brief Measure the communication calls made by the production CA integrator.
 *
 * The solver launches one MPI process per node and lets that process drive one or two local
 * GPUs. Keeping the same layout here is important: four MPI processes would create the same
 * number of NCCL ranks, but would price a different host-submission and synchronization path.
 */
[[nodiscard]] ProductionResult run_production_mpi(
    const Args& a, int rank, int nranks)
{
    if (nranks != 2)
        die("production calibration requires exactly two MPI processes, one per node");

    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    std::vector<char> all_hosts(
        static_cast<std::size_t>(nranks) * MPI_MAX_PROCESSOR_NAME, 0);
    MPI_Allgather(
        host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
        all_hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    auto host_at = [&](int process) -> const char* {
        return &all_hosts[
            static_cast<std::size_t>(process) * MPI_MAX_PROCESSOR_NAME];
    };

    int distinct_hosts = 0;
    for (int i = 0; i < nranks; ++i) {
        bool seen = false;
        for (int j = 0; j < i; ++j)
            if (std::strcmp(host_at(i), host_at(j)) == 0) {
                seen = true;
                break;
            }
        if (!seen) ++distinct_hosts;
    }
    if (distinct_hosts != nranks)
        die("production calibration requires one MPI process on each of two distinct nodes");

    const int local_gpus = static_cast<int>(a.devices.size());
    int min_local_gpus = 0;
    int max_local_gpus = 0;
    MPI_Allreduce(
        &local_gpus, &min_local_gpus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(
        &local_gpus, &max_local_gpus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (min_local_gpus != max_local_gpus)
        die("every MPI process must use the same number of local GPUs");

    int visible_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visible_devices));
    std::string device_name;
    int local_contended = 0;
    for (int device : a.devices) {
        if (device >= visible_devices)
            die("--devices names a GPU that is not visible to this MPI process");
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (device_name.empty())
            device_name = properties.name;
        else if (device_name != properties.name)
            die("production calibration requires identical local GPU models");
        const DeviceContention contention = check_device_contention();
        if (contention.contended) {
            std::printf("  MPI rank %d device %d:", rank, device);
            report_contention(contention);
            local_contended = 1;
        }
    }
    constexpr int kDeviceNameBytes = 256;
    char local_device_name[kDeviceNameBytes] = {};
    std::snprintf(
        local_device_name, sizeof(local_device_name), "%s",
        device_name.c_str());
    std::vector<char> all_device_names(
        static_cast<std::size_t>(nranks * kDeviceNameBytes), 0);
    MPI_Allgather(
        local_device_name, kDeviceNameBytes, MPI_CHAR,
        all_device_names.data(), kDeviceNameBytes, MPI_CHAR,
        MPI_COMM_WORLD);
    for (int process = 1; process < nranks; ++process) {
        const char* remote =
            &all_device_names[
                static_cast<std::size_t>(process * kDeviceNameBytes)];
        if (std::strcmp(local_device_name, remote) != 0)
            die("production calibration requires identical GPU models on every node");
    }
    int any_contended = 0;
    MPI_Allreduce(
        &local_contended, &any_contended, 1, MPI_INT, MPI_MAX,
        MPI_COMM_WORLD);
    if (any_contended)
        die("production calibration requires every participating GPU to be idle");

    const int participants = nranks * local_gpus;
    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<ncclComm_t> comms(static_cast<std::size_t>(local_gpus));
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(local_gpus));
    NCCL_CHECK(ncclGroupStart());
    for (int local = 0; local < local_gpus; ++local) {
        CUDA_CHECK(cudaSetDevice(a.devices[static_cast<std::size_t>(local)]));
        const int global_rank = rank * local_gpus + local;
        NCCL_CHECK(ncclCommInitRank(
            &comms[static_cast<std::size_t>(local)],
            participants, id, global_rank));
    }
    NCCL_CHECK(ncclGroupEnd());

    const long long bandwidth_doubles =
        a.bw_bytes / static_cast<long long>(sizeof(double));
    const long long max_collective_doubles =
        std::max(100LL, bandwidth_doubles);
    const long long max_halo_doubles = 4LL * 97LL * 97LL;
    std::vector<double*> collective(
        static_cast<std::size_t>(local_gpus), nullptr);
    std::vector<double*> halo_send(
        static_cast<std::size_t>(local_gpus), nullptr);
    std::vector<double*> halo_recv(
        static_cast<std::size_t>(local_gpus), nullptr);
    for (int local = 0; local < local_gpus; ++local) {
        CUDA_CHECK(cudaSetDevice(a.devices[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaStreamCreate(
            &streams[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaMalloc(
            &collective[static_cast<std::size_t>(local)],
            static_cast<std::size_t>(max_collective_doubles)
                * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            &halo_send[static_cast<std::size_t>(local)],
            static_cast<std::size_t>(2 * max_halo_doubles)
                * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            &halo_recv[static_cast<std::size_t>(local)],
            static_cast<std::size_t>(2 * max_halo_doubles)
                * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            collective[static_cast<std::size_t>(local)], 0,
            static_cast<std::size_t>(max_collective_doubles)
                * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            halo_send[static_cast<std::size_t>(local)], 0,
            static_cast<std::size_t>(2 * max_halo_doubles)
                * sizeof(double)));
        CUDA_CHECK(cudaMemset(
            halo_recv[static_cast<std::size_t>(local)], 0,
            static_cast<std::size_t>(2 * max_halo_doubles)
                * sizeof(double)));
    }

    auto sync_local = [&]() {
        for (int local = 0; local < local_gpus; ++local) {
            CUDA_CHECK(cudaSetDevice(
                a.devices[static_cast<std::size_t>(local)]));
            CUDA_CHECK(cudaStreamSynchronize(
                streams[static_cast<std::size_t>(local)]));
        }
    };

    auto timed_allreduce =
        [&](long long count, long long iterations) -> double {
            sync_local();
            MPI_Barrier(MPI_COMM_WORLD);
            const double begin = MPI_Wtime();
            for (long long iteration = 0;
                 iteration < iterations; ++iteration) {
                NCCL_CHECK(ncclGroupStart());
                for (int local = 0; local < local_gpus; ++local) {
                    CUDA_CHECK(cudaSetDevice(
                        a.devices[static_cast<std::size_t>(local)]));
                    NCCL_CHECK(ncclAllReduce(
                        collective[static_cast<std::size_t>(local)],
                        collective[static_cast<std::size_t>(local)],
                        static_cast<std::size_t>(count), ncclDouble,
                        ncclSum,
                        comms[static_cast<std::size_t>(local)],
                        streams[static_cast<std::size_t>(local)]));
                }
                NCCL_CHECK(ncclGroupEnd());
            }
            sync_local();
            return (MPI_Wtime() - begin)
                / static_cast<double>(iterations);
        };

    auto timed_halo =
        [&](long long count, long long iterations) -> double {
            sync_local();
            MPI_Barrier(MPI_COMM_WORLD);
            const double begin = MPI_Wtime();
            for (long long iteration = 0;
                 iteration < iterations; ++iteration) {
                NCCL_CHECK(ncclGroupStart());
                for (int local = 0; local < local_gpus; ++local) {
                    CUDA_CHECK(cudaSetDevice(
                        a.devices[static_cast<std::size_t>(local)]));
                    const int global_rank = rank * local_gpus + local;
                    ncclComm_t comm =
                        comms[static_cast<std::size_t>(local)];
                    cudaStream_t stream =
                        streams[static_cast<std::size_t>(local)];
                    double* send =
                        halo_send[static_cast<std::size_t>(local)];
                    double* receive =
                        halo_recv[static_cast<std::size_t>(local)];
                    if (global_rank > 0) {
                        NCCL_CHECK(ncclRecv(
                            receive, static_cast<std::size_t>(count),
                            ncclDouble, global_rank - 1, comm, stream));
                        NCCL_CHECK(ncclSend(
                            send, static_cast<std::size_t>(count),
                            ncclDouble, global_rank - 1, comm, stream));
                    }
                    if (global_rank + 1 < participants) {
                        NCCL_CHECK(ncclRecv(
                            receive + max_halo_doubles,
                            static_cast<std::size_t>(count),
                            ncclDouble, global_rank + 1, comm, stream));
                        NCCL_CHECK(ncclSend(
                            send + max_halo_doubles,
                            static_cast<std::size_t>(count),
                            ncclDouble, global_rank + 1, comm, stream));
                    }
                }
                NCCL_CHECK(ncclGroupEnd());
            }
            sync_local();
            return (MPI_Wtime() - begin)
                / static_cast<double>(iterations);
        };

    ProductionResult result;
    result.participants = participants;
    result.processes = nranks;
    result.hosts = distinct_hosts;
    result.local_gpus = local_gpus;
    result.device_name = device_name;
    if (rank == 0) {
        result.process_hosts.reserve(static_cast<std::size_t>(nranks));
        for (int process = 0; process < nranks; ++process)
            result.process_hosts.emplace_back(host_at(process));
    }

    auto record =
        [&](ProductionMeasurement measurement,
            const std::vector<double>& local_samples) {
            std::vector<double> gathered;
            if (rank == 0)
                gathered.resize(
                    static_cast<std::size_t>(nranks * a.repeats));
            MPI_Gather(
                local_samples.data(), a.repeats, MPI_DOUBLE,
                gathered.data(), a.repeats, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
            if (rank != 0) return;

            measurement.local.reserve(static_cast<std::size_t>(nranks));
            std::vector<double> global_samples(
                static_cast<std::size_t>(a.repeats), 0.0);
            for (int process = 0; process < nranks; ++process) {
                const auto begin =
                    gathered.begin()
                    + static_cast<std::ptrdiff_t>(process * a.repeats);
                const auto end = begin + a.repeats;
                const std::vector<double> samples(begin, end);
                measurement.local.push_back(summarize(samples));
                for (int repeat = 0; repeat < a.repeats; ++repeat)
                    global_samples[static_cast<std::size_t>(repeat)] =
                        std::max(
                            global_samples[static_cast<std::size_t>(repeat)],
                            samples[static_cast<std::size_t>(repeat)]);
            }
            measurement.global = summarize(global_samples);
            const double bytes =
                static_cast<double>(measurement.payload_doubles)
                * sizeof(double);
            if (measurement.operation == "allreduce")
                measurement.achieved_gbs =
                    bus_gbs(
                        bytes, participants,
                        measurement.global.median);
            else
                measurement.achieved_gbs =
                    2.0 * static_cast<double>(participants - 1) * bytes
                    / measurement.global.median * 1e-9;
            result.measurements.push_back(std::move(measurement));
        };

    const std::vector<std::pair<std::string, long long>>
        collective_payloads{
            {"norm", 1},
            {"gram_s1", 1},
            {"projection_s1", 25},
            {"gram_s4", 16},
            {"projection_s4", 100},
            {"bandwidth", bandwidth_doubles}};
    for (const auto& [label, count] : collective_payloads) {
        const long long iterations =
            label == "bandwidth"
                ? std::max(10LL, a.iters / 100)
                : a.iters;
        (void)timed_allreduce(
            count, label == "bandwidth"
                ? 5LL : std::min(50LL, iterations));
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(a.repeats));
        for (int repeat = 0; repeat < a.repeats; ++repeat)
            samples.push_back(timed_allreduce(count, iterations));
        ProductionMeasurement measurement;
        measurement.operation = "allreduce";
        measurement.label = label;
        measurement.payload_doubles = count;
        measurement.iterations = iterations;
        record(std::move(measurement), samples);
    }

    for (int n : {61, 77, 97}) {
        for (int s : {1, 4}) {
            const long long count =
                static_cast<long long>(s)
                * static_cast<long long>(n)
                * static_cast<long long>(n);
            (void)timed_halo(
                count, std::min(20LL, a.halo_iters));
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(a.repeats));
            for (int repeat = 0; repeat < a.repeats; ++repeat)
                samples.push_back(
                    timed_halo(count, a.halo_iters));
            ProductionMeasurement measurement;
            measurement.operation = "halo";
            measurement.label =
                "n" + std::to_string(n)
                + "_s" + std::to_string(s);
            measurement.n = n;
            measurement.s = s;
            measurement.payload_doubles = count;
            measurement.iterations = a.halo_iters;
            record(std::move(measurement), samples);
        }
    }

    for (int local = 0; local < local_gpus; ++local) {
        CUDA_CHECK(cudaSetDevice(
            a.devices[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaFree(
            collective[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaFree(
            halo_send[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaFree(
            halo_recv[static_cast<std::size_t>(local)]));
        CUDA_CHECK(cudaStreamDestroy(
            streams[static_cast<std::size_t>(local)]));
        NCCL_CHECK(ncclCommDestroy(
            comms[static_cast<std::size_t>(local)]));
    }
    return result;
}

}  // namespace
#endif  // CAKSM_HAVE_MPI

int main(int argc, char** argv)
{
    int rank = 0, nranks = 1;
#ifdef CAKSM_HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

    const Args a = parse_args(argc, argv);

    char device_name[256] = "unknown";
    Result r;
    const char* rung = "device-p2p";
    int distinct_hosts = 1;
    bool contended = false;

    /*
     * MPI bootstrap check, before any measurement.
     *
     * Asking for the NODE rung while MPI_COMM_WORLD has one rank is not a geometry mistake: it
     * means MPI_Init succeeded but the ranks never found each other, so each process is a
     * singleton. Every one then falls into the single-process path below, drives its own
     * node's two GPUs, and reports a perfectly good DEVICE_P2P number under a NODE label. The
     * tier guard catches that afterward, but reports it as a geometry error, which sends you
     * to fix srun flags that are already correct.
     *
     * The usual cause is the launcher's PMI. Open MPI 5.x speaks PMIx, and srun provides it
     * only when built with PMIx support and asked (`srun --mpi=pmix`). Without it every rank
     * initializes alone and MPI_Comm_size returns 1 on all of them.
     */
    if (a.tier == "node" && nranks <= 1) {
        std::fprintf(stderr,
            "--tier node was requested, but MPI_COMM_WORLD has %d rank(s): the ranks never\n"
            "formed a communicator, so each process is a singleton. This is an MPI bootstrap\n"
            "failure, not a bad srun geometry; your --nodes/--ntasks flags are probably fine.\n"
            "\n"
            "Open MPI 5.x needs PMIx from the launcher. Try, in order:\n"
            "  srun --mpi=pmix   ...      (see `srun --mpi=list` for what this site supports)\n"
            "  srun --mpi=pmi2   ...\n"
            "  mpirun -np 2 -N 1 ...      (bypass srun's PMI entirely)\n"
            "\n"
            "Refusing to measure: a singleton would silently produce a DEVICE_P2P number and\n"
            "record it against the NODE rung, which is a wrong constant that looks right.\n",
            nranks);
        die("aborting");
    }
#ifndef CAKSM_HAVE_MPI
    if (a.tier == "node")
        die("--tier node requested, but this binary was built without MPI (cmake reported\n"
            "'NCCL only, no MPI'). There is no fabric path compiled in. Load an MPI module\n"
            "and reconfigure.");
#endif

    if (a.production) {
        if (a.tier != "node")
            die("--production requires --tier node");
#ifdef CAKSM_HAVE_MPI
        const ProductionResult production =
            run_production_mpi(a, rank, nranks);
        if (rank == 0) {
            report_production(a, production);
            write_production_csv(a, production);
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
#else
        die("--production requires an MPI-enabled build");
#endif
    }

    if (nranks > 1) {
#ifdef CAKSM_HAVE_MPI
        r = run_mpi(a, rank, nranks, device_name, sizeof(device_name), distinct_hosts,
                    contended);
        rung = distinct_hosts > 1 ? "node" : "device-p2p";
#endif
    } else {
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 2)
            die("Need at least 2 visible GPUs for the DEVICE_P2P rung: it is a crossing, and\n"
                "one device has nothing to cross. Check CUDA_VISIBLE_DEVICES and that the\n"
                "allocation requested --gpus-per-node=2.");

        // Every participating device must be idle: a collective costs what its slowest
        // participant costs, so one contended GPU contaminates the whole measurement.
        for (int d = 0; d < ndev; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            const DeviceContention c = check_device_contention();
            if (c.contended) {
                std::printf("  device %d:", d);
                report_contention(c);
                contended = true;
            }
        }
        r = run_single_process(a, ndev, device_name, sizeof(device_name));
        rung = "device-p2p";
    }

    // Detected rung against the asserted one. Recording a node-tier latency in the intra-node
    // slot would put a wrong constant into R_h's numerator with no symptom.
    if (a.tier != "auto" && a.tier != rung && rank == 0) {
        std::fprintf(stderr,
            "Tier mismatch: --tier %s but the launch shape is the '%s' rung (%d participants "
            "across %d host(s)).\nFix the geometry rather than the flag.\n",
            a.tier.c_str(), rung, r.participants, r.hosts);
        die("aborting");
    }

    if (rank == 0) {
        report(a, r, rung, device_name, !contended);
        write_csv(a, r, rung, device_name, contended);
        if (std::strcmp(rung, "device-p2p") == 0 && r.hosts == 1)
            std::printf("\nNote: measured with %s, on one node.\n",
                        nranks > 1 ? "MPI ranks" : "ncclCommInitAll and no MPI");
    }

#ifdef CAKSM_HAVE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
