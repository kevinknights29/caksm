/**
 * @file ca_matrix_powers.cu
 * @brief Correctness and traffic harness for the pricing matrix-powers kernel.
 *
 * Two jobs. It checks the matrix-free recurrence against the assembled operator,
 * which is what any tile-geometry change is gated on, and it reports what the
 * kernel costs vertically: shared tile volume, ghost redundancy, modeled
 * compulsory and tiled traffic, and the achieved rate against the device's DRAM
 * and L2 roofs.
 *
 * A verdict is only named from a measured DRAM byte count. Pass --ncu-dram-bytes
 * from an ncu sector capture of the same point; without it the row is diagnostic
 * and the verdict stays undetermined, because a modeled numerator cannot settle
 * which roof a kernel is near.
 *
 * @author Kevin Knights
 * @date 2026-07-26
 */


#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ca_pricing_gpu.hpp"
#include "gpu_contention.cuh"
#include "gpu_machine.hpp"
#include "gpu_pde_matrix_powers_streamed.cuh"
#include "regime.hpp"

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t e_ = (call);                                                \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                     \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

// Build identity a result row carries, so a configuration can be recovered from
// the record alone. The register cap and the configuration name are supplied by
// the build definition of each swept variant; the defaults describe an ordinary
// hand-built binary rather than pretending one was not built.
#ifndef CAKSM_MPK_REGISTER_CAP
#define CAKSM_MPK_REGISTER_CAP 0
#endif
#ifndef CAKSM_BUILD_TYPE
#define CAKSM_BUILD_TYPE "unspecified"
#endif
#ifndef CAKSM_MPK_CONFIGURATION
#define CAKSM_MPK_CONFIGURATION "default"
#endif

namespace {

/// Which recurrence builds the basis. All three produce the same subspace and
/// differ only in the conditioning of the columns that span it.
enum class PolynomialBasis {
    Monomial,
    Newton,
    Chebyshev
};

/// Lower-case name for the record. Matches the string the figures key on.
[[nodiscard]] const char* basis_name(PolynomialBasis basis)
{
    switch (basis) {
        case PolynomialBasis::Monomial: return "monomial";
        case PolynomialBasis::Newton: return "newton";
        case PolynomialBasis::Chebyshev: return "chebyshev";
    }
    return "unknown";
}

[[nodiscard]] const char* carveout_name(MpkSharedCarveout carveout)
{
    return carveout == MpkSharedCarveout::Maximum ? "max" : "default";
}

/// The interior extents one block owns, whichever family staged them. Both
/// families stage (x+2s)(y+2s)(z+2s) around it, so one geometry drives the
/// traffic model and the grid count for either.
struct TileGeometry {
    int x = kMpkBlockX;
    int y = kMpkBlockY;
    int z = kMpkBlockZ;
};

/// Command line, already validated by parse_args.
struct Args {
    int n = 31;
    int s = 4;
    int device = 0;
    int repeats = 7;
    double scale = 0.01;
    int certificate_block_width = 0;
    bool certificate_block_width_set = false;
    bool rainbow = false;
    PolynomialBasis basis = PolynomialBasis::Monomial;
    MpkSharedCarveout shared_carveout = MpkSharedCarveout::Default;
    /// Which kernel family builds the basis. The plane-streamed family is a
    /// separate implementation under test, not a mode of the accepted one.
    MpkKernelFamily family = MpkKernelFamily::FullVolume;
    /// z planes one streamed block owns. Zero streams the whole extent. Ignored
    /// by the full-volume family, whose z extent is a compile-time tile bound.
    int stream_height = kMpkStreamHeight;
    /// DRAM bytes from an ncu sector capture of the same point. Negative means the
    /// run is diagnostic: no vertical verdict is assigned without a measurement.
    double ncu_dram_bytes = -1.0;
    /// Execute exactly one basis evaluation and exit.  The ncu harness uses this
    /// mode so a sector count cannot accidentally include the correctness launch
    /// or any of the timing repeats.
    bool profile_only = false;
};

/**
 * What limits the kernel, which is a different question from where its traffic
 * goes. Traffic attribution is reported separately as the measured-over-modeled
 * byte ratio; this names the resource the kernel is actually waiting on.
 *
 * There is no occupancy verdict. The tile sweep measured lower occupancy running
 * faster at both production points, so occupancy does not identify the limiter
 * here and a threshold on it would have labeled the wrong cause.
 *
 * LatencyBound is the honest residual: no roof is within reach, and the kernel is
 * limited by the cost of its memory access pattern rather than by any bandwidth
 * or launch ceiling. Which geometry minimizes that cost is not decidable from one
 * point, so it is settled by the tile sweep and reported across rows.
 */
enum class VerticalVerdict {
    DramBound,
    L2Bound,
    LaunchBound,
    LatencyBound,
    Undetermined
};

/// Name for the record. "undetermined" is emitted rather than omitted, so a
/// diagnostic row cannot be read as a missing field.
[[nodiscard]] const char* verdict_name(VerticalVerdict verdict)
{
    switch (verdict) {
        case VerticalVerdict::DramBound: return "DRAM-bound";
        case VerticalVerdict::L2Bound: return "L2-bound";
        case VerticalVerdict::LaunchBound: return "launch-bound";
        case VerticalVerdict::LatencyBound: return "latency-bound";
        case VerticalVerdict::Undetermined: return "undetermined";
    }
    return "unknown";
}

/// Parse and validate the command line, or throw.
///
/// Note the two width conventions this executable sits between: --s here is a
/// recurrence degree and materializes s+1 columns, while the integrator's --s
/// is a block width that consumes s. --certificate-block-width takes the
/// integrator's convention, so a prediction can be compared with a solver run
/// without being off by one column.
[[nodiscard]] Args parse_args(int argc, char** argv)
{
    Args a;
    for (int q = 1; q < argc; ++q) {
        const std::string arg = argv[q];
        auto next = [&]() -> const char* {
            if (++q >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
                std::exit(EXIT_FAILURE);
            }
            return argv[q];
        };
        if      (arg == "--n")       a.n = std::stoi(next());
        else if (arg == "--s")       a.s = std::stoi(next());
        else if (arg == "--device")  a.device = std::stoi(next());
        else if (arg == "--repeats") a.repeats = std::stoi(next());
        else if (arg == "--scale")   a.scale = std::stod(next());
        else if (arg == "--certificate-block-width") {
            a.certificate_block_width = std::stoi(next());
            a.certificate_block_width_set = true;
        }
        else if (arg == "--ncu-dram-bytes")
            a.ncu_dram_bytes = std::stod(next());
        else if (arg == "--profile-only")
            a.profile_only = true;
        else if (arg == "--stream-height")
            a.stream_height = std::stoi(next());
        else if (arg == "--kernel-family") {
            const std::string family = next();
            if (family == "full-volume")
                a.family = MpkKernelFamily::FullVolume;
            else if (family == "plane-streamed")
                a.family = MpkKernelFamily::PlaneStreamed;
            else {
                std::fprintf(
                    stderr,
                    "--kernel-family must be full-volume or plane-streamed\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--shared-carveout") {
            const std::string carveout = next();
            if (carveout == "default")
                a.shared_carveout = MpkSharedCarveout::Default;
            else if (carveout == "max")
                a.shared_carveout = MpkSharedCarveout::Maximum;
            else {
                std::fprintf(
                    stderr, "--shared-carveout must be default or max\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--basis") {
            const std::string basis = next();
            if      (basis == "monomial")  a.basis = PolynomialBasis::Monomial;
            else if (basis == "newton")    a.basis = PolynomialBasis::Newton;
            else if (basis == "chebyshev") a.basis = PolynomialBasis::Chebyshev;
            else {
                std::fprintf(
                    stderr,
                    "--basis must be monomial, newton, or chebyshev\n");
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "--option") {
            const std::string option = next();
            if      (option == "basket")  a.rainbow = false;
            else if (option == "rainbow") a.rainbow = true;
            else {
                std::fprintf(stderr, "--option must be basket or rainbow\n");
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "--help") {
            std::printf(
                "Usage: ./ca-matrix-powers [--n N] [--s S] [--device D]\n"
                "                           [--option basket|rainbow]\n"
                "                           [--basis monomial|newton|chebyshev]\n"
                "                           [--scale H] [--repeats K]\n"
                "                           [--certificate-block-width W]\n"
                "                           [--ncu-dram-bytes B]\n"
                "                           [--shared-carveout default|max]\n"
                "                           [--kernel-family full-volume|plane-streamed]\n"
                "                           [--stream-height H]\n"
                "                           [--profile-only]\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (a.n < 3) {
        std::fprintf(stderr, "--n must be >= 3\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.s < 1 || a.s > kMpkMaxS) {
        std::fprintf(stderr, "--s must be in [1, %d]\n", kMpkMaxS);
        std::exit(EXIT_FAILURE);
    }
    if (!a.certificate_block_width_set)
        a.certificate_block_width = a.s + 1;
    if (a.certificate_block_width < 1
        || a.certificate_block_width > a.s + 1) {
        std::fprintf(
            stderr,
            "--certificate-block-width must be in [1, s+1]\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.repeats < 1) {
        std::fprintf(stderr, "--repeats must be >= 1\n");
        std::exit(EXIT_FAILURE);
    }
    if (a.stream_height < 0) {
        std::fprintf(
            stderr, "--stream-height must be >= 0 (0 streams the full z)\n");
        std::exit(EXIT_FAILURE);
    }
    return a;
}

/// Median of a repeat distribution. Takes its argument by value because it
/// sorts in place.
[[nodiscard]] double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/// Logical global-memory traffic requested by one basis evaluation.  This is
/// deliberately distinct from an ncu DRAM-sector count: the latter includes
/// cache effects, while this model counts only valid (non-zero-fill) tile
/// loads, all basis writes, and Basket boundary data.
struct TrafficModel {
    int chunks = 0;
    int pde_launches = 0;
    int tail_launches = 0;
    int max_chunk = 0;
    int shared_buffers = 0;
    int64_t tile_values = 0;
    int64_t input_values = 0;
    int64_t basis_write_values = 0;
    int64_t face_values = 0;
    int64_t boundary_tail_values = 0;
    int64_t tail_kernel_values = 0;
};

/// Along one axis, the total number of in-domain planes the tiles actually
/// load, summed over every tile origin.
///
/// A tile at the domain edge has part of its ghost shell outside the grid, and
/// the kernel zero-fills that rather than reading it. Counting the full tile
/// extent everywhere would charge the model for loads that never happen, which
/// is exactly the over-count the earlier traffic estimate suffered from.
[[nodiscard]] int64_t valid_extent_sum(int n, int block, int halo)
{
    int64_t total = 0;
    for (int origin = 0; origin < n; origin += block) {
        const int lo = std::max(0, origin - halo);
        const int hi = std::min(n, origin + block + halo);
        total += std::max(0, hi - lo);
    }
    return total;
}

/// Along one axis, how many tiles touch the upper boundary plane. Only those
/// tiles read the Basket face coefficients, so only they are charged for them.
[[nodiscard]] int64_t upper_face_tile_count(int n, int block, int halo)
{
    int64_t total = 0;
    for (int origin = 0; origin < n; origin += block) {
        if (origin - halo <= n - 1 && n - 1 < origin + block + halo)
            ++total;
    }
    return total;
}

/// Accumulate the traffic one launch requests, into the running model.
///
/// A chunk stages its tile once and writes every column it produces. Chebyshev
/// stages a second input tile when it has a predecessor, and rewrites the
/// column it was seeded with, which is what a chunk boundary costs. Basket adds
/// the face coefficients and augmented-tail values that the boundary-touching
/// tiles read, charged per recurrence step because the valid region shrinks as
/// the step index rises.
void add_chunk_traffic(
    TrafficModel& traffic, int n, const TileGeometry& tile, int chunk,
    bool chebyshev, bool has_previous, bool basket)
{
    const int64_t N =
        static_cast<int64_t>(n) * static_cast<int64_t>(n)
        * static_cast<int64_t>(n);
    const int64_t x_values = valid_extent_sum(n, tile.x, chunk);
    const int64_t y_values = valid_extent_sum(n, tile.y, chunk);
    const int64_t z_values = valid_extent_sum(n, tile.z, chunk);
    const int64_t valid_tile_values = x_values * y_values * z_values;
    const int64_t staged =
        static_cast<int64_t>(tile.x + 2 * chunk)
        * static_cast<int64_t>(tile.y + 2 * chunk)
        * static_cast<int64_t>(tile.z + 2 * chunk);

    ++traffic.chunks;
    ++traffic.pde_launches;
    ++traffic.tail_launches;
    traffic.max_chunk = std::max(traffic.max_chunk, chunk);
    traffic.tile_values = std::max(traffic.tile_values, staged);
    traffic.input_values += valid_tile_values;
    if (chebyshev && has_previous)
        traffic.input_values += valid_tile_values;
    // Every chunk writes its current column and each column it produces.
    // Chebyshev therefore rewrites the column at a chunk boundary.
    traffic.basis_write_values += N * static_cast<int64_t>(chunk + 1);

    if (basket) {
        for (int step = 1; step <= chunk; ++step) {
            const int halo = chunk - step;
            const int64_t vx = valid_extent_sum(n, tile.x, halo);
            const int64_t vy = valid_extent_sum(n, tile.y, halo);
            const int64_t vz = valid_extent_sum(n, tile.z, halo);
            const int64_t bx = upper_face_tile_count(n, tile.x, halo);
            const int64_t by = upper_face_tile_count(n, tile.y, halo);
            const int64_t bz = upper_face_tile_count(n, tile.z, halo);
            const int64_t face_visits =
                bx * vy * vz + by * vx * vz + bz * vx * vy;
            // Each visited upper face reads three coefficients and the three
            // augmented-tail values paired with them.
            traffic.face_values += 3 * face_visits;
            traffic.boundary_tail_values += 3 * face_visits;
        }
    }

    if (chebyshev) {
        // current, optional previous, output column zero, and chunk outputs.
        traffic.tail_kernel_values +=
            3 + (has_previous ? 3 : 0) + 3 + 3 * chunk;
    }
}

/// The whole basis evaluation's traffic, launch by launch.
///
/// In the full-volume family, monomial and Newton run one launch for the full
/// width, while Chebyshev needs two predecessors live, so its shared tile holds
/// three buffers and the width is walked in chunks. The plane-streamed family
/// pays a third predecessor as one more plane per level rather than a third
/// volume, so it is never chunked and Chebyshev costs it one launch too.
[[nodiscard]] TrafficModel traffic_model(
    int n, const TileGeometry& tile, int steps, PolynomialBasis basis,
    bool basket, bool chunked)
{
    TrafficModel traffic;
    const bool chebyshev = basis == PolynomialBasis::Chebyshev;
    traffic.shared_buffers = chebyshev ? 3 : 2;
    if (chebyshev && chunked) {
        int offset = 0;
        while (offset < steps) {
            const int chunk =
                std::min(kMpkPreferredS, steps - offset);
            add_chunk_traffic(
                traffic, n, tile, chunk, true, offset > 0, basket);
            offset += chunk;
        }
        return traffic;
    }
    // add_chunk_traffic already charges the Chebyshev tail, which is the same
    // accounting whether or not the width was walked in chunks.
    add_chunk_traffic(traffic, n, tile, steps, chebyshev, false, basket);
    if (!chebyshev) {
        // start tail + column-zero tail write, then one read/write pair per
        // recurrence.  Newton additionally reads one scalar shift per step.
        traffic.tail_kernel_values =
            6 + 6 * static_cast<int64_t>(steps)
            + (basis == PolynomialBasis::Newton ? steps : 0);
    }
    return traffic;
}

/// A device name with its spaces closed up, for a whitespace-delimited record.
///
/// The marker lines are parsed field by field on whitespace, so a value that
/// contains a space would silently become two fields and shift every column
/// after it. Substituting rather than quoting keeps the parsers unchanged.
[[nodiscard]] std::string record_token(const char* text)
{
    std::string token = text != nullptr ? text : "";
    if (token.empty()) return "unspecified";
    for (char& character : token)
        if (character == ' ' || character == ',') character = '_';
    return token;
}

/// The tile the selected family stages around each interior point.
[[nodiscard]] TileGeometry tile_geometry(const Args& args)
{
    if (args.family == MpkKernelFamily::PlaneStreamed) {
        return TileGeometry{
            kMpkStreamTileX, kMpkStreamTileY,
            mpk_stream_effective_height(args.n, args.stream_height)};
    }
    return TileGeometry{kMpkBlockX, kMpkBlockY, kMpkBlockZ};
}

/// Thread blocks one PDE launch dispatches over an n-cubed grid.
[[nodiscard]] int64_t grid_blocks_for(int n, const TileGeometry& tile)
{
    return static_cast<int64_t>((n + tile.x - 1) / tile.x)
         * static_cast<int64_t>((n + tile.y - 1) / tile.y)
         * static_cast<int64_t>((n + tile.z - 1) / tile.z);
}

/**
 * Scatter a logically indexed column into the physical layout the kernel reads.
 *
 * The two coincide unless a padded row stride is compiled in, in which case the
 * physical column is longer and carries gaps between rows. Those gaps are
 * written zero here and never written again by any kernel, which is the
 * precondition under which a whole-column dot product or norm still reduces over
 * the logical field alone. padding_residual below is what checks it held.
 */
void scatter_to_physical(
    const Eigen::VectorXd& logical, int n, int64_t physical_ld,
    std::vector<double>& physical)
{
    physical.assign(static_cast<std::size_t>(physical_ld), 0.0);
    const int pitch = mpk_pitch_x(n);
    const int64_t logical_N =
        static_cast<int64_t>(n) * static_cast<int64_t>(n)
        * static_cast<int64_t>(n);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                physical[static_cast<std::size_t>(
                    mpk_field_index(i, j, k, n, pitch))] =
                        logical[(static_cast<int64_t>(k) * n + j) * n + i];
    for (int c = 0; c < 3; ++c)
        physical[static_cast<std::size_t>(mpk_physical_length(n) + c)] =
            logical[logical_N + c];
}

/// The inverse, so a physical column can be compared with a logical reference.
void gather_to_logical(const double* physical, int n, Eigen::VectorXd& logical)
{
    const int pitch = mpk_pitch_x(n);
    const int64_t logical_N =
        static_cast<int64_t>(n) * static_cast<int64_t>(n)
        * static_cast<int64_t>(n);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                logical[(static_cast<int64_t>(k) * n + j) * n + i] =
                    physical[mpk_field_index(i, j, k, n, pitch)];
    for (int c = 0; c < 3; ++c)
        logical[logical_N + c] = physical[mpk_physical_length(n) + c];
}

/// Largest magnitude anywhere in the padded gaps of a physical column.
///
/// Reported rather than tolerated. A padded layout may only share the solver's
/// whole-column reductions while this is exactly zero; anything else means the
/// reductions have to be rewritten to walk logical rows.
[[nodiscard]] double padding_residual(const double* physical, int n)
{
    if (mpk_pad_length(n) == 0) return 0.0;
    const int pitch = mpk_pitch_x(n);
    double worst = 0.0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = n; i < pitch; ++i)
                worst = std::max(
                    worst, std::fabs(physical[mpk_field_index(i, j, k, n, pitch)]));
    return worst;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Args args = parse_args(argc, argv);
        const CaPricingModel model;
        // The full-volume Chebyshev kernel is excluded from the qualifier
        // because a chunk boundary passes the same address as input and output.
        // The streamed family never chunks, so its Chebyshev path carries it.
        const bool restricted_kernel =
            kMpkRestrictEnabled
            && (args.family == MpkKernelFamily::PlaneStreamed
                || args.basis != PolynomialBasis::Chebyshev);

        CUDA_CHECK(cudaSetDevice(args.device));
        const DeviceContention contention = check_device_contention();
        report_toolkit();
        report_contention(contention);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

        const bool streamed = args.family == MpkKernelFamily::PlaneStreamed;
        const bool chebyshev = args.basis == PolynomialBasis::Chebyshev;
        const TileGeometry tile = tile_geometry(args);

        // A swept tile geometry can ask for more shared memory than the device
        // will opt into. That is a property of the geometry, not a failure of the
        // run, so it is reported and separated from a numerical failure by its
        // exit code rather than crashing inside cudaFuncSetAttribute.
        const std::size_t requested_shared =
            streamed
                ? mpk_stream_shared_bytes(args.s, chebyshev)
                : (chebyshev
                       ? gpu_pde_chebyshev_shared_bytes(args.s)
                       : gpu_pde_matrix_powers_shared_bytes(args.s));
        if (requested_shared > prop.sharedMemPerBlockOptin) {
            std::printf(
                "CA polynomial basis\n"
                "  device: %s\n"
                "  %s tile %dx%dx%d at s=%d needs %.1f KiB shared, above the %.1f KiB opt-in limit\n"
                "  verdict: geometry not admissible at this width\n",
                prop.name, mpk_family_name(args.family),
                tile.x, tile.y, tile.z, args.s,
                static_cast<double>(requested_shared) / 1024.0,
                static_cast<double>(prop.sharedMemPerBlockOptin) / 1024.0);
            return 2;
        }

        const PDESystem sys = build_pde_system(
            args.n, model.strike, model.rate, model.expiry,
            model.sigma, model.rho, model.weight, model.spot,
            model.alpha, args.rainbow);
        const GpuPdeOperator op = make_gpu_pde_operator(sys, model, args.rainbow);
        // Two leading dimensions, because the reference and the device need not
        // agree on the layout. The assembled operator is logical by construction,
        // so every Eigen quantity below is sized logically; ld is what the device
        // columns actually occupy, which is longer under a padded row stride.
        const int64_t logical_ld = static_cast<int64_t>(sys.N) + 3;
        const int64_t ld = mpk_column_length(args.n);

        const SpMat A = args.rainbow ? sys.A : build_A_tilde(sys.A, sys.B, sys.N);
        Eigen::VectorXd start(logical_ld);
        start.setZero();
        start.head(sys.N) = sys.u0;
        if (!args.rainbow) start.tail(3) = make_s_vec(0.0);
        start.normalize();

        const GershgorinInterval interval =
            scaled_gershgorin_interval(sys.A, args.scale, true);
        const std::vector<double> shifts =
            real_leja_shifts(interval, args.s);
        const double newton_normalization =
            0.5 * (interval.upper - interval.lower);
        const double chebyshev_center =
            0.5 * (interval.lower + interval.upper);
        const double chebyshev_half_width =
            0.5 * (interval.upper - interval.lower);
        if (args.basis == PolynomialBasis::Chebyshev
            && !(chebyshev_half_width > 0.0))
            throw std::runtime_error("Chebyshev enclosure has zero width");

        if (!(newton_normalization > 0.0))
            throw std::runtime_error("spectral enclosure has zero width");

        Eigen::MatrixXd monomial_reference(logical_ld, args.s + 1);
        Eigen::MatrixXd newton_reference(logical_ld, args.s + 1);
        Eigen::MatrixXd chebyshev_reference(logical_ld, args.s + 1);
        monomial_reference.col(0) = start;
        newton_reference.col(0) = start;
        chebyshev_reference.col(0) = start;
        auto apply_scaled = [&](const Eigen::Ref<const Eigen::VectorXd>& x) {
            Eigen::VectorXd y = Eigen::VectorXd::Zero(logical_ld);
            if (args.rainbow)
                y.head(sys.N).noalias() = args.scale * (A * x.head(sys.N));
            else
                y.noalias() = args.scale * (A * x);
            return y;
        };
        for (int q = 1; q <= args.s; ++q) {
            monomial_reference.col(q) =
                apply_scaled(monomial_reference.col(q - 1));
            newton_reference.col(q) =
                (apply_scaled(newton_reference.col(q - 1))
                 - shifts[static_cast<std::size_t>(q - 1)]
                       * newton_reference.col(q - 1))
                / newton_normalization;
            {
                Eigen::VectorXd x =
                    (apply_scaled(chebyshev_reference.col(q - 1))
                     - chebyshev_center * chebyshev_reference.col(q - 1))
                    / chebyshev_half_width;
                chebyshev_reference.col(q) =
                    q == 1 ? x : 2.0 * x - chebyshev_reference.col(q - 2);
            }
        }
        const Eigen::MatrixXd& reference =
            args.basis == PolynomialBasis::Newton
                ? newton_reference
                : (args.basis == PolynomialBasis::Chebyshev
                       ? chebyshev_reference : monomial_reference);
        const double monomial_kappa = condition_number(monomial_reference);
        const double newton_kappa = condition_number(newton_reference);
        const double chebyshev_kappa = condition_number(chebyshev_reference);
        const double basis_kappa = condition_number(reference);
        const int block_width = args.certificate_block_width;
        const Eigen::MatrixXd monomial_block =
            monomial_reference.leftCols(block_width);
        const Eigen::MatrixXd newton_block =
            newton_reference.leftCols(block_width);
        const Eigen::MatrixXd chebyshev_block =
            chebyshev_reference.leftCols(block_width);
        const double block_monomial_kappa =
            condition_number(monomial_block);
        const double block_newton_kappa =
            condition_number(newton_block);
        const double block_chebyshev_kappa =
            condition_number(chebyshev_block);
        const double block_basis_kappa =
            args.basis == PolynomialBasis::Newton
                ? block_newton_kappa
                : (args.basis == PolynomialBasis::Chebyshev
                       ? block_chebyshev_kappa : block_monomial_kappa);

        // The a-priori certified width. The pricing operator is non-normal and its
        // n-cubed rows are not diagonalized here, so the prediction samples the
        // scaled Gershgorin enclosure uniformly. It is a prediction from the
        // enclosure, not from the spectrum, and is reported as such.
        constexpr int enclosure_samples = 512;
        Eigen::VectorXd enclosure(enclosure_samples);
        Eigen::VectorXd weights(enclosure_samples);
        for (int q = 0; q < enclosure_samples; ++q) {
            enclosure[q] =
                interval.lower
                + (interval.upper - interval.lower)
                    * static_cast<double>(q)
                    / static_cast<double>(enclosure_samples - 1);
            weights[q] = 1.0 / std::sqrt(static_cast<double>(enclosure_samples));
        }
        const int enclosure_s_max =
            predicted_s_max(enclosure, weights, kMpkMaxS);
        // Ask the same enclosure-only question for all three recurrences. The
        // monomial degree result above remains the compatibility field; the new
        // fields use block width directly, matching what the integrator requests.
        const std::vector<double> prediction_shifts =
            real_leja_shifts(interval, kMpkMaxS);
        const Eigen::VectorXd normalized_enclosure =
            (enclosure.array() - chebyshev_center)
            / chebyshev_half_width;
        Eigen::MatrixXd predicted_monomial(
            enclosure_samples, kMpkMaxS + 1);
        Eigen::MatrixXd predicted_newton(
            enclosure_samples, kMpkMaxS + 1);
        Eigen::MatrixXd predicted_chebyshev(
            enclosure_samples, kMpkMaxS + 1);
        predicted_monomial.col(0) = weights;
        predicted_newton.col(0) = weights;
        predicted_chebyshev.col(0) = weights;
        for (int q = 1; q <= kMpkMaxS; ++q) {
            predicted_monomial.col(q) =
                predicted_monomial.col(q - 1).array()
                * enclosure.array();
            predicted_newton.col(q) =
                predicted_newton.col(q - 1).array()
                * (enclosure.array()
                   - prediction_shifts[static_cast<std::size_t>(q - 1)])
                / newton_normalization;
            if (q == 1) {
                predicted_chebyshev.col(q) =
                    normalized_enclosure.array() * weights.array();
            } else {
                predicted_chebyshev.col(q) =
                    2.0 * normalized_enclosure.array()
                        * predicted_chebyshev.col(q - 1).array()
                    - predicted_chebyshev.col(q - 2).array();
            }
        }
        auto predicted_certified_block_width =
            [](const Eigen::MatrixXd& candidates) {
                int width_max = 1;
                for (int width = 2; width <= kMpkMaxS; ++width) {
                    const Eigen::MatrixXd prefix =
                        candidates.leftCols(width);
                    if (condition_number(prefix) < cholqr_kappa_limit())
                        width_max = width;
                }
                return width_max;
            };
        const int predicted_monomial_block_width_max =
            predicted_certified_block_width(predicted_monomial);
        const int predicted_newton_block_width_max =
            predicted_certified_block_width(predicted_newton);
        const int predicted_chebyshev_block_width_max =
            predicted_certified_block_width(predicted_chebyshev);
        const int predicted_block_width_max =
            args.basis == PolynomialBasis::Newton
                ? predicted_newton_block_width_max
                : (args.basis == PolynomialBasis::Chebyshev
                       ? predicted_chebyshev_block_width_max
                       : predicted_monomial_block_width_max);
        // The same question asked of the basis this harness actually built.
        int reference_s_max = 0;
        for (int q = 1; q <= args.s; ++q) {
            const Eigen::MatrixXd prefix = reference.leftCols(q + 1);
            if (condition_number(prefix) < cholqr_kappa_limit())
                reference_s_max = q;
        }
        const int basis_block_width_max =
            std::min(kMpkMaxS, reference_s_max + 1);

        const std::vector<double> face_b = make_gpu_face_b(sys, model);
        // The start vector in the layout the kernel reads. Under a padded row
        // stride this also establishes the invariant the padded arm depends on:
        // every gap starts at exactly zero and no kernel writes one afterwards.
        std::vector<double> physical_start;
        scatter_to_physical(start, args.n, ld, physical_start);

        double *d_start = nullptr, *d_B = nullptr;
        double *d_face_b = nullptr, *d_shifts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_start, static_cast<std::size_t>(ld) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            &d_B, static_cast<std::size_t>(ld) * static_cast<std::size_t>(args.s + 1)
                * sizeof(double)));
        // Basis columns start zeroed so a padded gap the kernel never writes
        // reads back as zero rather than as whatever the allocator returned.
        CUDA_CHECK(cudaMemset(
            d_B, 0,
            static_cast<std::size_t>(ld)
                * static_cast<std::size_t>(args.s + 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(
            d_start, physical_start.data(),
            static_cast<std::size_t>(ld) * sizeof(double),
            cudaMemcpyHostToDevice));
        if (!args.rainbow) {
            CUDA_CHECK(cudaMalloc(&d_face_b, face_b.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(
                d_face_b, face_b.data(), face_b.size() * sizeof(double),
                cudaMemcpyHostToDevice));
        }
        if (args.basis == PolynomialBasis::Newton) {
            CUDA_CHECK(cudaMalloc(
                &d_shifts, static_cast<std::size_t>(args.s) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(
                d_shifts, shifts.data(),
                static_cast<std::size_t>(args.s) * sizeof(double),
                cudaMemcpyHostToDevice));
        }

        const TrafficModel traffic = traffic_model(
            args.n, tile, args.s, args.basis, !args.rainbow, !streamed);
        auto launch_basis = [&]() {
            if (streamed) {
                if (args.basis == PolynomialBasis::Newton) {
                    return gpu_pde_stream_newton_basis(
                        d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                        d_shifts, newton_normalization, args.stream_height,
                        nullptr, args.shared_carveout);
                }
                if (chebyshev) {
                    return gpu_pde_stream_chebyshev_basis(
                        d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                        chebyshev_center, chebyshev_half_width,
                        args.stream_height, nullptr, args.shared_carveout);
                }
                return gpu_pde_stream_matrix_powers(
                    d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                    args.stream_height, nullptr, args.shared_carveout);
            }
            if (args.basis == PolynomialBasis::Newton) {
                return gpu_pde_newton_basis(
                    d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                    d_shifts, newton_normalization, nullptr,
                    args.shared_carveout);
            }
            if (chebyshev) {
                return gpu_pde_chebyshev_basis(
                    d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                    chebyshev_center, chebyshev_half_width, nullptr,
                    args.shared_carveout);
            }
            return gpu_pde_matrix_powers(
                d_start, d_B, ld, args.s, op, d_face_b, args.scale,
                nullptr, args.shared_carveout);
        };
        auto release_device = [&]() {
            CUDA_CHECK(cudaFree(d_start));
            CUDA_CHECK(cudaFree(d_B));
            if (d_face_b != nullptr) CUDA_CHECK(cudaFree(d_face_b));
            if (d_shifts != nullptr) CUDA_CHECK(cudaFree(d_shifts));
        };

        if (args.profile_only) {
            // No GPU kernel has run before this point.  Keep this branch separate
            // from both validation and timing so ncu sees exactly one complete
            // basis evaluation (one or more recurrence chunks, as reported).
            CUDA_CHECK(cudaDeviceSynchronize());
            const cudaError_t profiled = launch_basis();
            if (profiled != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "Profile-only basis evaluation failed: %s\n",
                    cudaGetErrorString(profiled));
                release_device();
                return profiled == cudaErrorInvalidValue ? 2 : EXIT_FAILURE;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::printf(
                "MPK_PROFILE basis_evaluations=1 basis=%s n=%d s=%d "
                "chunks=%d pde_launches=%d tail_launches=%d "
                "shared_carveout=%s shared_pad_x=%d family=%s "
                "tile=%dx%dx%d threads=%d stream_height=%d contended=%d\n",
                basis_name(args.basis), args.n, args.s, traffic.chunks,
                traffic.pde_launches, traffic.tail_launches,
                carveout_name(args.shared_carveout), kMpkSharedPadX,
                mpk_family_name(args.family), tile.x, tile.y, tile.z,
                streamed ? kMpkStreamThreads : kMpkThreadsPerBlock,
                streamed ? tile.z : 0,
                contention.contended ? 1 : 0);
            release_device();
            return EXIT_SUCCESS;
        }

        // The device reports an opt-in shared-memory limit, but a request at exactly
        // that limit can still be refused once the reserved allocation is counted.
        // A swept geometry must report that as inadmissible, not die inside a check.
        const cudaError_t launched = launch_basis();
        if (launched == cudaErrorInvalidValue) {
            (void)cudaGetLastError();
            std::printf(
                "CA polynomial basis\n"
                "  device: %s\n"
                "  %s tile %dx%dx%d at s=%d requested %.1f KiB shared and the device "
                "refused it, against a reported %.1f KiB opt-in limit\n"
                "  verdict: geometry not admissible at this width\n",
                prop.name, mpk_family_name(args.family),
                tile.x, tile.y, tile.z, args.s,
                static_cast<double>(requested_shared) / 1024.0,
                static_cast<double>(prop.sharedMemPerBlockOptin) / 1024.0);
            return 2;
        }
        CUDA_CHECK(launched);
        CUDA_CHECK(cudaDeviceSynchronize());

        Eigen::MatrixXd measured(ld, args.s + 1);
        CUDA_CHECK(cudaMemcpy(
            measured.data(), d_B,
            static_cast<std::size_t>(ld) * static_cast<std::size_t>(args.s + 1)
                * sizeof(double),
            cudaMemcpyDeviceToHost));

        double worst_abs = 0.0;
        double worst_rel = 0.0;
        double worst_padding = 0.0;
        int worst_power = 0;
        bool finite_errors = true;
        // Kept per column rather than only as a maximum, because the operator
        // gate is stated per column: an error that grows with the power is a
        // different failure from one that appears at a single boundary row.
        std::vector<double> column_abs;
        std::vector<double> column_rel;
        column_abs.reserve(static_cast<std::size_t>(args.s + 1));
        column_rel.reserve(static_cast<std::size_t>(args.s + 1));
        // The device columns are physical and the reference is logical, so each
        // column is packed before it is compared. The two coincide unless a
        // padded row stride is compiled in.
        Eigen::VectorXd packed(logical_ld);
        for (int q = 0; q <= args.s; ++q) {
            gather_to_logical(measured.col(q).data(), args.n, packed);
            worst_padding = std::max(
                worst_padding, padding_residual(measured.col(q).data(), args.n));
            const double abs_err =
                (packed - reference.col(q)).lpNorm<Eigen::Infinity>();
            const double denom = reference.col(q).lpNorm<Eigen::Infinity>();
            const double rel_err = denom > 0.0 ? abs_err / denom : abs_err;
            column_abs.push_back(abs_err);
            column_rel.push_back(rel_err);
            if (!std::isfinite(abs_err) || !std::isfinite(rel_err)) {
                finite_errors = false;
                worst_abs = abs_err;
                worst_rel = rel_err;
                worst_power = q;
                break;
            }
            if (rel_err > worst_rel) {
                worst_abs = abs_err;
                worst_rel = rel_err;
                worst_power = q;
            }
        }

        // The per-column record the operator gate is stated in. Emitted before
        // the verdict so a failing run still shows which power the error entered
        // at, which is what distinguishes a boundary-row mistake from an error
        // that compounds through the recurrence.
        for (std::size_t q = 0; q < column_abs.size(); ++q) {
            std::printf(
                "MPK_COLUMN family=%s option=%s basis=%s n=%d s=%d column=%zu "
                "abs=%.9e rel=%.9e\n",
                mpk_family_name(args.family),
                args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
                args.n, args.s, q, column_abs[q], column_rel[q]);
        }

        // A geometry that alters numerics is never timed: the comparison against
        // the assembled operator gates the timing loop, not the exit code. A
        // padded layout carries one further condition, that its gaps stayed
        // exactly zero, because that is what lets a reduction ignore them.
        const bool correct =
            finite_errors && (worst_rel <= 5e-11 || worst_abs <= 5e-13)
            && worst_padding == 0.0;
        if (!correct) {
            std::fprintf(
                stderr,
                "FAIL: matrix-powers mismatch at %s tile %dx%dx%d, s=%d: "
                "abs=%.6e rel=%.6e at column %d, padding residual %.6e\n",
                mpk_family_name(args.family), tile.x, tile.y, tile.z, args.s,
                worst_abs, worst_rel, worst_power, worst_padding);
            return EXIT_FAILURE;
        }

        cudaEvent_t begin{}, end{};
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(args.repeats));
        for (int q = 0; q < args.repeats; ++q) {
            CUDA_CHECK(cudaEventRecord(begin));
            CUDA_CHECK(launch_basis());
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            timings.push_back(static_cast<double>(ms) * 1e-3);
        }
        const double seconds = median(timings);
        const auto timing_extrema =
            std::minmax_element(timings.begin(), timings.end());
        const double seconds_min = *timing_extrema.first;
        const double seconds_max = *timing_extrema.second;

        // Compulsory traffic: the start vector read once, the s+1 basis columns
        // written once. No implementation of this recurrence can move less.
        const double compulsory_bytes =
            8.0 * static_cast<double>(sys.N) * static_cast<double>(args.s + 2);
        // What the implementation asks the memory hierarchy for.  Unlike the
        // old full-tile estimate, this excludes out-of-domain locations that the
        // kernel fills with zero and includes Basket face/tail reads.  Chebyshev
        // is modeled chunk by chunk, including its second input after chunk zero.
        const int64_t grid_blocks = grid_blocks_for(args.n, tile);
        const int64_t interior_points =
            static_cast<int64_t>(tile.x) * tile.y * tile.z;
        const double redundancy =
            static_cast<double>(traffic.tile_values)
            / static_cast<double>(interior_points);
        const double effective_read_redundancy =
            static_cast<double>(traffic.input_values)
            / static_cast<double>(sys.N);
        const double input_bytes =
            8.0 * static_cast<double>(traffic.input_values);
        const double basis_write_bytes =
            8.0 * static_cast<double>(traffic.basis_write_values);
        const double face_bytes =
            8.0 * static_cast<double>(traffic.face_values);
        const double boundary_tail_bytes =
            8.0 * static_cast<double>(traffic.boundary_tail_values);
        const double tail_kernel_bytes =
            8.0 * static_cast<double>(traffic.tail_kernel_values);
        const double tiled_bytes =
            input_bytes + basis_write_bytes + face_bytes
            + boundary_tail_bytes + tail_kernel_bytes;
        const double compulsory_gbs =
            seconds > 0.0 ? compulsory_bytes / seconds * 1e-9 : 0.0;
        const double tiled_gbs =
            seconds > 0.0 ? tiled_bytes / seconds * 1e-9 : 0.0;

        const GpuMachine& machine = lookup_gpu_machine_for_device(prop.name);
        const double dram_roof_gbs =
            machine.hbm_bw_gbs_achieved > 0.0
                ? machine.hbm_bw_gbs_achieved : machine.hbm_bw_gbs;
        const double l2_roof_gbs = machine.l2_bw_gbs_achieved;
        // With an ncu sector count the denominator is a measurement.  Without
        // one the rates remain a modeled diagnostic and no verdict is assigned.
        const bool measured_dram = args.ncu_dram_bytes > 0.0;
        const bool recordable_dram =
            measured_dram && !contention.contended;
        const double achieved_dram_bytes =
            measured_dram ? args.ncu_dram_bytes : tiled_bytes;
        const double achieved_dram_gbs =
            seconds > 0.0 ? achieved_dram_bytes / seconds * 1e-9 : 0.0;
        const double dram_fraction =
            dram_roof_gbs > 0.0 ? achieved_dram_gbs / dram_roof_gbs : 0.0;
        const double l2_fraction =
            l2_roof_gbs > 0.0 ? tiled_gbs / l2_roof_gbs : 0.0;
        const int launches = traffic.pde_launches + traffic.tail_launches;
        const double launch_fraction =
            seconds > 0.0
                ? static_cast<double>(launches) * machine.t_kernel_launch_s
                    / seconds
                : 0.0;

        const int threads_per_block =
            streamed ? kMpkStreamThreads : kMpkThreadsPerBlock;
        int active_blocks = 0;
        cudaFuncAttributes function_attributes{};
        if (streamed) {
            CUDA_CHECK(gpu_pde_stream_occupancy(
                args.s, chebyshev, &active_blocks, args.shared_carveout));
            CUDA_CHECK(gpu_pde_stream_function_attributes(
                args.s, chebyshev, &function_attributes,
                args.shared_carveout));
        } else {
            CUDA_CHECK(gpu_pde_basis_occupancy(
                args.s, chebyshev, &active_blocks, args.shared_carveout));
            CUDA_CHECK(gpu_pde_basis_function_attributes(
                args.s, chebyshev, &function_attributes,
                args.shared_carveout));
        }
        const double occupancy =
            static_cast<double>(active_blocks * threads_per_block)
            / static_cast<double>(prop.maxThreadsPerMultiProcessor);
        // Times the grid covers the device at this residency. Below one the
        // launch cannot fill it however high the occupancy per block is, which
        // is the term a long streamed segment trades redundancy against.
        const double waves =
            active_blocks > 0 && prop.multiProcessorCount > 0
                ? static_cast<double>(grid_blocks)
                    / static_cast<double>(
                          static_cast<int64_t>(active_blocks)
                          * prop.multiProcessorCount)
                : 0.0;
        const double points_per_thread =
            static_cast<double>(interior_points)
            / static_cast<double>(threads_per_block);
        // Block-wide barriers per block. The streamed family synchronizes once
        // per pipeline step rather than twice per recurrence step, so the two
        // families are not comparable per launch and the count is reported as
        // its own quantity rather than folded into a rate.
        // Counted at the chunk width rather than the requested width, because a
        // chunked Chebyshev launch runs its own shorter recurrence.
        const int64_t barriers_per_block =
            streamed
                ? mpk_stream_barriers_per_block(traffic.max_chunk, tile.z)
                    * static_cast<int64_t>(traffic.pde_launches)
                : mpk_barriers_per_block(traffic.max_chunk)
                    * static_cast<int64_t>(traffic.pde_launches);
        const int64_t staged_points = traffic.tile_values;
        const double redundant_fraction =
            static_cast<double>(staged_points - interior_points)
            / static_cast<double>(interior_points);

        // A modeled byte count is useful for a smoke test, but it cannot close
        // the vertical coordinate, so a causal verdict needs a measured DRAM
        // count first. The tests are roofs only: a kernel far from every roof is
        // limited by its access pattern, and saying so is more honest than
        // picking whichever threshold happens to trip.
        VerticalVerdict verdict = VerticalVerdict::Undetermined;
        if (recordable_dram) {
            if (launch_fraction >= 0.25)
                verdict = VerticalVerdict::LaunchBound;
            else if (dram_fraction >= 0.75)
                verdict = VerticalVerdict::DramBound;
            else if (l2_fraction >= 0.75)
                verdict = VerticalVerdict::L2Bound;
            else
                verdict = VerticalVerdict::LatencyBound;
        }
        // Traffic attribution, reported beside the verdict and never folded into
        // it. Below one the tiling's redundant reads are absorbed before DRAM;
        // above one the model under-counts, which narrow tiles do because a tile
        // row spans a partial sector.
        const double measured_over_modeled =
            tiled_bytes > 0.0 ? achieved_dram_bytes / tiled_bytes : 0.0;
        // A latency-bound verdict is the same word at 55% of the DRAM roof and
        // at 2%, so the nearest roof and its distance are reported beside it.
        // This introduces no threshold: it names the larger of the two measured
        // fractions, which is what the verdict already tested and discarded.
        const char* nearest_roof =
            dram_fraction >= l2_fraction ? "DRAM" : "L2";
        const double nearest_roof_fraction =
            std::max(dram_fraction, l2_fraction);

        std::printf("CA polynomial basis\n");
        std::printf("  device: %s\n", prop.name);
        std::printf("  option: %s | basis=%s | n=%d | N=%d | s=%d | scale=%.3e\n",
                    args.rainbow ? "rainbow" : "basket",
                    basis_name(args.basis),
                    args.n, sys.N, args.s, args.scale);
        if (args.basis == PolynomialBasis::Newton) {
            std::printf(
                "  spectral enclosure: [%.6e, %.6e] | normalization: %.6e | shifts:",
                interval.lower, interval.upper, newton_normalization);
            for (double shift : shifts) std::printf(" %.3e", shift);
            std::printf("\n");
        } else if (args.basis == PolynomialBasis::Chebyshev) {
            std::printf(
                "  spectral enclosure: [%.6e, %.6e] | center: %.6e | half-width: %.6e\n",
                interval.lower, interval.upper,
                chebyshev_center, chebyshev_half_width);
        }
        std::printf("  shared memory/block: %.1f KiB | device opt-in limit: %.1f KiB\n",
                    static_cast<double>(requested_shared) / 1024.0,
                    static_cast<double>(prop.sharedMemPerBlockOptin) / 1024.0);
        std::printf(
            "  kernel arm: family=%s | restrict=%s | shared pad x=%d | "
            "carveout requested=%s reported=%d | barrier elision=%s | "
            "pitch alignment=%d doubles\n",
            mpk_family_name(args.family),
            restricted_kernel ? "on" : "off", kMpkSharedPadX,
            carveout_name(args.shared_carveout),
            function_attributes.preferredShmemCarveout,
            kMpkElideBasisBarrier ? "on" : "off", kMpkPitchAlignment);
        std::printf(
            "  launch: %d threads/block | %.2f interior points/thread | "
            "%d registers/thread | %d B local frame | %zu B static shared\n",
            threads_per_block, points_per_thread, function_attributes.numRegs,
            static_cast<int>(function_attributes.localSizeBytes),
            function_attributes.sharedSizeBytes);
        std::printf(
            "  occupancy: %d blocks/SM | %.1f%% resident threads | "
            "%.2f waves over %d SMs\n",
            active_blocks, 100.0 * occupancy, waves,
            prop.multiProcessorCount);
        std::printf("  basis kappa: %.6e\n", basis_kappa);
        std::printf(
            "  comparison kappa: monomial=%.6e newton=%.6e chebyshev=%.6e\n",
            monomial_kappa, newton_kappa, chebyshev_kappa);
        std::printf(
            "  certified width: enclosure prediction=%d | this basis=%d | certificate limit=%.6e\n",
            enclosure_s_max, reference_s_max, cholqr_kappa_limit());
        std::printf(
            "  block-width view: requested=%d | enclosure prediction=%d | "
            "this basis=%d\n",
            block_width, predicted_block_width_max, basis_block_width_max);
        std::printf(
            "  predicted block widths: monomial=%d newton=%d chebyshev=%d\n",
            predicted_monomial_block_width_max,
            predicted_newton_block_width_max,
            predicted_chebyshev_block_width_max);
        std::printf(
            "  requested-prefix kappa: this basis=%.6e | monomial=%.6e "
            "newton=%.6e chebyshev=%.6e\n",
            block_basis_kappa, block_monomial_kappa,
            block_newton_kappa, block_chebyshev_kappa);
        std::printf(
            "MPK_CERTIFICATE option=%s basis=%s n=%d s=%d predicted_s_max=%d "
            "basis_s_max=%d basis_kappa=%.9e monomial_kappa=%.9e "
            "newton_kappa=%.9e chebyshev_kappa=%.9e limit=%.9e contended=%d "
            "block_width=%d predicted_block_width_max=%d "
            "basis_block_width_max=%d "
            "predicted_monomial_block_width_max=%d "
            "predicted_newton_block_width_max=%d "
            "predicted_chebyshev_block_width_max=%d "
            "block_basis_kappa=%.9e block_monomial_kappa=%.9e "
            "block_newton_kappa=%.9e block_chebyshev_kappa=%.9e\n",
            args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
            args.n, args.s, enclosure_s_max, reference_s_max, basis_kappa,
            monomial_kappa, newton_kappa, chebyshev_kappa,
            cholqr_kappa_limit(), contention.contended ? 1 : 0,
            block_width, predicted_block_width_max, basis_block_width_max,
            predicted_monomial_block_width_max,
            predicted_newton_block_width_max,
            predicted_chebyshev_block_width_max,
            block_basis_kappa, block_monomial_kappa,
            block_newton_kappa, block_chebyshev_kappa);
        std::printf(
            "  worst error: abs=%.6e rel=%.6e at column %d | "
            "padding residual %.6e\n",
            worst_abs, worst_rel, worst_power, worst_padding);
        std::printf(
            "  tile: %dx%dx%d = %lld interior points | largest staged tile %lld doubles | "
            "nominal redundancy %.2fx (%.2f ghost/interior) | %lld blocks/PDE launch\n",
            tile.x, tile.y, tile.z, static_cast<long long>(interior_points),
            static_cast<long long>(staged_points), redundancy,
            redundant_fraction, static_cast<long long>(grid_blocks));
        std::printf(
            "  execution: chunks=%d | chunk width max=%d | PDE launches=%d | "
            "tail launches=%d | shared buffers=%d\n",
            traffic.chunks, traffic.max_chunk, traffic.pde_launches,
            traffic.tail_launches, traffic.shared_buffers);
        std::printf(
            "  modeled traffic: valid tile reads %.3f MiB (%.2fx N) | "
            "basis writes %.3f MiB | Basket face %.3f MiB | "
            "Basket boundary tail %.3f MiB | tail kernels %.3f MiB\n",
            input_bytes / (1024.0 * 1024.0), effective_read_redundancy,
            basis_write_bytes / (1024.0 * 1024.0),
            face_bytes / (1024.0 * 1024.0),
            boundary_tail_bytes / (1024.0 * 1024.0),
            tail_kernel_bytes / (1024.0 * 1024.0));
        std::printf(
            "  traffic totals: compulsory %.3f MiB | modeled %.3f MiB (%.2fx) | %s DRAM %.3f MiB\n",
            compulsory_bytes / (1024.0 * 1024.0),
            tiled_bytes / (1024.0 * 1024.0),
            compulsory_bytes > 0.0 ? tiled_bytes / compulsory_bytes : 0.0,
            measured_dram ? "ncu" : "modeled",
            achieved_dram_bytes / (1024.0 * 1024.0));
        std::printf(
            "  timing: min=%.6e median=%.6e max=%.6e s (%zu runs) | "
            "compulsory-traffic rate: %.2f GB/s | modeled-traffic rate: %.2f GB/s\n",
            seconds_min, seconds, seconds_max, timings.size(),
            compulsory_gbs, tiled_gbs);
        std::printf(
            "  roofs (%s): DRAM %.1f GB/s (%.1f%% reached) | L2 %.1f GB/s (%.1f%% reached) | launch share %.1f%%\n",
            std::string(machine.key).c_str(), dram_roof_gbs,
            100.0 * dram_fraction, l2_roof_gbs, 100.0 * l2_fraction,
            100.0 * launch_fraction);
        std::printf(
            "  traffic attribution: measured/modeled=%.3f (%s)\n",
            measured_over_modeled,
            measured_over_modeled < 1.0
                ? "redundant reads absorbed before DRAM"
                : "model under-counts; narrow tile rows span partial sectors");
        // A verdict is only ever assigned from an idle-device ncu byte count, so
        // saying so on every supported row is noise. The parenthetical is
        // reserved for the two ways a row falls short of carrying one.
        if (recordable_dram) {
            std::printf(
                "  vertical verdict: %s | nearest roof %s at %.1f%%\n",
                verdict_name(verdict), nearest_roof,
                100.0 * nearest_roof_fraction);
        } else {
            std::printf(
                "  vertical verdict: %s | nearest roof %s at %.1f%% (%s)\n",
                verdict_name(verdict), nearest_roof,
                100.0 * nearest_roof_fraction,
                measured_dram
                    ? "diagnostic only; the device is contended"
                    : "diagnostic only; ncu DRAM measurement required");
        }
        std::printf(
            "MPK_VERTICAL machine=%s option=%s basis=%s n=%d N=%d s=%d "
            "tile=%dx%dx%d interior=%d tile_doubles=%lld redundancy=%.6f "
            "effective_read_redundancy=%.6f blocks=%lld total_blocks=%lld "
            "chunks=%d chunk_max=%d pde_launches=%d tail_launches=%d "
            "shared_buffers=%d seconds=%.9e seconds_min=%.9e "
            "seconds_median=%.9e seconds_max=%.9e repeats=%zu "
            "compulsory_bytes=%.0f tiled_bytes=%.0f input_bytes=%.0f "
            "basis_write_bytes=%.0f face_bytes=%.0f boundary_tail_bytes=%.0f "
            "tail_kernel_bytes=%.0f "
            "dram_bytes=%.0f dram_source=%s compulsory_gbs=%.6f tiled_gbs=%.6f "
            "dram_roof_gbs=%.6f l2_roof_gbs=%.6f dram_fraction=%.6f "
            "l2_fraction=%.6f launch_fraction=%.6f occupancy=%.6f "
            "restrict=%d shared_pad_x=%d shared_carveout=%s "
            "reported_carveout=%d "
            "measured_over_modeled=%.6f nearest_roof=%s "
            "nearest_roof_fraction=%.6f verdict=%s "
            "contended=%d\n",
            std::string(machine.key).c_str(),
            args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
            args.n, sys.N, args.s,
            tile.x, tile.y, tile.z, static_cast<int>(interior_points),
            static_cast<long long>(traffic.tile_values), redundancy,
            effective_read_redundancy,
            static_cast<long long>(grid_blocks),
            static_cast<long long>(
                grid_blocks * static_cast<int64_t>(traffic.pde_launches)),
            traffic.chunks, traffic.max_chunk, traffic.pde_launches,
            traffic.tail_launches, traffic.shared_buffers,
            seconds, seconds_min, seconds, seconds_max, timings.size(),
            compulsory_bytes, tiled_bytes, input_bytes, basis_write_bytes,
            face_bytes, boundary_tail_bytes, tail_kernel_bytes,
            achieved_dram_bytes,
            measured_dram ? "ncu" : "modeled",
            compulsory_gbs, tiled_gbs, dram_roof_gbs, l2_roof_gbs,
            dram_fraction, l2_fraction, launch_fraction, occupancy,
            restricted_kernel ? 1 : 0, kMpkSharedPadX,
            carveout_name(args.shared_carveout),
            function_attributes.preferredShmemCarveout,
            measured_over_modeled, nearest_roof, nearest_roof_fraction,
            verdict_name(verdict), contention.contended ? 1 : 0);

        // The tuning record. Separate from MPK_VERTICAL because it answers a
        // different question: that record is about which roof the kernel is
        // near, this one is about which configuration produced the time, and a
        // sweep has to be able to attribute a row to a build without consulting
        // the runner's memory of which executable it invoked. Spill loads and
        // stores are not runtime-visible and come from the ptxas resource
        // report, so the local frame size stands in for them here.
        int runtime_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        std::printf(
            "MPK_TUNING configuration=%s family=%s tile_x=%d tile_y=%d tile_z=%d "
            "stream_height=%d threads_per_block=%d shared_pad_x=%d "
            "pitch_alignment=%d register_cap=%d elide_basis_barrier=%d "
            "restrict=%d option=%s basis=%s n=%d s=%d "
            "cuda_runtime=%d gpu_name=%s build_type=%s sm=%d%d "
            "registers_per_thread=%d local_frame_bytes=%d "
            "static_shared_bytes=%zu dynamic_shared_bytes=%zu "
            "shared_optin_bytes=%zu "
            "median_ms=%.6f min_ms=%.6f max_ms=%.6f repetitions=%zu "
            "useful_points=%lld staged_points=%lld redundant_fraction=%.6f "
            "points_per_thread=%.6f blocks=%lld active_blocks_per_sm=%d "
            "occupancy=%.6f waves=%.6f multiprocessors=%d "
            "pde_launches=%d tail_launches=%d chunks=%d "
            "barriers_per_block=%lld "
            "worst_rel=%.6e worst_abs=%.6e padding_residual=%.6e "
            "correctness_status=%s contended=%d\n",
            CAKSM_MPK_CONFIGURATION,
            mpk_family_name(args.family), tile.x, tile.y, tile.z,
            streamed ? tile.z : 0, threads_per_block, kMpkSharedPadX,
            kMpkPitchAlignment, CAKSM_MPK_REGISTER_CAP,
            kMpkElideBasisBarrier ? 1 : 0, restricted_kernel ? 1 : 0,
            args.rainbow ? "rainbow" : "basket", basis_name(args.basis),
            args.n, args.s, runtime_version,
            record_token(prop.name).c_str(),
            record_token(CAKSM_BUILD_TYPE).c_str(),
            prop.major, prop.minor,
            function_attributes.numRegs,
            static_cast<int>(function_attributes.localSizeBytes),
            function_attributes.sharedSizeBytes, requested_shared,
            static_cast<std::size_t>(prop.sharedMemPerBlockOptin),
            seconds * 1e3, seconds_min * 1e3, seconds_max * 1e3,
            timings.size(),
            static_cast<long long>(
                static_cast<int64_t>(sys.N) * (args.s + 1)),
            static_cast<long long>(staged_points), redundant_fraction,
            points_per_thread, static_cast<long long>(grid_blocks),
            active_blocks, occupancy, waves, prop.multiProcessorCount,
            traffic.pde_launches, traffic.tail_launches, traffic.chunks,
            static_cast<long long>(barriers_per_block),
            worst_rel, worst_abs, worst_padding, "pass",
            contention.contended ? 1 : 0);

        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
        release_device();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
