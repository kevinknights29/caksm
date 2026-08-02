/**
 * @file scaling.cpp
 * @brief OpenMP strong/weak-scaling harness for the baseline KSM-EI solver.
 *
 * This is a measurement instrument to show how the two hot kernels of
 * one Arnoldi cycle respond to added cores, at a frozen kernel mix, so that
 * any change across P is attributable to hardware.
 * It does not measure solution accuracy (convergence tolerance is not an observable here).
 *
 * What it implements, and why:
 *
 *   - Fixed Krylov dimension m (default 8), convergence check disabled, so every
 *     time step runs exactly m Arnoldi iterations. Freezing m freezes the GS/SpMV
 *     work ratio (MGS ~ m^2, SpMV ~ m), so the kernel mix does not drift across P or n.
 *
 *   - Hand-rolled CSR + row-partitioned SpMV. Each thread owns a contiguous,
 *     disjoint slice of the output vector y = Ax (sole writer, no atomics, clean
 *     first-touch), and partition seams are cache-line aligned (8 doubles / 64 B)
 *     to avoid false sharing.
 *
 *   - Parallel Modified Gram-Schmidt (MGS): each dot product and the final norm
 *     is a global reduction across threads, exactly the horizontal-communication
 *     cost the study exists to expose. The reduction comes from
 *     include/reduction.hpp, which scripts/regime/calibrate_alpha.sh also calibrates
 *     against; `--reduce linear` reproduces the old barrier-plus-scan artifact so
 *     its inflation can be measured, `tree` is the default and what a reported
 *     result should use.
 *
 *   - Two initialization arms, selected with --arm: A parks the matrix on
 *     thread 0 (naive master-thread init); B distributes it across the slices
 *     that will later compute on it (parallel first-touch). The same partition
 *     drives first-touch and compute, so placement matches computation.
 *
 *   - Per-kernel timers (SpMV / MGS / dense expm / other), so one run yields
 *     SpMV's plateau and MGS's roll-off on the same axis.
 *
 *   - Median-of-repeats timing with an inter-quartile-range error bar: timing
 *     noise is right-skewed on a boost-enabled chip, and the median reports
 *     throughput that can actually be sustained.
 *
 * Thread count P is taken from the OpenMP runtime (OMP_NUM_THREADS), placement
 * and binding come from OMP_PLACES=cores and OMP_PROC_BIND=close, set by the
 * sweep scripts. One invocation measures one (arm, n, P) point and appends one
 * CSV row.
 *
 * Usage:
 *   OMP_NUM_THREADS=P OMP_PLACES=cores OMP_PROC_BIND=close \
 *     ./scaling --arm A|B --n N [--steps S] [--m M] [--repeats R]
 *               [--option basket|rainbow] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-07-06
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

#include "config.hpp"
#include "pde_operators.hpp"
#include "reduction.hpp"

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static inline double elapsed_s(Clock::time_point t0)
{
    return Sec(Clock::now() - t0).count();
}

// Row partition
//
// Split the rows [0, aug_N) into P contiguous blocks whose interior seams are
// aligned to the cache line (8 doubles = 64 B). The same partition drives the
// first-touch loop and every compute loop (SpMV write, GS daxpy/scale, dot
// reduction), so thread t owns the identical index range everywhere, which is
// what makes first-touch meaningful and the measurement repeatable.
struct Partition {
    int                  P;
    std::vector<int64_t> start;  // size P+1; start[0]=0, start[P]=aug_N
};

// SpMV compute schedule: the experimental axis that tests why the strong-scaling
// transition happens. The transition is hypothesized to be a per-CCX L3-capacity
// effect: once each CCX's share of the matrix (3 cores x ~ws/P) fits in its own
// 16 MiB slice and stays resident across the ei_steps * m SpMVs, reads come from L3
// instead of DRAM. The three schedules break that hypothesis in different ways:
//
//   BLOCK    contiguous band per core, fixed across all SpMVs. Each CCX re-reads
//            the same ~3*ws/P MiB every SpMV, so it becomes L3-resident at P>=12
//            (the locality-preserving control).
//
//   CYCLIC   cache-line blocks dealt round-robin, fixed across SpMVs. Rows are
//            scattered across the whole matrix, but each CCX's volume is still
//            3*ws/P MiB and sets the same L3 residency, so this is predicted to
//            still transition; it isolates contiguity at fixed volume, a control
//            that shows scatter alone changes nothing.
//
//   ROTATE   contiguous bands, but which band a core computes rotates by one
//            every SpMV. Over the ei_steps * m reuse window each CCX sweeps every
//            band, so its 16 MiB slice can never retain a stable cache-fitting
//            subset: the per-CCX footprint becomes the whole matrix and every
//            SpMV pays DRAM. If the per-CCX-capacity mechanism is correct, the
//            transition must degrade here.
enum class Scheme { BLOCK, CYCLIC, ROTATE };

static Scheme parse_scheme(std::string_view s)
{
    if (s == "block")  return Scheme::BLOCK;
    if (s == "cyclic") return Scheme::CYCLIC;
    if (s == "rotate") return Scheme::ROTATE;
    throw std::invalid_argument("Unknown --sched: " + std::string(s)
                                + " (block|cyclic|rotate)");
}

static const char* scheme_name(Scheme s)
{
    switch (s) {
        case Scheme::BLOCK:  return "block";
        case Scheme::CYCLIC: return "cyclic";
        case Scheme::ROTATE: return "rotate";
    }
    return "?";
}

static Partition make_partition(int64_t aug_N, int P)
{
    constexpr int64_t line = 8;  // doubles per 64 B cache line
    const int64_t nblk = (aug_N + line - 1) / line;  // # cache-line blocks

    Partition part;
    part.P = P;
    part.start.resize(static_cast<std::size_t>(P) + 1);
    for (int t = 0; t <= P; ++t) {
        // Distribute cache-line blocks evenly, then scale back to rows.  Interior
        // seams land on multiples of 8 rows; the final seam is exactly aug_N.
        const int64_t blk = (static_cast<int64_t>(t) * nblk) / P;
        part.start[static_cast<std::size_t>(t)] = std::min(aug_N, blk * line);
    }
    part.start[static_cast<std::size_t>(P)] = aug_N;
    return part;
}

// CSR matrix with placement under our control
//
// Eigen stores SparseMatrix<double> column-major (CSC).
// Row-partitioning the output of y = Ax demands row-contiguous storage,
// so we hold our own CSR arrays and decide per arm which thread first-touches which rows.
struct CsrMatrix {
    int64_t              n_rows = 0;
    int64_t              nnz    = 0;
    std::vector<int64_t> row_ptr;   // size n_rows + 1
    std::vector<int32_t> col_idx;   // size nnz  (int32: matches the memory model)
    std::vector<double>  vals;      // size nnz
};

// Reference CSR built on the master thread from an Eigen operator.
// This is the "source of truth". The working CsrMatrix is a placement-controlled copy of it.
static CsrMatrix csr_from_eigen(const SpMat& A)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> Ar = A;
    Ar.makeCompressed();

    CsrMatrix c;
    c.n_rows = Ar.rows();
    c.nnz    = Ar.nonZeros();
    c.row_ptr.resize(static_cast<std::size_t>(c.n_rows) + 1);
    c.col_idx.resize(static_cast<std::size_t>(c.nnz));
    c.vals.resize(static_cast<std::size_t>(c.nnz));

    const int* outer = Ar.outerIndexPtr();  // size n_rows + 1
    const int* inner = Ar.innerIndexPtr();  // size nnz
    const double* v  = Ar.valuePtr();       // size nnz
    for (int64_t r = 0; r <= c.n_rows; ++r)
        c.row_ptr[static_cast<std::size_t>(r)] = outer[r];
    for (int64_t k = 0; k < c.nnz; ++k) {
        c.col_idx[static_cast<std::size_t>(k)] = inner[k];
        c.vals[static_cast<std::size_t>(k)]    = v[k];
    }
    return c;
}

// Placement-controlled working state.
//
// All large buffers (the CSR arrays, the Arnoldi basis V, the work vector w)
// live here as flat std::vectors we first-touch ourselves. V is column-major
// (aug_N x (m+1)) so V.col(j) is a contiguous slice a row-block can stream.
struct WorkState {
    int64_t aug_N = 0;
    int     m     = 0;

    CsrMatrix           A;        // working matrix (placement per arm)
    std::vector<double> V;        // aug_N * (m+1), column-major
    std::vector<double> w;        // aug_N: SpMV output / GS work vector
};

// Arm A: naive master-thread init. malloc + fill from thread 0. First-touch
// parks the entire matrix (and V, w) on thread 0's slice(s).
static void place_naive(WorkState& st, const CsrMatrix& ref)
{
    st.A = ref;  // full copy on the master thread
    std::fill(st.V.begin(), st.V.end(), 0.0);
    std::fill(st.w.begin(), st.w.end(), 0.0);
}

// Arm B: parallel first-touch. Threads are already bound (OMP_PROC_BIND=close).
// Each thread first-touches exactly the row partition it will later compute on,
// so the matrix distributes across the slices doing the work.
static void place_first_touch(WorkState& st, const CsrMatrix& ref, const Partition& part)
{
    st.A.n_rows = ref.n_rows;
    st.A.nnz    = ref.nnz;
    st.A.row_ptr.resize(static_cast<std::size_t>(ref.n_rows) + 1);
    st.A.col_idx.resize(static_cast<std::size_t>(ref.nnz));
    st.A.vals.resize(static_cast<std::size_t>(ref.nnz));

    const int64_t m1 = st.m + 1;

    #pragma omp parallel num_threads(part.P)
    {
        const int t   = omp_get_thread_num();
        const int64_t r0 = part.start[static_cast<std::size_t>(t)];
        const int64_t r1 = part.start[static_cast<std::size_t>(t) + 1];

        // Row pointers and the nonzeros they bound: first write == first touch.
        for (int64_t r = r0; r < r1; ++r)
            st.A.row_ptr[static_cast<std::size_t>(r)] = ref.row_ptr[static_cast<std::size_t>(r)];
        if (t == part.P - 1)
            st.A.row_ptr[static_cast<std::size_t>(ref.n_rows)] =
                ref.row_ptr[static_cast<std::size_t>(ref.n_rows)];

        const int64_t k0 = ref.row_ptr[static_cast<std::size_t>(r0)];
        const int64_t k1 = ref.row_ptr[static_cast<std::size_t>(r1)];
        for (int64_t k = k0; k < k1; ++k) {
            st.A.col_idx[static_cast<std::size_t>(k)] = ref.col_idx[static_cast<std::size_t>(k)];
            st.A.vals[static_cast<std::size_t>(k)]    = ref.vals[static_cast<std::size_t>(k)];
        }

        // The basis columns and work vector, touched on the same row slice.
        for (int64_t col = 0; col < m1; ++col)
            for (int64_t r = r0; r < r1; ++r)
                st.V[static_cast<std::size_t>(col * st.aug_N + r)] = 0.0;
        for (int64_t r = r0; r < r1; ++r)
            st.w[static_cast<std::size_t>(r)] = 0.0;
    }
}

// Parallel primitives (all driven by the same row partition)
// One CSR row: y[r] = sum_k A(r,k) x[k]. Shared by every schedule so they differ
// only in which rows a thread visits, never in the arithmetic or the byte model.
static inline void spmv_row(const CsrMatrix& A, const double* x, double* y, int64_t r)
{
    double acc = 0.0;
    const int64_t k0 = A.row_ptr[static_cast<std::size_t>(r)];
    const int64_t k1 = A.row_ptr[static_cast<std::size_t>(r) + 1];
    for (int64_t k = k0; k < k1; ++k)
        acc += A.vals[static_cast<std::size_t>(k)]
             * x[A.col_idx[static_cast<std::size_t>(k)]];
    y[r] = acc;
}

// SPMD SpMV: the rows this thread computes for one SpMV, under the chosen
// schedule. Called from inside a persistent parallel region (it opens no region
// of its own), so the only synchronization charged to the kernel is genuine
// barrier latency. spmv_row does the arithmetic; the schedule selects only which
// rows, so byte model and flop count are identical across schedules. `offset`
// (the running SpMV index) is used only by ROTATE.
static inline void spmv_thread_rows(const CsrMatrix& A, const double* x, double* y,
                                    const Partition& part, int64_t aug_N,
                                    Scheme sched, int t, int offset)
{
    switch (sched) {
        case Scheme::BLOCK: {
            // Contiguous band per thread, fixed across SpMVs. Sole writer of its
            // y-slice. Band seams cache-line aligned (no false sharing).
            const int64_t r0 = part.start[static_cast<std::size_t>(t)];
            const int64_t r1 = part.start[static_cast<std::size_t>(t) + 1];
            for (int64_t r = r0; r < r1; ++r) spmv_row(A, x, y, r);
            break;
        }
        case Scheme::CYCLIC: {
            // Cache-line blocks (8 rows) dealt round-robin: scatters each thread's
            // rows across the whole matrix while keeping each output line wholly
            // owned by one thread (no false sharing). Per-thread volume is
            // unchanged vs BLOCK, so this does NOT reduce per-CCX footprint.
            constexpr int64_t C = 8;
            const int64_t nblk = (aug_N + C - 1) / C;
            for (int64_t b = t; b < nblk; b += part.P) {
                const int64_t lo = b * C;
                const int64_t hi = std::min(aug_N, lo + C);
                for (int64_t r = lo; r < hi; ++r) spmv_row(A, x, y, r);
            }
            break;
        }
        case Scheme::ROTATE: {
            // Contiguous bands, but thread t computes band (t + offset) mod P.
            // offset advances every SpMV, so over the reuse window each core (hence
            // each CCX) sweeps every band: the per-CCX resident set never
            // stabilizes below capacity and every SpMV re-reads from DRAM.
            const int band = (t + offset) % part.P;
            const int64_t r0 = part.start[static_cast<std::size_t>(band)];
            const int64_t r1 = part.start[static_cast<std::size_t>(band) + 1];
            for (int64_t r = r0; r < r1; ++r) spmv_row(A, x, y, r);
            break;
        }
    }
}

// A thread's partial sum of a dot product over its block-partition rows. The
// cross-thread combine happens in the caller after a barrier (see run_solve):
// that barrier + combine is the global reduction whose dependency chain limits
// MGS scaling. `#pragma omp simd` lets each partial vectorize so a single thread
// is not capped below L3 bandwidth by a scalar reduction.
static inline double partial_dot(const double* a, const double* b,
                                 int64_t r0, int64_t r1)
{
    double s = 0.0;
    #pragma omp simd reduction(+ : s)
    for (int64_t r = r0; r < r1; ++r) s += a[r] * b[r];
    return s;
}

// One instrumented solve
//
// Runs the full ei_steps time-stepping loop with a fixed Krylov dimension m and
// the convergence check disabled. Returns per-kernel wall times for this solve;
// the caller takes the median over repeats.
struct SolveTimes {
    double spmv  = 0.0;
    double gs    = 0.0;
    double expm  = 0.0;
    double other = 0.0;
    double total = 0.0;
    double price = 0.0;  // spot price, a correctness sanity value
};

static SolveTimes run_solve(WorkState& st, const PDESystem& sys, const Config& cfg,
                            const Partition& part, Scheme sched, ReduceKind reduce)
{
    const int     N       = sys.N;
    const int     p       = 3;
    const bool    basket  = sys.has_forcing;
    const int64_t aug_N   = st.aug_N;
    const int     m       = st.m;
    const int     P       = part.P;
    const double  dt      = cfg.t_final / cfg.ei_steps;

    double* V = st.V.data();
    double* w = st.w.data();

    // Constructed outside every parallel region and reused across time steps, so its
    // generation counters serialize successive reductions the way they were designed to.
    // Every thread must call reduce() the same number of times, in the same order: the
    // fixed-m loop below guarantees that, and a convergence exit would not.
    TeamReducer red(P, reduce);

    // Small dense objects (m is fixed and tiny -> permanently L1/L2-resident).
    // H and f are touched only by the master thread, fbuf broadcasts f to all.
    MatXd H = MatXd::Zero(m + 1, m);
    std::vector<double> u(static_cast<std::size_t>(N));
    std::vector<double> fbuf(static_cast<std::size_t>(m));

    SolveTimes ts;
    Clock::time_point last;   // master-only running timestamp for phase timing
    const auto wall0 = Clock::now();

    double t_curr = 0.0;
    for (int step = 0; step < cfg.ei_steps; ++step) {
        const double h    = std::min(dt, cfg.t_final - t_curr);
        const double tau0 = t_curr;
        const VecXd s_aug = basket ? make_s_vec(tau0) : VecXd();

        // One persistent parallel region per time step (was ~80 fork/joins/step of
        // per-primitive regions). Threads run SPMD over the fixed block partition;
        // the only synchronization charged to the kernels is genuine barrier and
        // reduction latency, exactly the MGS cost the study means to measure. Global
        // reductions go through `red` (TeamReducer, constructed above), so every
        // thread returns the same reduced scalar without a second broadcast.
        #pragma omp parallel num_threads(P)
        {
            const int t   = omp_get_thread_num();
            const int64_t r0 = part.start[static_cast<std::size_t>(t)];
            const int64_t r1 = part.start[static_cast<std::size_t>(t) + 1];

            #pragma omp masked
            { last = Clock::now(); }

            // Build + normalize the Arnoldi start vector V.col(0)
            for (int64_t r = r0; r < r1 && r < N; ++r)
                V[static_cast<std::size_t>(r)] =
                    (step == 0) ? sys.u0[static_cast<Eigen::Index>(r)]
                                : u[static_cast<std::size_t>(r)];
            #pragma omp masked
            {
                if (basket)
                    for (int i = 0; i < p; ++i)
                        V[static_cast<std::size_t>(N + i)] = s_aug[i];
                H.setZero();
            }
            #pragma omp barrier   // V.col(0) fully written before the norm reduction

            double beta = std::sqrt(red.reduce(t, partial_dot(V, V, r0, r1)));
            const double binv = (beta > 0.0) ? 1.0 / beta : 0.0;
            for (int64_t r = r0; r < r1; ++r) V[static_cast<std::size_t>(r)] *= binv;
            #pragma omp barrier   // V.col(0) normalized before first SpMV reads it
            #pragma omp masked
            {
                const auto now = Clock::now();
                ts.other += Sec(now - last).count();
                last = now;
            }

            // Fixed-dimension Arnoldi: exactly m iterations (no convergence exit)
            for (int j = 0; j < m; ++j) {
                const double* Vj = V + static_cast<std::size_t>(j) * aug_N;

                // SpMV: w = A * V.col(j)  (schedule chooses which rows this thread owns)
                spmv_thread_rows(st.A, Vj, w, part, aug_N, sched, t, step * m + j);
                #pragma omp barrier
                #pragma omp masked
                {
                    const auto now = Clock::now();
                    ts.spmv += Sec(now - last).count();
                    last = now;
                }

                // Modified Gram-Schmidt against V.col(0..j): each dot is a global
                // reduction (barrier + combine) (the synchronization-bound cost).
                for (int i = 0; i <= j; ++i) {
                    const double* Vi = V + static_cast<std::size_t>(i) * aug_N;
                    const double hij = red.reduce(t, partial_dot(w, Vi, r0, r1));
                    #pragma omp masked
                    { H(i, j) = hij; }
                    for (int64_t r = r0; r < r1; ++r)
                        w[static_cast<std::size_t>(r)] -= hij * Vi[static_cast<std::size_t>(r)];
                    #pragma omp barrier   // w updated before the next dot reads it
                }
                const double hnorm = std::sqrt(red.reduce(t, partial_dot(w, w, r0, r1)));
                #pragma omp masked
                { H(j + 1, j) = hnorm; }
                #pragma omp masked
                {
                    const auto now = Clock::now();
                    ts.gs += Sec(now - last).count();
                    last = now;
                }

                // Normalize the next basis vector. Proceed even on a
                // tiny hnorm: at fixed m this is a throughput study, not a
                // convergence run, so breakdown handling is irrelevant to timing.
                const double inv = (hnorm > 0.0) ? 1.0 / hnorm : 0.0;
                double* Vn = V + static_cast<std::size_t>(j + 1) * aug_N;
                for (int64_t r = r0; r < r1; ++r)
                    Vn[static_cast<std::size_t>(r)] = w[static_cast<std::size_t>(r)] * inv;
                #pragma omp barrier   // V.col(j+1) ready before it is used next iter
                #pragma omp masked
                {
                    const auto now = Clock::now();
                    ts.other += Sec(now - last).count();
                    last = now;
                }
            }

            // Dense expm on the small m x m Hessenberg block (master only)
            #pragma omp masked
            {
                const auto te = Clock::now();
                const MatXd H_m = H.topLeftCorner(m, m);
                const VecXd f   = (h * H_m).exp().col(0);
                for (int k = 0; k < m; ++k) fbuf[static_cast<std::size_t>(k)] = f[k];
                ts.expm += Sec(Clock::now() - te).count();
                last = Clock::now();
            }
            #pragma omp barrier   // fbuf visible before reconstruction reads it

            // u = beta * V[:, 0, ..., m-1] * f  (dense mat-vec over the row partition).
            // beta is each thread's identical local copy of the reduced norm.
            for (int64_t r = r0; r < r1 && r < N; ++r) {
                double acc = 0.0;
                for (int k = 0; k < m; ++k)
                    acc += V[static_cast<std::size_t>(k) * aug_N + r]
                         * fbuf[static_cast<std::size_t>(k)];
                u[static_cast<std::size_t>(r)] = beta * acc;
            }
            #pragma omp barrier
            #pragma omp masked
            {
                const auto now = Clock::now();
                ts.other += Sec(now - last).count();
                last = now;
            }
        }  // end persistent parallel region

        t_curr += h;
    }

    ts.total = elapsed_s(wall0);

    // Correctness sanity: option price at the spot node.
    VecXd u_vec = Eigen::Map<VecXd>(u.data(), N);
    ts.price = extract_price(u_vec, sys.grid, cfg.initial_prices);
    return ts;
}

// median / IQR over the timed repeats (warm-up already discarded by the caller)
struct Stat {
    double median = 0.0;
    double q1     = 0.0;  // inter-quartile-range lower edge (error bar)
    double q3     = 0.0;  // inter-quartile-range upper edge
};

static double quantile(std::vector<double> v, double q)  // v taken by value: sorted here
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return v[lo] + frac * (v[hi] - v[lo]);
}

static Stat summarize(const std::vector<double>& v)
{
    return { quantile(v, 0.5), quantile(v, 0.25), quantile(v, 0.75) };
}

// CLI
struct ScalingArgs {
    Config       pde;
    char         arm     = 'B';
    Scheme       sched   = Scheme::BLOCK;  // SpMV compute schedule (locality axis)
    ReduceKind   reduce  = ReduceKind::TREE;  // reduction primitive; see the file header
    int          m       = 8;
    int          repeats = 7;   // >= 5-7 timed iterations for a stable median (spec 10)
    std::string  csv_path;
};

static ReduceKind parse_reduce(std::string_view s)
{
    if (s == "tree")   return ReduceKind::TREE;
    if (s == "linear") return ReduceKind::LINEAR;
    throw std::invalid_argument("Unknown --reduce: " + std::string(s) + " (tree|linear)");
}

static ScalingArgs parse_scaling_args(std::span<const char* const> args)
{
    ScalingArgs a;
    a.pde.ei_steps = a.pde.temporal_steps;  // fixed step count. --tol is irrelevant here

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        auto next = [&]() -> std::string_view {
            if (++i >= args.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return args[i];
        };
        if      (arg == "--n")       a.pde.n = std::stoi(std::string(next()));
        else if (arg == "--steps") { a.pde.temporal_steps = std::stoi(std::string(next()));
                                     a.pde.ei_steps = a.pde.temporal_steps; }
        else if (arg == "--m")       a.m = std::stoi(std::string(next()));
        else if (arg == "--repeats") a.repeats = std::stoi(std::string(next()));
        else if (arg == "--csv")     a.csv_path = std::string(next());
        else if (arg == "--sched")   a.sched = parse_scheme(next());
        else if (arg == "--reduce")  a.reduce = parse_reduce(next());
        else if (arg == "--arm") {
            const auto v = next();
            if      (v == "A" || v == "a" || v == "naive")       a.arm = 'A';
            else if (v == "B" || v == "b" || v == "first-touch") a.arm = 'B';
            else throw std::invalid_argument("Unknown arm: " + std::string(v) + " (A|B)");
        }
        else if (arg == "--option") {
            const auto v = next();
            if      (v == "basket")  a.pde.option_type = EuropeanOptionType::CALL_BASKET;
            else if (v == "rainbow") a.pde.option_type = EuropeanOptionType::CALL_MIN_RAINBOW;
            else throw std::invalid_argument("Unknown option type: " + std::string(v));
        }
        else if (arg == "--help") {
            std::println("Usage: OMP_NUM_THREADS=P OMP_PLACES=cores OMP_PROC_BIND=close \\");
            std::println("         ./scaling --arm A|B --n N [--sched block|cyclic|rotate]");
            std::println("                   [--reduce tree|linear]");
            std::println("                   [--steps S] [--m M] [--repeats R]");
            std::println("                   [--option basket|rainbow] [--csv PATH]");
            std::println("");
            std::println("  --arm A|B      init strategy: A = naive master-thread,");
            std::println("                 B = parallel first-touch (spec section 7)");
            std::println("  --sched SCHED  SpMV compute schedule (locality axis):");
            std::println("                 block  = contiguous band per core (locality-preserving)");
            std::println("                 cyclic = round-robin cache-line blocks (scatter, same volume)");
            std::println("                 rotate = per-SpMV band rotation (locality-destroying)");
            std::println("  --reduce KIND  reduction primitive for MGS's global reductions:");
            std::println("                 tree   = binary tree, point-to-point (default; the");
            std::println("                          primitive calibrate_alpha.sh calibrates)");
            std::println("                 linear = barrier + O(P) redundant scan; the historical");
            std::println("                          artifact, kept only to size its inflation of the");
            std::println("                          GS curve. Do not report a linear result.");
            std::println("  --n N          grid points per dimension (default 15)");
            std::println("  --steps S      KSM-EI time steps (default 100)");
            std::println("  --m M          FIXED Krylov dimension (default 8, spec section 4)");
            std::println("  --repeats R    timed repeats after 1 warm-up (default 7, spec 10)");
            std::println("  --option TYPE  basket (default) or rainbow");
            std::println("  --csv PATH     append one result row (header written if new)");
            std::println("");
            std::println("  Thread count P is read from the OpenMP runtime (OMP_NUM_THREADS).");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (a.m < 1) throw std::invalid_argument("--m must be >= 1");
    return a;
}

// CSV row (one per invocation, header written if the file does not yet exist)
static const char* kCsvHeader =
    "arm,sched,reduce,option,n,N,aug_N,nnz,P,ccx_engaged,ei_steps,m,repeats,"
    "spmv_ms,spmv_q1,spmv_q3,gs_ms,gs_q1,gs_q3,expm_ms,expm_q1,expm_q3,"
    "other_ms,other_q1,other_q3,total_ms,total_q1,total_q3,"
    "spmv_ws_mib,gs_ws_mib,ccx_demand_spmv,ccx_demand_gs,places,bind,price\n";

int main(int argc, char* argv[])
{
    try {
        const ScalingArgs a = parse_scaling_args(
            std::span<const char* const>(argv, static_cast<std::size_t>(argc)));

        const int  P       = omp_get_max_threads();
        const bool rainbow = (a.pde.option_type == EuropeanOptionType::CALL_MIN_RAINBOW);
        const char* places = std::getenv("OMP_PLACES");
        const char* bind   = std::getenv("OMP_PROC_BIND");

        std::println("KSM-EI OpenMP scaling harness");
        std::println("  arm={}  option={}  n={}  steps={}  m={} (fixed)  P={}  repeats={}",
                     a.arm, rainbow ? "rainbow" : "basket",
                     a.pde.n, a.pde.ei_steps, a.m, P, a.repeats);
        std::println("  OMP_PLACES={}  OMP_PROC_BIND={}",
                     places ? places : "(unset)", bind ? bind : "(unset)");

        // Build the PDE system and the (augmented) operator once.
        std::println("  [Building PDE system...]");
        const PDESystem sys = build_pde_system(
            a.pde.n, a.pde.strike_price, a.pde.risk_free_rate, a.pde.t_final,
            a.pde.sigma, a.pde.rho_off, a.pde.weight, a.pde.initial_prices,
            a.pde.alpha, rainbow);

        const int     N     = sys.N;
        const int     p     = 3;
        const int64_t aug_N = sys.has_forcing ? int64_t(N) + p : int64_t(N);
        const SpMat   A_op  = sys.has_forcing ? build_A_tilde(sys.A, sys.B, N) : sys.A;

        const CsrMatrix ref = csr_from_eigen(A_op);
        const int64_t   nnz = ref.nnz;

        // Working-set sizes and CCX demand (spec 5/6), constants under strong scaling.
        const double spmv_ws = double(nnz) * 12.0 + double(aug_N + 1) * 4.0 + double(aug_N) * 16.0;
        const double gs_ws   = double(a.m) * double(aug_N) * 8.0;
        const double MiB     = 1024.0 * 1024.0;
        const int ccx_engaged   = (P + 2) / 3;                              // ceil(P/3)
        const int ccx_dem_spmv  = static_cast<int>(std::ceil(spmv_ws / (16.0 * MiB)));
        const int ccx_dem_gs    = static_cast<int>(std::ceil(gs_ws   / (16.0 * MiB)));

        const Partition part = make_partition(aug_N, P);

        WorkState st;
        st.aug_N = aug_N;
        st.m     = a.m;
        st.V.resize(static_cast<std::size_t>(aug_N) * (a.m + 1));
        st.w.resize(static_cast<std::size_t>(aug_N));

        // Place the data per arm. This is the controlled variable of the A/B
        // experiment: naive parks everything on thread 0; first-touch
        // distributes across the slices that will compute on it.
        std::println("  [Placing data: arm {}]", a.arm);
        if (a.arm == 'A') place_naive(st, ref);
        else              place_first_touch(st, ref, part);

        // Warm-up (discarded): first-touch page faults + boost ramp (spec 10).
        std::println("  [Warm-up + {} timed repeats...  sched={}  reduce={}]",
                     a.repeats, scheme_name(a.sched), reduce_kind_name(a.reduce));
        (void)run_solve(st, sys, a.pde, part, a.sched, a.reduce);

        std::vector<double> spmv, gs, expm, other, total;
        double price = 0.0;
        for (int r = 0; r < a.repeats; ++r) {
            const SolveTimes tsr = run_solve(st, sys, a.pde, part, a.sched, a.reduce);
            spmv.push_back(tsr.spmv * 1e3);
            gs.push_back(tsr.gs * 1e3);
            expm.push_back(tsr.expm * 1e3);
            other.push_back(tsr.other * 1e3);
            total.push_back(tsr.total * 1e3);
            price = tsr.price;
        }

        const Stat s_spmv = summarize(spmv);
        const Stat s_gs   = summarize(gs);
        const Stat s_expm = summarize(expm);
        const Stat s_oth  = summarize(other);
        const Stat s_tot  = summarize(total);

        // stdout report
        std::println("");
        std::println("=== Scaling point  (arm {}, sched {}, reduce {}, n={}, P={}) ===",
                     a.arm, scheme_name(a.sched), reduce_kind_name(a.reduce), a.pde.n, P);
        std::println("||  N={}  aug_N={}  nnz={}  CCX engaged={}", N, aug_N, nnz, ccx_engaged);
        std::println("||  SpMV ws={:.1f} MiB (demands {} CCX)   GS ws={:.1f} MiB (demands {} CCX)",
                     spmv_ws / MiB, ccx_dem_spmv, gs_ws / MiB, ccx_dem_gs);
        std::println("||  {:<14} {:>10}  {:>10}  {:>10}", "Kernel", "median(ms)", "q1(ms)", "q3(ms)");
        std::println("||  {}", std::string(48, '-'));
        auto row = [](std::string_view name, const Stat& s) {
            std::println("||  {:<14} {:>10.3f}  {:>10.3f}  {:>10.3f}", name, s.median, s.q1, s.q3);
        };
        row("SpMV",         s_spmv);
        row("Gram-Schmidt", s_gs);
        row("dense expm",   s_expm);
        row("other",        s_oth);
        std::println("||  {}", std::string(48, '-'));
        row("total",        s_tot);
        std::println("||  price @ spot = {:.6f}", price);
        std::println("===");

        // CSV append
        if (!a.csv_path.empty()) {
            const bool exists = std::filesystem::exists(a.csv_path)
                             && std::filesystem::file_size(a.csv_path) > 0;
            std::ofstream f(a.csv_path, std::ios::app);
            if (!f) throw std::runtime_error("Cannot open CSV: " + a.csv_path);
            if (!exists) f << kCsvHeader;
            f << a.arm << ',' << scheme_name(a.sched) << ','
              << reduce_kind_name(a.reduce) << ','
              << (rainbow ? "rainbow" : "basket") << ','
              << a.pde.n << ',' << N << ',' << aug_N << ',' << nnz << ','
              << P << ',' << ccx_engaged << ',' << a.pde.ei_steps << ',' << a.m << ','
              << a.repeats << ','
              << s_spmv.median << ',' << s_spmv.q1 << ',' << s_spmv.q3 << ','
              << s_gs.median   << ',' << s_gs.q1   << ',' << s_gs.q3   << ','
              << s_expm.median << ',' << s_expm.q1 << ',' << s_expm.q3 << ','
              << s_oth.median  << ',' << s_oth.q1  << ',' << s_oth.q3  << ','
              << s_tot.median  << ',' << s_tot.q1  << ',' << s_tot.q3  << ','
              << (spmv_ws / MiB) << ',' << (gs_ws / MiB) << ','
              << ccx_dem_spmv << ',' << ccx_dem_gs << ','
              << (places ? places : "") << ',' << (bind ? bind : "") << ','
              << price << '\n';
            std::println("  [Appended row to {}]", a.csv_path);
        }

    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
