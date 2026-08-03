/**
 * @file ca_integrator_memory.hpp
 * @brief Allocation-level memory model for the production CA integrators.
 *
 * The model counts every explicit device allocation made by the one-GPU
 * CholQR2 integrator and by one slab of the distributed integrator, a slab
 * being the contiguous range of z-planes of the n-cubed grid that one GPU owns.
 *
 * It counts only what this source allocates. The CUDA context, cuBLAS and
 * cuSOLVER workspaces, and the NCCL buffers are left to usable_bytes below,
 * which holds back a fraction of the device against them; nothing here is
 * derived from a profiler.
 *
 * The basis buffer keeps its s+1 columns. The exact-depth arm of --arm consumes
 * only s of them, but both it and the as-measured arm are selected at run time
 * inside one binary, and the recorded largest-common-grid prediction was taken
 * against this allocation. Shrinking the buffer would silently move that
 * prediction, so the column is retained and the arm difference stays in the
 * recurrence, not the footprint.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ca_integrator_memory {

struct Estimate {
    std::uint64_t vector_bytes = 0;
    std::uint64_t halo_bytes = 0;
    std::uint64_t face_bytes = 0;
    std::uint64_t small_bytes = 0;
    std::uint64_t total_bytes = 0;
};

[[nodiscard]] inline std::uint64_t saturated_add(
    std::uint64_t lhs, std::uint64_t rhs)
{
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    return rhs > limit - lhs ? limit : lhs + rhs;
}

[[nodiscard]] inline std::uint64_t saturated_multiply(
    std::uint64_t lhs, std::uint64_t rhs)
{
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    if (lhs == 0 || rhs == 0) return 0;
    return lhs > limit / rhs ? limit : lhs * rhs;
}

[[nodiscard]] inline std::uint64_t cube(int n)
{
    const std::uint64_t value = static_cast<std::uint64_t>(n);
    return saturated_multiply(saturated_multiply(value, value), value);
}

[[nodiscard]] inline std::uint64_t square(int n)
{
    const std::uint64_t value = static_cast<std::uint64_t>(n);
    return saturated_multiply(value, value);
}

[[nodiscard]] inline std::uint64_t matrix_bytes(
    std::uint64_t rows, std::uint64_t columns)
{
    return saturated_multiply(
        saturated_multiply(rows, columns), sizeof(double));
}

/**
 * Bytes the one-GPU integrator allocates at a given (n, m, s).
 *
 * Three groups, split because they scale differently. The vector group is m+s+6 columns of the
 * augmented length n^3+3, so it grows as n^3 and is what sets the grid ceiling. The face group
 * is the Basket boundary forcing at 9 planes of n^2, absent under Rainbow. The small group is
 * everything whose size follows m and s rather than n: the Hessenberg, the block projection and
 * its reorthogonalization copy, the Gram and the two triangular factors, the certificate
 * scalars, the exponential coefficients, and the cuSOLVER potrf workspace whose size only the
 * library can report, hence potrf_lwork as an argument.
 *
 * Every arithmetic step saturates rather than wraps, so an n far past the device is reported as
 * a refusal to fit instead of a small number that appears to.
 */
[[nodiscard]] inline Estimate one_gpu(
    int n, int m, int s, bool basket, int potrf_lwork,
    std::size_t candidate_bytes)
{
    Estimate estimate;
    const std::uint64_t ld = saturated_add(cube(n), 3);
    const std::uint64_t target = static_cast<std::uint64_t>(m + 1);
    const std::uint64_t block = static_cast<std::uint64_t>(s);

    // start, scratch, B(s+1), V(m+1), action, and the timed state.
    estimate.vector_bytes = matrix_bytes(
        ld, static_cast<std::uint64_t>(m + s + 6));
    if (basket)
        estimate.face_bytes = matrix_bytes(square(n), 9);

    std::uint64_t small_doubles = 0;
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(target, target - 1)); // H
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(2 * target, block));  // C, C2
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(3 * block, block));   // G, R1, local_R
    small_doubles = saturated_add(small_doubles, 1);             // kappa
    small_doubles = saturated_add(
        small_doubles, static_cast<std::uint64_t>(m));           // f
    small_doubles = saturated_add(
        small_doubles,
        static_cast<std::uint64_t>(std::max(potrf_lwork, 0)));
    estimate.small_bytes = saturated_add(
        saturated_multiply(small_doubles, sizeof(double)),
        saturated_add(candidate_bytes, 3 * sizeof(int)));

    estimate.total_bytes = saturated_add(
        saturated_add(estimate.vector_bytes, estimate.face_bytes),
        estimate.small_bytes);
    return estimate;
}

/**
 * Bytes one slab of the distributed integrator allocates, for the slab that global_rank owns.
 *
 * The same three groups as one_gpu over the local z range only, plus two costs the one-GPU path
 * does not pay. The halo buffer holds the owned planes flanked by s planes on each side, which
 * is the depth one matrix-powers block reaches and therefore what a single exchange has to
 * carry. The integer group holds the certificate and candidate verdicts and three copies of the
 * agreement tuple, the minimum, the maximum, and the latched comparison, one set of
 * decision_fields each.
 *
 * The split is deliberately uneven: z_end - z_begin uses truncating division, so ranks differ by
 * at most one plane and the estimate is evaluated per rank rather than assumed uniform.
 */
[[nodiscard]] inline Estimate distributed_slab(
    int n, int m, int s, bool basket, int world_gpus, int global_rank,
    int potrf_lwork, std::size_t candidate_bytes,
    std::size_t certificate_bytes, int decision_fields)
{
    Estimate estimate;
    const int z_begin = global_rank * n / world_gpus;
    const int z_end = (global_rank + 1) * n / world_gpus;
    const std::uint64_t n2 = square(n);
    const std::uint64_t local_n =
        saturated_multiply(
            static_cast<std::uint64_t>(z_end - z_begin), n2);
    const std::uint64_t ld = saturated_add(local_n, 3);
    const std::uint64_t target = static_cast<std::uint64_t>(m + 1);
    const std::uint64_t block = static_cast<std::uint64_t>(s);

    // state, start, action, B(s+1), and V(m+1).
    estimate.vector_bytes = matrix_bytes(
        ld, static_cast<std::uint64_t>(m + s + 5));
    estimate.halo_bytes = matrix_bytes(
        saturated_multiply(
            static_cast<std::uint64_t>(z_end - z_begin + 2 * s), n2),
        1);
    if (basket)
        estimate.face_bytes = matrix_bytes(n2, 9);

    std::uint64_t small_doubles = 0;
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(target, target - 1)); // H
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(2 * target, block));  // C, C2
    small_doubles = saturated_add(
        small_doubles, saturated_multiply(3 * block, block));   // G, R1, local_R
    small_doubles = saturated_add(small_doubles, 2);             // kappa, scalar
    small_doubles = saturated_add(
        small_doubles, target);                                 // f
    small_doubles = saturated_add(
        small_doubles,
        static_cast<std::uint64_t>(std::max(potrf_lwork, 0)));
    // Candidate result; two potrf status words; one deferred-certificate
    // verdict; and the min, max, and latched agreement tuples.
    const std::uint64_t integer_bytes = saturated_multiply(
        saturated_add(
            2, saturated_multiply(
                   3, static_cast<std::uint64_t>(
                          std::max(decision_fields, 0)))),
        sizeof(int));
    const std::uint64_t auxiliary_bytes = saturated_add(
        saturated_add(candidate_bytes, certificate_bytes), integer_bytes);
    estimate.small_bytes = saturated_add(
        saturated_multiply(small_doubles, sizeof(double)),
        auxiliary_bytes);

    estimate.total_bytes = saturated_add(
        saturated_add(estimate.vector_bytes, estimate.halo_bytes),
        saturated_add(estimate.face_bytes, estimate.small_bytes));
    return estimate;
}

/**
 * What an estimate is allowed to fill, from a cudaMemGetInfo pair and a reserve fraction.
 *
 * Two ceilings, and the lower one wins. Free bytes are what the device has right now; the
 * reserve-limited figure is total bytes less the fraction held back for the allocations this
 * model does not count. Bounding by free alone would let a prediction depend on whatever else
 * happened to be resident, and bounding by the reserve alone would let it ignore a device that
 * is already occupied.
 */
[[nodiscard]] inline std::uint64_t usable_bytes(
    std::uint64_t free_bytes, std::uint64_t total_bytes, double reserve_fraction)
{
    const double bounded = std::clamp(reserve_fraction, 0.0, 1.0);
    // Decimal command-line fractions such as 0.10 are commonly represented a
    // few ulps above their mathematical value. The retained fraction is
    // therefore nudged upward by one double ulp before the integer floor, so an
    // exact 10% reserve on 1000 bytes yields 900 rather than 899.
    const double retained =
        std::nextafter(1.0 - bounded, 1.0);
    const auto reserve_limited = static_cast<std::uint64_t>(
        static_cast<long double>(total_bytes)
        * static_cast<long double>(retained));
    return std::min(free_bytes, reserve_limited);
}

/**
 * Largest odd grid whose estimate still fits in capacity_bytes.
 *
 * The largest-common-grid decision needs one n that every topology, option and width can
 * allocate, and the answer has to be known before anything is allocated: at these sizes a
 * failed attempt is a job that dies minutes in. Footprint is monotone in n, so a bisection over
 * the odd grids finds it in a few evaluations, and the caller supplies the estimator so the same
 * search serves the one-GPU and per-slab models. Odd only, because the grid keeps a node exactly
 * at the spot where the price is read.
 */
template <class Estimator>
[[nodiscard]] int largest_odd_grid(
    std::uint64_t capacity_bytes, Estimator&& estimate)
{
    // PDESystem::N is currently an int. 1290^3 is the largest admissible
    // cube below INT_MAX, so this is also a structural upper bound.
    constexpr int min_grid = 3;
    constexpr int max_grid = 1289;
    int low = 0;
    int high = (max_grid - min_grid) / 2;
    int accepted = 0;
    while (low <= high) {
        const int middle = low + (high - low) / 2;
        const int candidate = min_grid + 2 * middle;
        if (estimate(candidate).total_bytes <= capacity_bytes) {
            accepted = candidate;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    return accepted;
}

} // namespace ca_integrator_memory
