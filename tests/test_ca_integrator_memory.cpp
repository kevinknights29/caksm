/**
 * @file test_ca_integrator_memory.cpp
 * @brief Allocation-model checks for the largest-common-grid experiment.
 */

#include <catch2/catch_test_macros.hpp>

#include "ca_integrator_memory.hpp"

using namespace ca_integrator_memory;

/// The one-GPU model must count every column at the augmented leading
/// dimension, and charge the Basket face buffer only under Basket.
///
/// Expected: vector bytes are exactly (m+s+6) columns of n^3+3 doubles; the
/// nine face planes appear for Basket and are absent for Rainbow; and those
/// faces are the only difference between the two options.
TEST_CASE("one-GPU production memory counts every leading-dimension vector",
          "[ca][memory]")
{
    constexpr int n = 31;
    constexpr int m = 24;
    constexpr int s = 4;
    const Estimate basket = one_gpu(n, m, s, true, 8, 64);
    const Estimate rainbow = one_gpu(n, m, s, false, 8, 64);
    const std::uint64_t ld =
        static_cast<std::uint64_t>(n) * n * n + 3;

    REQUIRE(basket.vector_bytes
            == ld * static_cast<std::uint64_t>(m + s + 6)
                * sizeof(double));
    REQUIRE(basket.face_bytes
            == 9ULL * n * n * sizeof(double));
    REQUIRE(rainbow.face_bytes == 0);
    REQUIRE(basket.total_bytes - rainbow.total_bytes
            == basket.face_bytes);
}

/// The per-slab model must charge each rank for its own z range, its deep halo,
/// and the integer buffers the agreement instrument adds.
///
/// Expected: the halo is the owned planes flanked by s on each side, so it
/// differs between ranks whenever the truncating split gives them different
/// depths; the last rank is never smaller than the first; the small-buffer
/// total matches the term-by-term count exactly; and dropping the candidate,
/// certificate and agreement arguments removes precisely those bytes and
/// nothing else.
TEST_CASE("distributed slab model includes deep halos and replicated faces",
          "[ca][memory]")
{
    constexpr int n = 97;
    constexpr int m = 24;
    constexpr int s = 4;
    constexpr std::size_t candidate_bytes = 64;
    constexpr std::size_t certificate_bytes = 16;
    constexpr int decision_fields = 4;
    const Estimate rank0 =
        distributed_slab(
            n, m, s, true, 4, 0, 8, candidate_bytes,
            certificate_bytes, decision_fields);
    const Estimate rank3 =
        distributed_slab(
            n, m, s, true, 4, 3, 8, candidate_bytes,
            certificate_bytes, decision_fields);

    const int z0 = n / 4;
    const int z3 = n - 3 * n / 4;
    REQUIRE(rank0.halo_bytes
            == static_cast<std::uint64_t>(z0 + 2 * s) * n * n
                * sizeof(double));
    REQUIRE(rank3.halo_bytes
            == static_cast<std::uint64_t>(z3 + 2 * s) * n * n
                * sizeof(double));
    REQUIRE(rank3.total_bytes >= rank0.total_bytes);
    const std::uint64_t expected_auxiliary =
        candidate_bytes + certificate_bytes
        + static_cast<std::uint64_t>(2 + 3 * decision_fields) * sizeof(int);
    const std::uint64_t target = m + 1;
    const std::uint64_t block = s;
    const std::uint64_t expected_small_doubles =
        target * (target - 1)
        + 2 * target * block
        + 3 * block * block
        + 2 + target + 8;
    REQUIRE(rank0.small_bytes
            == expected_small_doubles * sizeof(double)
                + expected_auxiliary);
    const Estimate without_auxiliary =
        distributed_slab(n, m, s, true, 4, 0, 8, 0, 0, 0);
    REQUIRE(rank0.small_bytes - without_auxiliary.small_bytes
            == candidate_bytes + certificate_bytes
                + 3ULL * decision_fields * sizeof(int));

    const Estimate without_agreement =
        distributed_slab(
            n, m, s, true, 4, 0, 8, candidate_bytes,
            certificate_bytes, 0);
    REQUIRE(rank0.small_bytes - without_agreement.small_bytes
            == 3ULL * decision_fields * sizeof(int));
}

/// The grid search must be monotone in n and must return the largest grid that
/// still fits, not the first that does.
///
/// Expected: footprint grows with n, so a capacity sized exactly to n=101
/// selects 101 rather than 103. The reserve is checked separately: it holds
/// back its fraction of total memory, never reports more than is free, and an
/// exact 10% reserve on 1000 bytes yields 900 rather than 899, which is the
/// floating-point edge the implementation nudges past.
TEST_CASE("largest odd grid obeys the reserve-limited capacity",
          "[ca][memory]")
{
    const Estimate at_101 = one_gpu(101, 24, 4, true, 0, 0);
    const Estimate at_103 = one_gpu(103, 24, 4, true, 0, 0);
    REQUIRE(at_103.total_bytes > at_101.total_bytes);
    REQUIRE(largest_odd_grid(
                at_101.total_bytes,
                [](int n) { return one_gpu(n, 24, 4, true, 0, 0); })
            == 101);
    REQUIRE(usable_bytes(950, 1000, 0.10) == 900);
    REQUIRE(usable_bytes(850, 1000, 0.10) == 850);
    REQUIRE(usable_bytes(1000, 1000, 0.0) == 1000);
}
