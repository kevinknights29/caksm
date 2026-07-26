/**
 * @file reduction.hpp
 * @brief The global-reduction primitive, in the two forms the study has to tell apart.
 *
 * R_h's numerator models one global reduction as a tree of depth log2(P).
 * The tree avoids barriers (one team barrier over 24 threads costs ~1 us): each node waits
 * only on its own children via flag-based synchronization, checked by verify_reducer(),
 * which every caller must run before timing. Both forms assume a bound team no larger than
 * the physical cores, since the tree spins rather than blocking.
 *
 * @author Kevin Knights
 * @date 2026-07-17
 */
#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

/// Which reduction primitive a run uses.
enum class ReduceKind {
    LINEAR,  ///< barrier + O(P) redundant scan: the artifact, kept so it can be measured
    TREE,    ///< binary tree, point-to-point flag sync: the primitive to calibrate against
};

[[nodiscard]] inline const char* reduce_kind_name(ReduceKind k) noexcept
{
    return k == ReduceKind::LINEAR ? "linear" : "tree";
}

/// Partial-buffer layout for the LINEAR form, kept only as a false-sharing diagnostic.
enum class Layout {
    SHARED,  ///< P doubles packed into ceil(P/8) lines; what the harness ran
    PADDED,  ///< one cache line per thread
};

[[nodiscard]] inline const char* layout_name(Layout l) noexcept
{
    return l == Layout::SHARED ? "shared" : "padded";
}

/**
 * @brief One thread's tree node: its partial, its generation flag, and nothing shared.
 *
 * Cache-line aligned so a node's flag never shares a line with a neighbor's: a node spins
 * on its child's flag, and false sharing would turn every spin into a coherence storm.
 * `local_gen` is owner-private, kept here to stay on the thread's own line.
 */
struct alignas(64) ReduceSlot {
    double                val = 0.0;   ///< this subtree's running sum
    std::atomic<uint32_t> gen{0};      ///< published generation: val is readable when set
    uint32_t              local_gen = 0;  ///< owner-private call counter
};

/**
 * @brief A reduction over a fixed, bound OpenMP team.
 *
 * Constructed outside a parallel region, then reduce() is called by every thread inside
 * one. Every thread must call it the same number of times and in the same order: the
 * generation counters serialize successive reductions, and a skipped call desynchronizes
 * the team permanently.
 */
class TeamReducer {
public:
    TeamReducer(int P, ReduceKind kind, Layout layout = Layout::SHARED)
        : P_(P), kind_(kind), layout_(layout),
          slots_(static_cast<std::size_t>(P)),
          // Two buffers: reduce_linear alternates on the generation's parity so it needs
          // only one barrier. See reduce_linear.
          linear_(2u * static_cast<std::size_t>(P) * stride_of(layout), 0.0)
    {
        if (P < 1) throw std::invalid_argument("TeamReducer: P must be >= 1");
    }

    /// Doubles between adjacent thread slots in the LINEAR buffer.
    [[nodiscard]] static constexpr std::size_t stride_of(Layout l) noexcept
    {
        return l == Layout::SHARED ? 1u : 8u;   // 8 doubles = 64 B
    }

    /**
     * @brief All-reduce `contrib` across the team; every thread returns the total.
     *
     * @param t thread index within the team (omp_get_thread_num()).
     */
    [[nodiscard]] double reduce(int t, double contrib)
    {
        return kind_ == ReduceKind::TREE ? reduce_tree(t, contrib)
                                         : reduce_linear(t, contrib);
    }

private:
    /**
     * The artifact, preserved so its cost can be measured: one team barrier and an O(P)
     * scan run redundantly on every thread.
     *
     * ONE barrier, not two: the buffer is double-buffered on the generation's parity, so
     * reduction g+1 writes the buffer g is not reading and the single barrier orders g+1's
     * deposits against g's scan. This matches the historical inline code in scaling.cpp,
     * which also paid one barrier per reduction; a two-barrier version would make this arm
     * slower than the code it reproduces and overstate the correction the tree buys.
     */
    [[nodiscard]] double reduce_linear(int t, double contrib)
    {
        const std::size_t stride = stride_of(layout_);
        ReduceSlot& me = slots_[static_cast<std::size_t>(t)];
        const std::size_t parity = (++me.local_gen) & 1u;
        const std::size_t base   = parity * static_cast<std::size_t>(P_) * stride;

        linear_[base + static_cast<std::size_t>(t) * stride] = contrib;
        #pragma omp barrier
        double acc = 0.0;
        for (int k = 0; k < P_; ++k)
            acc += linear_[base + static_cast<std::size_t>(k) * stride];
        return acc;
    }

    /**
     * Binary tree up-phase, then a single-flag broadcast.
     *
     * UP-PHASE. At level d a still-climbing thread has (t & (2d-1)) == 0 and receives from
     * t+d; (t & (2d-1)) == d means its subtree is complete, so it publishes and stops. A
     * thread waits only on the child it is about to add, never on the team.
     *
     * ORDERING. `val` is written before `gen` (release) and read after `gen` (acquire), so
     * the partial is visible exactly when the flag says it is.
     *
     * The broadcast doubles as the inter-iteration barrier: a thread cannot start g+1 and
     * overwrite its `val` until thread 0, which sets the broadcast flag only after the full
     * up-phase, has read it.
     */
    [[nodiscard]] double reduce_tree(int t, double contrib)
    {
        ReduceSlot& me = slots_[static_cast<std::size_t>(t)];
        const uint32_t g = ++me.local_gen;
        me.val = contrib;

        for (int d = 1; d < P_; d <<= 1) {
            const int mask = (d << 1) - 1;
            if ((t & mask) == 0) {
                const int partner = t + d;
                if (partner < P_) {
                    ReduceSlot& ch = slots_[static_cast<std::size_t>(partner)];
                    while (ch.gen.load(std::memory_order_acquire) != g) { /* spin on one line */ }
                    me.val += ch.val;
                }
            } else {
                // (t & mask) == d: subtree complete. Publish and leave the climb.
                me.gen.store(g, std::memory_order_release);
                break;
            }
        }

        if (t == 0) {
            bcast_ = me.val;
            bcast_gen_.store(g, std::memory_order_release);
        } else {
            while (bcast_gen_.load(std::memory_order_acquire) != g) { /* spin */ }
        }
        return bcast_;
    }

    int        P_;
    ReduceKind kind_;
    Layout     layout_;
    std::vector<ReduceSlot> slots_;    ///< tree nodes; also carries each thread's local_gen
    std::vector<double>     linear_;   ///< LINEAR partial buffers, two of them

    alignas(64) double                bcast_ = 0.0;
    alignas(64) std::atomic<uint32_t> bcast_gen_{0};
};

/**
 * @brief Check a reducer against a closed-form sum. Returns an empty string on success.
 *
 * A subtly wrong tree does not crash; it returns a plausible number slightly too small,
 * which would propagate into every timing without a symptom. Contribution 1 + t sums to
 * P(P+1)/2 exactly in double, so any mismatch is a real bug. Several rounds are run because
 * generation g colliding with g+1 cannot appear in a single round.
 *
 * @return "" if every round reduced correctly, else a description of the first mismatch.
 */
[[nodiscard]] inline std::string verify_reducer(int P, ReduceKind kind, Layout layout,
                                                int rounds = 64)
{
    TeamReducer red(P, kind, layout);
    const double expected = static_cast<double>(P) * static_cast<double>(P + 1) / 2.0;

    std::vector<double> got(static_cast<std::size_t>(rounds) * static_cast<std::size_t>(P), 0.0);

    #pragma omp parallel num_threads(P)
    {
        const int t = omp_get_thread_num();
        for (int r = 0; r < rounds; ++r) {
            const double acc = red.reduce(t, 1.0 + static_cast<double>(t));
            got[static_cast<std::size_t>(r) * static_cast<std::size_t>(P)
                + static_cast<std::size_t>(t)] = acc;
        }
    }

    for (int r = 0; r < rounds; ++r)
        for (int t = 0; t < P; ++t) {
            const double v = got[static_cast<std::size_t>(r) * static_cast<std::size_t>(P)
                               + static_cast<std::size_t>(t)];
            if (v != expected)
                return "reduction is WRONG: P=" + std::to_string(P)
                     + " kind=" + reduce_kind_name(kind)
                     + " round=" + std::to_string(r)
                     + " thread=" + std::to_string(t)
                     + " got=" + std::to_string(v)
                     + " expected=" + std::to_string(expected);
        }
    return {};
}
