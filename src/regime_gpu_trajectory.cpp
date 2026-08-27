/**
 * @file regime_gpu_trajectory.cpp
 * @brief Places the real Basket operator on the GPU regime map once per participant/node
 *        topology, and writes the canonical CSV the poster's trajectory figure reads.
 *
 * The topology-aware companion to regime_gpu_place.cpp. That tool sweeps a grid against one
 * device and one rung; this one holds the operator fixed and sweeps the arrangement, which is
 * the axis the map predicts but nothing so far has drawn.
 *
 * Host-only and CUDA-free, like its sibling, so it runs on a login node or a laptop. Nothing
 * here links a solver or a timer. The three inputs are all measurements taken before this tool
 * ran: the operator and its converged Krylov dimension come from the accepted placement table,
 * the device capacities from the calibrated preset, and the collective latency from the
 * calibration that measured that exact arrangement. What the tool contributes is the
 * predeclared byte model between them.
 *
 * Two policies, answering different questions and never joined by a line:
 *
 *   fixed-global  One global grid sequence, every topology. Sharding cuts the local working
 *                 set, so R_v falls with participant count while R_h rises. This is the
 *                 refinement trajectory of panel B.
 *   fixed-local   The global grid grows with the participant count so that the heaviest slab
 *                 stays about the same size. R_v is then held roughly still and the motion in
 *                 R_h is the collective's alone, which is what panel C tests.
 *
 * The measured Krylov dimension is read from data/regime/regime_placement.csv rather than
 * taken on the command line, so it cannot be invented at a call site. The tool also recomputes
 * N and nnz from its own stencil model and refuses any row whose values disagree with the
 * table: that check is what establishes that the placement measurement and this model describe
 * one operator.
 *
 * The arrangements come from kGpuTopologies, keyed by machine, so one tool serves a two-GPU
 * cluster and an eight-GPU node without an edit. Selecting a subset by name is what a
 * publication run does, so it repeats the same arrangements rather than whatever the table has
 * since grown.
 *
 * Usage:
 *   ./regime-gpu-trajectory [--machine v100-pcie-16gb] [--policy both|fixed-global|fixed-local]
 *                           [--placement-csv PATH]
 *                           [--topologies "1gpu-1node 8gpu-1node"]
 *                           [--n1-list "25 30 40 ..."]
 *                           [--local-ref 40] [--s 8] [--x-reuse 1.0] [--csv PATH]
 *
 * @author Kevin Knights
 * @date 2026-08-23
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <print>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_machine.hpp"
#include "gpu_regime.hpp"
#include "gpu_topology.hpp"

namespace {

/// One accepted row of the placement table: the operator's measured properties at one grid.
struct PlacementRow {
    int     n1        = 0;
    int64_t n         = 0;
    int64_t nnz       = 0;
    int     m         = 0;
    int     exit_status = 1;
    std::string report;
    std::string validation;   ///< PASS, or why the row was rejected
};

/// One per-topology solver run: what the arrangement actually converged to.
struct MeasuredRow {
    int         m         = 0;
    std::string validation;      ///< PASS, or the verdict that blocked it
    bool        contended = true;
    std::string log;
};

/// Keyed (topology, n). Empty when no synge measurement run has been made.
using MeasuredTable = std::map<std::pair<std::string, int>, MeasuredRow>;

struct Args {
    std::string_view machine = "v100-pcie-16gb";
    std::string policy = "both";
    std::string placement_csv = "data/regime/regime_placement.csv";
    /// Topology keys to place, in the order given. Empty places every arrangement the table
    /// carries for this machine. Naming them is what makes a publication run repeat the same
    /// arrangements rather than whatever the table has grown since.
    std::vector<std::string> topologies;
    std::vector<int> n1_list;     ///< empty = every accepted grid in the table
    int    local_ref = 40;        ///< single-GPU grid the fixed-local arm is sized against
    std::string measured_csv;     ///< optional per-topology solver measurements
    /**
     * The shared numerical contract's Krylov ceiling, which is the GPU build's
     * kGpuCaMaxM. A grid needing more than this converges on the CPU and cannot
     * converge on the GPU, so it is not a comparable point and is dropped from
     * both sides rather than drawn on one.
     */
    int    contract_m_ceiling = 39;
    int    s = 8;
    double x_reuse = 1.0;
    std::string csv_path;
    std::string ladder_csv;       ///< optional reduction-ladder table
    int    ladder_n1 = 61;        ///< grid whose measured m prices the ladder
};

[[nodiscard]] std::vector<int> parse_int_list(std::string_view s)
{
    std::vector<int> out;
    std::istringstream in{std::string(s)};
    int v = 0;
    while (in >> v) out.push_back(v);
    if (out.empty()) throw std::invalid_argument("Empty integer list");
    return out;
}

[[nodiscard]] std::vector<std::string> parse_word_list(std::string_view s)
{
    std::vector<std::string> out;
    std::istringstream in{std::string(s)};
    std::string w;
    while (in >> w) out.push_back(w);
    if (out.empty()) throw std::invalid_argument("Empty topology list");
    return out;
}

[[nodiscard]] Args parse_args(std::span<const char* const> argv)
{
    Args a;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view arg = argv[i];
        auto next = [&]() -> std::string_view {
            if (++i >= argv.size())
                throw std::invalid_argument("Missing value for " + std::string(arg));
            return argv[i];
        };
        if      (arg == "--machine")       a.machine = next();
        else if (arg == "--policy")        a.policy = std::string(next());
        else if (arg == "--placement-csv") a.placement_csv = std::string(next());
        else if (arg == "--topologies")    a.topologies = parse_word_list(next());
        else if (arg == "--n1-list")       a.n1_list = parse_int_list(next());
        else if (arg == "--local-ref")     a.local_ref = std::stoi(std::string(next()));
        else if (arg == "--measured-csv")  a.measured_csv = std::string(next());
        else if (arg == "--contract-m-ceiling")
            a.contract_m_ceiling = std::stoi(std::string(next()));
        else if (arg == "--s")             a.s = std::stoi(std::string(next()));
        else if (arg == "--x-reuse")       a.x_reuse = std::stod(std::string(next()));
        else if (arg == "--csv")           a.csv_path = std::string(next());
        else if (arg == "--ladder-csv")    a.ladder_csv = std::string(next());
        else if (arg == "--ladder-n1")     a.ladder_n1 = std::stoi(std::string(next()));
        else if (arg == "--help") {
            std::println("Usage: ./regime-gpu-trajectory [--machine KEY] [--policy P]");
            std::println("                               [--placement-csv PATH]");
            std::println("                               [--n1-list \"25 30 40\"] [--local-ref N]");
            std::println("                               [--s S] [--x-reuse F] [--csv PATH]");
            std::println("");
            std::println("  --machine KEY   v100-pcie-16gb (synge) or h200 (gpu03). The preset");
            std::println("                  must carry a measured ladder and a measured");
            std::println("                  bandwidth.");
            std::println("  --topologies    a quoted list of topology keys, in the order to place");
            std::println("                  them, e.g. \"1gpu-1node 8gpu-1node\". Default: every");
            std::println("                  arrangement kGpuTopologies carries for this machine.");
            std::println("                  Name them for a publication run, so it repeats the");
            std::println("                  same arrangements rather than whatever the table has");
            std::println("                  since grown.");
            std::println("  --policy P      fixed-global, fixed-local, or both (default).");
            std::println("  --placement-csv the accepted placement table. Supplies the measured");
            std::println("                  Krylov dimension and the operator's N and nnz, which");
            std::println("                  are cross-checked against the stencil model here.");
            std::println("  --n1-list       global grids for the fixed-global arm. Default: every");
            std::println("                  accepted grid in the placement table.");
            std::println("  --contract-m-ceiling M");
            std::println("                  the shared contract's Krylov ceiling (default 39,");
            std::println("                  the GPU build's kGpuCaMaxM). A grid whose measured m");
            std::println("                  reaches it is dropped: it converges on the CPU and");
            std::println("                  cannot on the GPU, so it is not a comparable point.");
            std::println("  --measured-csv  per-topology solver measurements from a synge run,");
            std::println("                  as scripts/regime/regime_gpu_trajectory.sh MEASURE=1");
            std::println("                  writes them. A (topology, n) covered there takes its");
            std::println("                  Krylov dimension and its validation verdict from that");
            std::println("                  run; anything not covered stays PLACEMENT-ONLY.");
            std::println("  --local-ref N   the single-GPU grid the fixed-local arm sizes its");
            std::println("                  heaviest slab against (default 40). Each topology");
            std::println("                  then takes the accepted grid whose slab comes");
            std::println("                  closest to that size.");
            std::println("  --ladder-csv    also write the reduction ladder: every rung's measured");
            std::println("                  cost against tau*, the cost at which R_v * R_h reaches 1");
            std::println("                  and the Upper-Right corner opens.");
            std::println("  --ladder-n1 N   the grid whose measured Krylov dimension prices that");
            std::println("                  ladder (default 61, the production point). tau* is free");
            std::println("                  of N but not of m, so the grid has to be named.");
            std::println("  --s S           certified block width, for the Gram gate only.");
            std::println("  --x-reuse F     SpMV source-vector reuse in [0, 1] (default 1).");
            std::exit(0);
        }
        else throw std::invalid_argument("Unknown flag: " + std::string(arg));
    }
    if (a.policy != "both" && a.policy != "fixed-global" && a.policy != "fixed-local")
        throw std::invalid_argument("--policy must be fixed-global, fixed-local or both");
    if (a.s < 1) throw std::invalid_argument("--s must be >= 1");
    if (!(a.x_reuse >= 0.0 && a.x_reuse <= 1.0))
        throw std::invalid_argument("--x-reuse must be in [0, 1]");
    return a;
}

/**
 * @brief Read the accepted placement table, keeping only rows that clear every gate.
 *
 * A row is accepted when it exited cleanly, is the only row for its grid, and its recorded N
 * and nnz match what basket_nnz() computes. The last check is the load-bearing one: it is what
 * says the operator this tool models is the operator that was measured, and a silent
 * disagreement there would move every point on the figure.
 */
[[nodiscard]] std::map<int, PlacementRow> load_placement(const std::string& path,
                                                        int contract_ceiling)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open placement table: " + path);

    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("Placement table is empty: " + path);

    std::vector<std::string> header;
    for (std::stringstream hs(line); std::getline(hs, line, ','); )
        header.push_back(line);
    auto column = [&](std::string_view want) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i)
            if (header[i] == want) return i;
        throw std::runtime_error("Placement table lacks a '" + std::string(want) + "' column");
    };
    const std::size_t c_n = column("n"), c_N = column("N"), c_nnz = column("nnz");
    const std::size_t c_m = column("m_measured"), c_exit = column("exit_status");
    const std::size_t c_report = column("report");
    const std::size_t c_ceil = column("m_ceiling");

    std::map<int, PlacementRow> rows;
    std::size_t duplicates = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f;
        for (std::stringstream fs(line); std::getline(fs, line, ','); ) f.push_back(line);
        if (f.size() <= c_report) continue;

        PlacementRow r;
        r.n1          = std::stoi(f[c_n]);
        r.n           = std::stoll(f[c_N]);
        r.nnz         = std::stoll(f[c_nnz]);
        r.m           = std::stoi(f[c_m]);
        r.exit_status = std::stoi(f[c_exit]);
        r.report      = f[c_report];
        const int row_ceiling = std::stoi(f[c_ceil]);

        // A row whose search saturated its own ceiling recorded the ceiling, not a
        // converged dimension: regime-control warns and still exits 0, so the CSV
        // alone cannot be trusted on that point without this check.
        if (r.exit_status != 0)                    r.validation = "FAIL:exit_status";
        else if (r.m < 1)                          r.validation = "FAIL:m";
        else if (r.m >= row_ceiling)               r.validation = "FAIL:saturated";
        else if (r.m > contract_ceiling)           r.validation = "FAIL:exceeds-contract";
        else if (r.n != grid_points(r.n1, 3))      r.validation = "FAIL:N-mismatch";
        else if (r.nnz != basket_nnz(r.n1))        r.validation = "FAIL:nnz-mismatch";
        else                                       r.validation = "PASS";

        if (!rows.emplace(r.n1, r).second) {
            ++duplicates;
            rows[r.n1].validation = "FAIL:duplicate";
        }
    }
    if (duplicates)
        std::println("  WARNING: {} duplicate grid(s) in the placement table; those grids are "
                     "blocked.", duplicates);
    return rows;
}

/**
 * @brief What the `validation` column of every row this tool writes says.
 *
 * Not PASS, and there is no flag that can make it PASS. A row here is placed from
 * measured constants but nothing in this process ran a solver, so no numerical
 * result was validated and no launch was checked against its declared topology.
 * The specification's acceptance gate asks for both, and meeting it needs a
 * measurement path on synge that this host-only tool deliberately does not have.
 *
 * The placement table's own verdict is not thrown away; it travels in the
 * `operator_validation` column, where it says what it actually covers.
 */
constexpr const char* kStance = "PLACEMENT-ONLY";

/**
 * @brief Read the per-topology solver measurements, if a synge run has produced any.
 *
 * Absent by default, and absence is not an error: the host-only path is the one
 * that runs off the cluster. A row present here upgrades its (topology, grid) from
 * a placed point to a measured one, and only then may `validation` say PASS.
 */
[[nodiscard]] MeasuredTable load_measured(const std::string& path)
{
    MeasuredTable table;
    if (path.empty()) return table;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open measured table: " + path);

    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("Measured table is empty: " + path);
    std::vector<std::string> header;
    for (std::stringstream hs(line); std::getline(hs, line, ','); )
        header.push_back(line);
    auto column = [&](std::string_view want) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i)
            if (header[i] == want) return i;
        throw std::runtime_error("Measured table lacks a '" + std::string(want)
                                 + "' column");
    };
    const std::size_t c_top = column("topology"), c_n = column("n_global");
    const std::size_t c_m = column("m_measured"), c_val = column("validation");
    const std::size_t c_con = column("contended"), c_log = column("log");

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f;
        for (std::stringstream fs(line); std::getline(fs, line, ','); ) f.push_back(line);
        if (f.size() <= c_log) continue;
        MeasuredRow r;
        r.m          = std::stoi(f[c_m]);
        r.validation = f[c_val];
        r.contended  = f[c_con] != "0";
        r.log        = f[c_log];
        table[{f[c_top], std::stoi(f[c_n])}] = r;
    }
    return table;
}

/// Quote a CSV field that may hold a comma. Named away from std::quoted, which ADL would
/// otherwise select for a std::string argument.
[[nodiscard]] std::string csv_quote(std::string_view s)
{
    return "\"" + std::string(s) + "\"";
}

/// The canonical CSV header. The declared prefix comes first, in the order the spec fixes it;
/// the auditing columns follow, so a reader parsing only the prefix still gets what it expects.
constexpr const char* kHeader =
    "machine,policy,topology,participants,nodes,local_gpus,n_global,N_global,"
    "N_local_max,nnz_local_max,m_measured,rv,rh,reduction_median_s,"
    "compute_between_reductions_s,repeats,contended,validation,report,"
    "tier,planes_max,rh_q1,rh_q3,reduction_q1_s,reduction_q3_s,nnz_global,"
    "working_set_bytes,l2_bytes,reductions,cycle_s,ai_spmv,ai_mgs,ai_gram,"
    "gate_spmv,gate_mgs,gate_gram,on_map,accepted,spmv_resident,mgs_resident,"
    "operator_validation,operator_machine,instrument,source";

void write_row(std::ostream& out, const GpuMachine& gm, std::string_view policy,
               const GpuTrajectoryPoint& pt, const PlacementRow& pr,
               const MeasuredRow* mr)
{
    const GpuTopology& t = *pt.topology;
    // A point only claims PASS when a solver actually ran at this arrangement and
    // this grid on an uncontended device. Everything else says what it is.
    const std::string stance = (mr && mr->validation == "PASS" && !mr->contended)
        ? std::string("PASS") : std::string(kStance);
    const std::string report = mr ? mr->log : pr.report;
    std::println(out,
        "{},{},{},{},{},{},{},{},{},{},{},{:.6g},{:.6g},{:.6e},{:.6e},{},{},{},{},"
        "{},{},{:.6g},{:.6g},{:.6e},{:.6e},{},{:.6e},{},{},{:.6e},{:.6g},{:.6g},{:.6g},"
        "{},{},{},{},{},{},{},{},{},{},{}",
        gm.key, policy, t.key, t.participants, t.nodes, t.local_gpus,
        pt.n1_global, pt.n_global, pt.n_local_max, pt.nnz_local_max, pt.m,
        pt.rv, pt.rh, t.t_reduce_s, pt.compute_between_reductions_s,
        t.repeats, t.contended ? 1 : 0, stance, csv_quote(report),
        tier_name(t.tier), pt.planes_max, pt.rh_q1, pt.rh_q3,
        t.t_reduce_q1_s, t.t_reduce_q3_s, pt.nnz_global,
        pt.working_set, static_cast<int64_t>(pt.l2), pt.reductions, pt.cycle_s,
        pt.gate_spmv.ai, pt.gate_mgs.ai, pt.gate_gram.ai,
        pt.gate_spmv.memory_bound ? 1 : 0, pt.gate_mgs.memory_bound ? 1 : 0,
        pt.gate_gram.memory_bound ? 1 : 0, pt.on_map ? 1 : 0,
        (pt.accepted && pr.validation == "PASS") ? 1 : 0,
        pt.spmv_resident ? 1 : 0, pt.mgs_resident ? 1 : 0,
        pr.validation, mr ? "v100-pcie-16gb (synge)" : "amd-3960x (puffin)",
        csv_quote(t.instrument), csv_quote(t.source));
}

/**
 * @brief Write the reduction ladder: each rung's measured cost against tau*.
 *
 * The ladder is the instrument the map is read with. Holding the operator and the
 * device fixed and changing only the rung moves R_h alone, so the corner opens at
 * whichever rung first costs more than tau*. That threshold is a property of the
 * operator and the machine, not of the rung, so it is one number for the whole
 * table and the comparison to read is each rung against it.
 *
 * tau* is free of N, since the working set and the cycle bytes are both linear in
 * it and the ratio cancels, but it is not free of m: more reductions per cycle
 * means a cheaper one suffices to reach the threshold. The grid is named for that
 * reason alone.
 */
void write_ladder(const std::string& path, const GpuMachine& gm, int n1, int m,
                  double x_reuse)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);

    const int64_t n   = grid_points(n1, 3);
    const int64_t nnz = basket_nnz(n1);
    const double  bw  = achieved_or_peak_bw_gbs(gm);

    std::println(out, "tier,label,t_reduce_s,t_reduce_q1_s,t_reduce_q3_s,lambda,"
                      "product,tau_star_s,corner_open,calibrated,n_global,m_measured,"
                      "instrument,source");

    // Poster labels: what each rung physically is, rather than its enum name. The
    // CUDA thread block is written in full, since "block" alone is a matrix-powers
    // group of vectors everywhere else in this project.
    struct Rung { ReductionTier tier; const char* label; };
    constexpr std::array<Rung, kTierCount> kRungs{{
        {ReductionTier::WARP,       "warp shuffle"},
        {ReductionTier::BLOCK,      "thread block"},
        {ReductionTier::GRID,       "grid, 2nd launch"},
        {ReductionTier::DEVICE_P2P, "device, intra-node"},
        {ReductionTier::NODE,       "node, fabric"},
        // Commas inside these are quoted at the point of writing below.
    }};

    for (const auto& [tier, label] : kRungs) {
        const RegimeInvariant ri =
            regime_invariant(gm, tier, nnz, n, m, x_reuse, bw);
        const double t = reduction_cost_s(gm, tier);
        // Where a topology measured this rung for its own arrangement, its
        // interquartile range travels with it; the on-device rungs have none of
        // their own, so the columns repeat the median rather than invent a spread.
        const GpuTopology* owner = nullptr;
        for (const auto& top : kGpuTopologies)
            if (top.machine == gm.key && top.tier == tier
                && std::abs(top.t_reduce_s / t - 1.0) < 0.02) { owner = &top; break; }
        std::println(out, "{},{},{:.6e},{:.6e},{:.6e},{:.6g},{:.6g},{:.6e},{},{},{},{},{},{}",
                     tier_name(tier), csv_quote(label), t,
                     owner ? owner->t_reduce_q1_s : t,
                     owner ? owner->t_reduce_q3_s : t,
                     ri.lambda, ri.product, ri.tau_star_s,
                     ri.upper_right ? 1 : 0,
                     gm.tier_calibrated[tier_index(tier)] ? 1 : 0,
                     n1, m,
                     csv_quote(owner ? owner->instrument : "calibrate-gpu-reduction"),
                     csv_quote(owner ? owner->source
                                     : "data/regime/calibrate_gpu_reduction.csv"));
    }
}

/**
 * @brief The accepted grid whose heaviest slab comes closest to the reference slab size.
 *
 * The fixed-local arm needs one grid per topology such that the local problem barely moves.
 * Interpolating m to hit an exactly constant slab would put an unmeasured Krylov dimension on
 * the figure, so the grid is chosen from the accepted set instead and the residual spread in
 * local size is reported rather than hidden. On synge's four topologies against a 40^3
 * reference the choice is 40, 50, 50 and 61, holding the heaviest slab within 8%.
 */
[[nodiscard]] int grid_for_fixed_local(const std::map<int, PlacementRow>& table,
                                       int participants, int64_t target_local)
{
    int best = 0;
    double best_err = 0.0;
    for (const auto& [n1, row] : table) {
        if (row.validation != "PASS") continue;
        const double err = std::abs(
            std::log(static_cast<double>(slab_points_max(n1, participants))
                     / static_cast<double>(target_local)));
        if (best == 0 || err < best_err) { best = n1; best_err = err; }
    }
    return best;
}

void report_point(const GpuTrajectoryPoint& pt, const PlacementRow& pr, bool measured)
{
    const GpuTopology& t = *pt.topology;
    std::println("  {:<16} n={:<4} N_local_max={:<9} m={:<3} R_v={:<9.4g} R_h={:<9.4g} "
                 "t_red={:>6.2f} us  {}",
                 t.key, pt.n1_global, pt.n_local_max, pt.m, pt.rv, pt.rh,
                 t.t_reduce_s * 1e6,
                 !(pt.accepted && pr.validation == "PASS") ? "BLOCKED"
                 : measured ? "accepted (measured)" : "accepted (placed)");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Args a = parse_args(std::span<const char* const>(argv, argc));
        const GpuMachine& gm = lookup_gpu_machine(a.machine);

        if (!gm.reduction_calibrated || !gm.roofline_gated)
            throw std::runtime_error(
                "Preset " + std::string(gm.key) + " is not fully calibrated. Run "
                "scripts/regime/calibrate_gpu.sh and calibrate_gpu_p2p.sh first; this figure "
                "admits no assumed constants.");

        std::println("===============================================================================");
        std::println(" GPU regime trajectory   machine={}  s={}  x_reuse={}",
                     gm.key, a.s, a.x_reuse);
        std::println("===============================================================================");
        std::println("");

        // The arrangements this machine has been measured on. An explicitly named list is
        // placed in the order given, and a key it names that the table does not carry is an
        // error rather than a silent omission: a publication run asking for 8gpu-1node must
        // fail if that arrangement is unmeasured, not quietly draw one point fewer.
        std::size_t available = 0;
        for (const auto& t : kGpuTopologies) if (t.machine == gm.key) ++available;
        if (available == 0)
            throw std::runtime_error(
                "No arrangement in kGpuTopologies is recorded for machine '"
                + std::string(gm.key) + "'. Calibrate one with "
                  "scripts/regime/calibrate_gpu_p2p.sh and record it in "
                  "include/gpu_topology.hpp before placing a trajectory on it.");

        std::vector<const GpuTopology*> selected;
        if (a.topologies.empty()) {
            for (const auto& t : kGpuTopologies)
                if (t.machine == gm.key) selected.push_back(&t);
        } else {
            for (const std::string& key : a.topologies) {
                const GpuTopology* t = lookup_topology(gm.key, key);
                if (!t)
                    throw std::runtime_error(
                        "Topology '" + key + "' is not recorded for machine '"
                        + std::string(gm.key) + "'. Calibrate that arrangement before asking "
                          "for it; an unmeasured one is absent on purpose and must not be "
                          "interpolated.");
                selected.push_back(t);
            }
        }

        std::println("Topologies: {} selected of {} recorded for {}",
                     selected.size(), available, gm.key);
        for (const GpuTopology* t : selected)
            std::println("            {:<16} {:>2} participant(s) over {} node(s), rung {:<11} "
                         "t_red={:>7.3f} us  {}",
                         t->key, t->participants, t->nodes, tier_name(t->tier),
                         t->t_reduce_s * 1e6,
                         topology_placeable(*t) ? "placeable"
                         : !t->calibrated       ? "BLOCKED: uncalibrated"
                         : t->contended         ? "BLOCKED: contended"
                                                : "BLOCKED: no link signature");
        std::println("");

        const auto table = load_placement(a.placement_csv, a.contract_m_ceiling);
        const auto measured = load_measured(a.measured_csv);
        const std::size_t passing = static_cast<std::size_t>(std::count_if(
            table.begin(), table.end(),
            [](const auto& kv) { return kv.second.validation == "PASS"; }));
        std::println("Placement table: {} grid(s), {} passing every gate  [{}]",
                     table.size(), passing, a.placement_csv);
        std::println("Shared contract: m must converge below {} on both machines "
                     "(the GPU build's\n                 kGpuCaMaxM). Grids failing it "
                     "are dropped from every panel:", a.contract_m_ceiling);
        bool any_dropped = false;
        for (const auto& [n1, row] : table) {
            if (row.validation == "PASS") continue;
            any_dropped = true;
            std::println("                 n={} m={} -> {}", n1, row.m, row.validation);
        }
        if (!any_dropped)
            std::println("                 none; every grid converges under the contract.");
        if (measured.empty())
            std::println("Measured table:  none. Every row will say PLACEMENT-ONLY. Run\n"
                         "                 MEASURE=1 scripts/regime/regime_gpu_trajectory.sh "
                         "on synge\n                 to take the per-topology solver runs.");
        else
            std::println("Measured table:  {} (topology, grid) run(s)  [{}]",
                         measured.size(), a.measured_csv);
        if (passing == 0)
            throw std::runtime_error("No placement row survives validation; nothing to place.");
        std::println("");

        std::ofstream csv;
        if (!a.csv_path.empty()) {
            csv.open(a.csv_path);
            if (!csv) throw std::runtime_error("Cannot write " + a.csv_path);
            std::println(csv, "{}", kHeader);
        }

        std::size_t written = 0, blocked = 0;
        auto emit = [&](std::string_view policy, const GpuTopology& t, int n1) {
            const auto it = table.find(n1);
            if (it == table.end()) {
                std::println("  {:<16} n={:<4} BLOCKED: not in the placement table", t.key, n1);
                ++blocked;
                return;
            }
            const PlacementRow& pr = it->second;
            // The measured Krylov dimension for this exact arrangement where a synge
            // run supplied one. Falling back to the puffin value is sound only
            // because m is a property of the operator and the tolerance, which is
            // what the placement report states; a measurement at the arrangement
            // itself is still better, so it wins whenever it exists.
            const auto mit = measured.find({std::string(t.key), n1});
            const MeasuredRow* mr = mit != measured.end() ? &mit->second : nullptr;
            if (mr && (mr->validation != "PASS" || mr->contended)) {
                std::println("  {:<16} n={:<4} BLOCKED: synge run says validation={} "
                             "contended={}", t.key, n1, mr->validation,
                             mr->contended ? 1 : 0);
                ++blocked;
                return;
            }
            const int m = mr ? mr->m : pr.m;
            const GpuTrajectoryPoint pt =
                place_gpu_trajectory(gm, t, n1, m, a.x_reuse, Precision::FP64, a.s);
            report_point(pt, pr, mr != nullptr);
            if (!(pt.accepted && pr.validation == "PASS")) { ++blocked; return; }
            if (csv) { write_row(csv, gm, policy, pt, pr, mr); ++written; }
        };

        if (a.policy == "both" || a.policy == "fixed-global") {
            std::vector<int> grids = a.n1_list;
            if (grids.empty())
                for (const auto& [n1, row] : table)
                    if (row.validation == "PASS") grids.push_back(n1);
            std::println("-- fixed-global: one grid sequence, every topology --------------------------");
            for (const GpuTopology* t : selected)
                for (int n1 : grids) emit("fixed-global", *t, n1);
            std::println("");
        }

        if (a.policy == "both" || a.policy == "fixed-local") {
            const int64_t target = slab_points_max(a.local_ref, 1);
            std::println("-- fixed-local: heaviest slab held near {} rows ({}^3 on one GPU) ----------",
                         target, a.local_ref);
            int64_t lo = 0, hi = 0;
            for (const GpuTopology* t : selected) {
                const int n1 = grid_for_fixed_local(table, t->participants, target);
                if (n1 == 0) { ++blocked; continue; }
                const int64_t local = slab_points_max(n1, t->participants);
                lo = (lo == 0) ? local : std::min(lo, local);
                hi = std::max(hi, local);
                emit("fixed-local", *t, n1);
            }
            if (lo > 0)
                std::println("  heaviest slab spans {} to {} rows, a {:.1f}% spread; the residual is "
                             "reported,\n  not removed, because removing it would need an "
                             "unmeasured Krylov dimension.",
                             lo, hi, 100.0 * (static_cast<double>(hi) / static_cast<double>(lo) - 1.0));
            std::println("");
        }

        if (!a.ladder_csv.empty()) {
            const auto lit = table.find(a.ladder_n1);
            if (lit == table.end() || lit->second.validation != "PASS")
                throw std::runtime_error(
                    "--ladder-n1 " + std::to_string(a.ladder_n1)
                    + " is not an accepted grid, so its Krylov dimension cannot price "
                      "the ladder.");
            const auto mit = measured.find({"1gpu-1node", a.ladder_n1});
            const int lm = mit != measured.end() && mit->second.validation == "PASS"
                         ? mit->second.m : lit->second.m;
            write_ladder(a.ladder_csv, gm, a.ladder_n1, lm, a.x_reuse);
            std::println("");
            std::println("-- reduction ladder at n={}, m={} ------------------------------------------",
                         a.ladder_n1, lm);
            std::println("  {}", a.ladder_csv);
        }

        std::println("===============================================================================");
        std::println("{} row(s) accepted, {} blocked.", written, blocked);
        if (!a.csv_path.empty()) std::println("  {}", a.csv_path);
        std::println("");
        std::println("Reading it:");
        if (measured.empty()) {
            std::println("  - validation=PLACEMENT-ONLY on every row. The coordinates are placed");
            std::println("    from measured constants, but no solver ran here, so no numerical");
            std::println("    result was validated on synge. operator_validation carries the");
            std::println("    placement table's verdict, which covers the operator and m only.");
        } else {
            std::println("  - validation=PASS marks a row whose arrangement and grid were run on");
            std::println("    synge and validated there; PLACEMENT-ONLY marks one that was not.");
            std::println("    A run reporting FAIL or a contended device blocks its point rather");
            std::println("    than being drawn with a caveat.");
        }
        std::println("  - accepted=1 means every gate this host-only tool can check held: the");
        std::println("    roofline gate, the topology's idle-device gate, and that placement row.");
        std::println("  - rv is the HEAVIEST SLAB's working set over ONE device's L2. Distributed GPU");
        std::println("    caches are not coherent, so the aggregate would be a fiction.");
        std::println("  - rh uses the collective measured for that exact participant and node count,");
        std::println("    not reduction_cost_s(tier): the rung alone cannot separate 2 participants");
        std::println("    on 2 nodes from 4.");
        std::println("  - fixed-global and fixed-local answer different questions and must never be");
        std::println("    joined by a line.");
        std::println("===============================================================================");
        return written > 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
        return 2;
    }
}
