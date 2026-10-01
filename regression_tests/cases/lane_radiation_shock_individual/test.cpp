#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionBoundaryCalculator.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/monte/deps/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/3D/output/write3D.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{

class RegressionMultigroupDiffusion : public MultigroupDiffusion
{
public:
    using MultigroupDiffusion::MultigroupDiffusion;

    bool supportsAllActiveIndividualGlobalStep() const override
    {
        // Regression-only hook: exercise the reduced solver on a synchronized
        // all-active event without exposing a production configuration knob.
        return std::getenv("RICH_TEST_FORCE_REDUCED_ALL_ACTIVE") == nullptr;
    }
};

size_t EnvironmentSize(char const* name, size_t fallback)
{
    char const* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    return static_cast<size_t>(std::stoull(value));
}

double EnvironmentDouble(char const* name, double fallback)
{
    char const* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    return std::stod(value);
}

std::uint64_t HashWord(std::uint64_t hash, std::uint64_t word)
{
    hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash *= UINT64_C(0x94d049bb133111eb);
    return hash ^ (hash >> 31);
}

std::uint64_t DoubleBits(double value)
{
    std::uint64_t result = 0;
    static_assert(sizeof(result) == sizeof(value),
                  "restart fingerprint assumes 64-bit doubles");
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint64_t HashVector(std::uint64_t hash, Vector3D const& value)
{
    hash = HashWord(hash, DoubleBits(value.x));
    hash = HashWord(hash, DoubleBits(value.y));
    return HashWord(hash, DoubleBits(value.z));
}

std::uint64_t HashTimeState(CellTimeState const& state)
{
    std::uint64_t hash = HashWord(UINT64_C(0x243f6a8885a308d3),
                                  static_cast<std::uint64_t>(state.cell_id));
    hash = HashWord(hash, state.begin_tick);
    hash = HashWord(hash, state.end_tick);
    hash = HashWord(hash, state.last_primitive_tick);
    hash = HashWord(hash, state.time_bin);
    hash = HashVector(hash, state.point_velocity);
    hash = HashVector(hash, state.cached_acceleration);
    return HashWord(hash, state.gravity_half_kick_pending ? 1 : 0);
}

std::uint64_t HashScheduleState(CellTimeState const& state)
{
    std::uint64_t hash = HashWord(UINT64_C(0x452821e638d01377),
                                  static_cast<std::uint64_t>(state.cell_id));
    hash = HashWord(hash, state.begin_tick);
    hash = HashWord(hash, state.end_tick);
    hash = HashWord(hash, state.last_primitive_tick);
    return HashWord(hash, state.time_bin);
}

std::uint64_t HashConserved(std::size_t cell_id, Conserved3D const& state)
{
    std::uint64_t hash = HashWord(UINT64_C(0x13198a2e03707344),
                                  static_cast<std::uint64_t>(cell_id));
    hash = HashWord(hash, DoubleBits(state.mass));
    hash = HashVector(hash, state.momentum);
    hash = HashWord(hash, DoubleBits(state.energy));
    hash = HashWord(hash, DoubleBits(state.internal_energy));
    hash = HashWord(hash, DoubleBits(state.Erad));
    for(double value : state.Eg)
        hash = HashWord(hash, DoubleBits(value));
    hash = HashWord(hash, DoubleBits(state.Erad_dt));
    hash = HashWord(hash, DoubleBits(state.Erad_dt_dt));
    for(double value : state.tracers)
        hash = HashWord(hash, DoubleBits(value));
    return hash;
}

void PrintIndividualRestartFingerprint(Simulation const& simulation,
                                       char const* phase)
{
    IndividualTimeStepScheduler const* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    if(scheduler == nullptr || !scheduler->initialized())
        return;

    std::vector<CellTimeState> const& states = scheduler->states();
    std::vector<ComputationalCell3D> const& cells = simulation.getCells();
    std::vector<Conserved3D> const& extensives = simulation.getExtensives();
    if(states.size() != cells.size() || states.size() != extensives.size())
        throw std::logic_error(
            "individual restart fingerprint arrays have inconsistent sizes");

    unsigned long long state_xor = 0;
    unsigned long long state_sum = 0;
    unsigned long long schedule_xor = 0;
    unsigned long long schedule_sum = 0;
    unsigned long long conserved_xor = 0;
    unsigned long long conserved_sum = 0;
    unsigned long long cell_count = static_cast<unsigned long long>(states.size());
    for(std::size_t index = 0; index < states.size(); ++index)
    {
        std::uint64_t const state_hash = HashTimeState(states[index]);
        std::uint64_t const schedule_hash = HashScheduleState(states[index]);
        std::uint64_t const conserved_hash =
            HashConserved(cells[index].ID, extensives[index]);
        state_xor ^= static_cast<unsigned long long>(state_hash);
        state_sum += static_cast<unsigned long long>(
            HashWord(state_hash, UINT64_C(0xa4093822299f31d0)));
        schedule_xor ^= static_cast<unsigned long long>(schedule_hash);
        schedule_sum += static_cast<unsigned long long>(
            HashWord(schedule_hash, UINT64_C(0x13198a2e03707344)));
        conserved_xor ^= static_cast<unsigned long long>(conserved_hash);
        conserved_sum += static_cast<unsigned long long>(
            HashWord(conserved_hash, UINT64_C(0x082efa98ec4e6c89)));
    }
    unsigned long long next_tick = static_cast<unsigned long long>(
        scheduler->nextEventTick());
    int minimum_force_all_active_latched =
        scheduler->forceAllActiveLatched() ? 1 : 0;
    int maximum_force_all_active_latched = minimum_force_all_active_latched;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &state_xor, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &state_sum, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &schedule_xor, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &schedule_sum, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &conserved_xor, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &conserved_sum, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &cell_count, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &next_tick, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_force_all_active_latched, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_force_all_active_latched, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(minimum_force_all_active_latched != maximum_force_all_active_latched)
        throw std::logic_error(
            "forced-active latch differs across restart fingerprint ranks");
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    RadiationRepairAccounting const& repair =
        scheduler->radiationRepairAccounting();
    IndividualRadiationDefectAccounting const& defect =
        scheduler->radiationDefectAccounting();
    if(rank == 0)
        std::clog << "INDIVIDUAL_RESTART_FINGERPRINT"
                  << " phase=" << phase
                  << " force_synchronized="
                  << (scheduler->options().force_synchronized ? 1 : 0)
                  << " force_all_active_latched="
                  << minimum_force_all_active_latched
                  << " current_tick=" << scheduler->currentTick()
                  << " next_tick=" << next_tick
                  << " time_origin_bits="
                  << DoubleBits(scheduler->timeOrigin())
                  << " time_quantum_bits="
                  << DoubleBits(scheduler->timeQuantum())
                  << " cells=" << cell_count
                  << " state_xor=" << state_xor
                  << " state_sum=" << state_sum
                  << " schedule_xor=" << schedule_xor
                  << " schedule_sum=" << schedule_sum
                  << " conserved_xor=" << conserved_xor
                  << " conserved_sum=" << conserved_sum
                  << " repair_cells=" << repair.repaired_cells
                  << " repair_groups=" << repair.repaired_groups
                  << " repair_injected_bits="
                  << DoubleBits(repair.cumulative_injected_energy)
                  << " repair_max_deficit_bits="
                  << DoubleBits(repair.maximum_relative_deficit)
                  << " repair_rep_cell=" << repair.representative_cell_id
                  << " repair_rep_group=" << repair.representative_group
                  << " repair_rep_rank=" << repair.representative_rank
                  << " repair_rep_original_bits="
                  << DoubleBits(repair.representative_original_extent)
                  << " repair_rep_floor_bits="
                  << DoubleBits(repair.representative_floor_extent)
                  << " repair_rep_injected_bits="
                  << DoubleBits(repair.representative_injected_extent)
                  << " repair_max_global_energy_bits="
                  << DoubleBits(repair.maximum_global_radiation_energy)
                  << " repair_next_warning_bits="
                  << DoubleBits(repair.next_warning_fraction)
                  << std::setprecision(
                         std::numeric_limits<long double>::max_digits10)
                  << " defect_cumulative_signed_extent="
                  << defect.cumulative_signed_extent
                  << " defect_cumulative_absolute_extent="
                  << defect.cumulative_absolute_extent
                  << " defect_initial_positive_global_extent="
                  << defect.initial_positive_global_extent
                  << " defect_last_normalization_scale="
                  << defect.last_normalization_scale
                  << " defect_maximum_event_absolute_fraction="
                  << defect.maximum_event_absolute_fraction
                  << " defect_maximum_local_fraction="
                  << defect.maximum_local_fraction
                  << " defect_maximum_local_tolerance_ratio="
                  << defect.maximum_local_tolerance_ratio
                  << " defect_accepted_dirichlet_candidates="
                  << defect.accepted_dirichlet_candidates
                  << " defect_rejections=" << defect.defect_rejections
                  << " defect_retry_substeps="
                  << defect.defect_retry_substeps
                  << " defect_config_version=" << defect.config_version
                  << " defect_local_withdrawal_limit="
                  << defect.local_withdrawal_limit
                  << " defect_local_absolute_limit="
                  << defect.local_absolute_limit
                  << " defect_event_absolute_target="
                  << defect.event_absolute_target
                  << " defect_cumulative_signed_limit="
                  << defect.cumulative_signed_limit
                  << " defect_cumulative_absolute_limit="
                  << defect.cumulative_absolute_limit
                  << " defect_cooldown_accepted_candidates="
                  << defect.cooldown_accepted_candidates
                  << " defect_cooldown_required_candidates="
                  << defect.cooldown_required_candidates
                  << " defect_cooldown_fraction_ceiling="
                  << defect.cooldown_fraction_ceiling
                  << " defect_history_complete="
                  << (defect.history_complete ? 1 : 0) << '\n';
}

struct IndividualTerminalTarget
{
    std::uint64_t tick = 0;
    double time_origin = 0;
    double represented_time = 0;
    double time_quantum = 0;
};

bool GetIndividualTerminalTarget(Simulation& simulation, double final_time,
                                 IndividualTerminalTarget& target)
{
    IndividualTimeStepScheduler* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    if(scheduler == nullptr || !scheduler->initialized())
        return false;

    long double const relative_tick =
        (static_cast<long double>(final_time) - scheduler->timeOrigin()) /
        scheduler->timeQuantum();
    if(!(std::isfinite(relative_tick) && relative_tick >= 0) ||
       relative_tick > static_cast<long double>(
           std::numeric_limits<std::uint64_t>::max()))
        throw std::runtime_error(
            "requested final time is outside the individual tick timeline");
    target.tick = static_cast<std::uint64_t>(
        std::round(relative_tick));
    target.time_origin = scheduler->timeOrigin();
    target.time_quantum = scheduler->timeQuantum();
    target.represented_time = target.time_origin +
        target.time_quantum * static_cast<double>(target.tick);
    return true;
}

bool PrepareIndividualTerminalEvent(
    Simulation& simulation, IndividualTerminalTarget const& target)
{
    IndividualTimeStepScheduler* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    if(scheduler == nullptr || !scheduler->initialized())
        throw std::logic_error(
            "individual terminal event requires an initialized scheduler");
    if(target.tick <= scheduler->currentTick())
        return false;

    unsigned long long clamped = static_cast<unsigned long long>(
        scheduler->clampToTerminalTick(target.tick));
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &clamped, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    if(clamped > 0)
    {
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        if(rank == 0)
            std::clog << "INDIVIDUAL_TERMINAL_EVENT_CLAMP"
                      << " terminal_tick=" << target.tick
                      << " represented_time=" << std::setprecision(17)
                      << target.represented_time
                      << " clamped_cells=" << clamped << std::endl;
    }
    return true;
}

struct GravityPolicyBaseline
{
    bool enabled = false;
    bool local_rank_active = false;
    std::unordered_map<std::size_t, double> mass_by_id;
};

GravityPolicyBaseline CaptureGravityPolicyBaseline(
    Simulation const& simulation)
{
    GravityPolicyBaseline baseline;
    IndividualTimeStepScheduler const* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    if(scheduler == nullptr || !scheduler->initialized())
        return baseline;

    std::vector<CellTimeState> const& states = scheduler->states();
    std::vector<ComputationalCell3D> const& cells = simulation.getCells();
    std::vector<Conserved3D> const& extensives = simulation.getExtensives();
    if(states.size() != cells.size() || cells.size() != extensives.size())
        throw std::logic_error(
            "Gravity baseline scheduler/state arrays have inconsistent sizes");

    unsigned long long next_tick = std::numeric_limits<unsigned long long>::max();
    for(CellTimeState const& state : states)
        next_tick = std::min(next_tick,
            static_cast<unsigned long long>(state.end_tick));
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &next_tick, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    for(CellTimeState const& state : states)
        baseline.local_rank_active = baseline.local_rank_active ||
            state.end_tick == next_tick;
    for(std::size_t i = 0; i < cells.size(); ++i)
        baseline.mass_by_id.emplace(cells[i].ID, extensives[i].mass);
    baseline.enabled = true;
    return baseline;
}

void PrintGravityPolicyDiagnostic(
    Simulation const& simulation,
    GravityPolicyBaseline const& baseline)
{
    IndividualTimeStepScheduler const* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    if(scheduler == nullptr || !scheduler->initialized())
        return;

    std::vector<CellTimeState> const& states = scheduler->states();
    unsigned long long reference_id = std::numeric_limits<unsigned long long>::max();
    unsigned long long active_cells = 0;
    unsigned long long active_ranks = baseline.local_rank_active ? 1 : 0;
    unsigned long long passive_mass_updates = 0;
    double minimum_speed = std::numeric_limits<double>::infinity();
    double maximum_speed = 0;
    for(CellTimeState const& state : states)
    {
        double const speed = std::sqrt(state.point_velocity.x * state.point_velocity.x +
            state.point_velocity.y * state.point_velocity.y +
            state.point_velocity.z * state.point_velocity.z);
        minimum_speed = std::min(minimum_speed, speed);
        maximum_speed = std::max(maximum_speed, speed);
        if(state.begin_tick == scheduler->currentTick())
        {
            ++active_cells;
            reference_id = std::min(reference_id,
                static_cast<unsigned long long>(state.cell_id));
        }
    }
    if(baseline.enabled && !baseline.local_rank_active)
    {
        std::vector<ComputationalCell3D> const& cells = simulation.getCells();
        std::vector<Conserved3D> const& extensives = simulation.getExtensives();
        if(cells.size() != extensives.size())
            throw std::logic_error(
                "Gravity diagnostic primitive/conserved sizes differ");
        for(std::size_t i = 0; i < cells.size(); ++i)
        {
            auto const found = baseline.mass_by_id.find(cells[i].ID);
            if(found == baseline.mass_by_id.end())
                continue;
            double const scale = std::max(
                std::abs(found->second), std::numeric_limits<double>::min());
            if(std::abs(extensives[i].mass - found->second) >
               64 * std::numeric_limits<double>::epsilon() * scale)
                ++passive_mass_updates;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &reference_id, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &active_cells, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &active_ranks, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &passive_mass_updates, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_speed, 1, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_speed, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    if(active_cells == 0 ||
       reference_id == std::numeric_limits<unsigned long long>::max())
        throw std::logic_error("Individual gravity diagnostic found no active cell");

    double acceleration[3] = {0, 0, 0};
    double conserved_energy = 0;
    int owners = 0;
    std::vector<Conserved3D> const& extensives = simulation.getExtensives();
    if(extensives.size() != states.size())
        throw std::logic_error(
            "Gravity diagnostic scheduler/conserved sizes differ");
    for(std::size_t i = 0; i < states.size(); ++i)
        if(static_cast<unsigned long long>(states[i].cell_id) == reference_id)
        {
            acceleration[0] = states[i].cached_acceleration.x;
            acceleration[1] = states[i].cached_acceleration.y;
            acceleration[2] = states[i].cached_acceleration.z;
            conserved_energy = extensives[i].energy;
            owners = 1;
            break;
        }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, acceleration, 3, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &conserved_energy, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    if(owners != 1)
        throw std::logic_error(
            "Individual gravity diagnostic reference ID is not uniquely owned");

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if(rank == 0)
        std::clog << std::setprecision(17)
                  << "GRAVITY_POLICY_PARITY cell_id=" << reference_id
                  << " active_cells=" << active_cells
                  << " active_ranks=" << active_ranks
                  << " passive_mass_updates=" << passive_mass_updates
                  << " ax=" << acceleration[0]
                  << " ay=" << acceleration[1]
                  << " az=" << acceleration[2]
                  << " energy=" << conserved_energy
                  << " velocity_span=" << maximum_speed - minimum_speed
                  << " position_policy=predicted_centroid_active_refresh\n";
}

CG::PreconditionerKind EnvironmentMGPreconditioner()
{
    char const* value = std::getenv("RICH_TEST_MG_PRECONDITIONER");
    if(value == nullptr || value[0] == '\0' ||
       std::string(value) == "cell_block")
        return CG::PreconditionerKind::CellBlockJacobi;
    if(std::string(value) == "cell_block_gs")
        return CG::PreconditionerKind::CellBlockGaussSeidel;
    if(std::string(value) == "rank_local_ilu0")
        return CG::PreconditionerKind::RankLocalILU0;
    if(std::string(value) == "cell_block_two_sweep")
        return CG::PreconditionerKind::CellBlockJacobiTwoSweep;
    if(std::string(value) == "cell_block_four_sweep")
        return CG::PreconditionerKind::CellBlockJacobiFourSweep;
    if(std::string(value) == "cell_block_eight_sweep")
        return CG::PreconditionerKind::CellBlockJacobiEightSweep;
    if(std::string(value) == "scalar")
        return CG::PreconditionerKind::ScalarJacobi;
    throw std::invalid_argument(
        "RICH_TEST_MG_PRECONDITIONER must be 'scalar', 'cell_block', "
        "'cell_block_gs', 'rank_local_ilu0', or "
        "'cell_block_two_sweep', 'cell_block_four_sweep', or "
        "'cell_block_eight_sweep'");
}

struct RadiationGrid
{
    std::vector<double> centers;
    std::vector<double> boundaries;
};

RadiationGrid MakeRadiationGrid(void)
{
    std::size_t const groups = ENERGY_GROUPS_NUM;
    double constexpr electron_volt = 1.602176634e-12;
    double constexpr minimum_energy = electron_volt;
    double constexpr maximum_energy = 2.0e6 * electron_volt;
    RadiationGrid grid;
    grid.centers.resize(groups);
    grid.boundaries.resize(groups + 1);
    double const ratio = std::pow(maximum_energy / minimum_energy,
        1.0 / static_cast<double>(groups));
    grid.boundaries[0] = minimum_energy;
    for(std::size_t group = 0; group < groups; ++group)
    {
        grid.boundaries[group + 1] = grid.boundaries[group] * ratio;
        grid.centers[group] = std::sqrt(
            grid.boundaries[group] * grid.boundaries[group + 1]);
        ComputationalCell3D::energyBoundaries[group] =
            grid.boundaries[group];
    }
    ComputationalCell3D::energyBoundaries[groups] =
        grid.boundaries[groups];
    return grid;
}

struct LaneEmdenProfile
{
    vector<double> xsi, theta;
    double n, alpha, rho_c, K;

    LaneEmdenProfile(double M, double R, double G)
    {
        xsi = read_vector("../../../data/xsi32.txt");
        theta = read_vector("../../../data/theta32.txt");
        xsi[0] = 0;
        n = 1.5;
        double endfactor = 2.714;
        alpha = R / xsi.back();
        rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
        K = G * alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));
    }

    double densityAt(double r, double R) const
    {
        if (r < R)
        {
            double t = LinearInterpolation(xsi, theta, r / alpha);
            return std::max(rho_c * std::pow(t, n), 1e-5);
        }
        else
        {
            double t = theta.back();
            return rho_c * std::pow(t, n);
        }
    }
};

std::vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double R,
                                          IdealGas const &eos,
                                          LaneEmdenProfile const &prof,
                                          RadiationGrid const& radiation_grid,
                                          double temperature_floor)
{
    size_t N = tess.GetPointNo();
    std::vector<ComputationalCell3D> res(N);

    for (size_t i = 0; i < N; ++i)
    {
        Vector3D const &point = tess.GetMeshPoint(i);
        double r = abs(point);
        res[i].density = prof.densityAt(r, R);
        double const P = prof.K * std::pow(res[i].density, 1 + 1.0 / prof.n);
        res[i].pressure = P;
        res[i].internal_energy = eos.dp2e(res[i].density, P, res[i].tracers,
                                          ComputationalCell3D::tracerNames);
        res[i].temperature = eos.de2T(res[i].density, res[i].internal_energy,
                                     res[i].tracers,
                                     ComputationalCell3D::tracerNames);
        if(res[i].temperature < temperature_floor)
        {
            res[i].temperature = temperature_floor;
            res[i].internal_energy = eos.dT2e(res[i].density,
                res[i].temperature, res[i].tracers,
                ComputationalCell3D::tracerNames);
            res[i].pressure = eos.de2p(res[i].density,
                res[i].internal_energy, res[i].tracers,
                ComputationalCell3D::tracerNames);
        }
        res[i].Erad = CG::radiation_constant * std::pow(res[i].temperature, 4) /
                      res[i].density;
        for(std::size_t group = 0; group < radiation_grid.centers.size(); ++group)
        {
            double const group_energy_density =
                planck_integral::planck_energy_density_group_integral(
                    radiation_grid.boundaries[group],
                    radiation_grid.boundaries[group + 1], res[i].temperature);
            res[i].Eg[group] = std::max(group_energy_density / res[i].density,
                                        res[i].Erad * 1e-16);
        }
    }

    return res;
}

double AddRadiationPulse(Tessellation3D const& tess,
                         std::vector<ComputationalCell3D>& cells,
                         RadiationGrid const& radiation_grid,
                         double radius,
                         double peak_temperature)
{
    double local_pulse_energy = 0;
    for(std::size_t index = 0; index < tess.GetPointNo(); ++index)
    {
        double const normalized_radius = abs(tess.GetCellCM(index)) / radius;
        if(normalized_radius < 1)
        {
            double const factor = 1 - normalized_radius * normalized_radius;
            double const added_energy_density = CG::radiation_constant *
                std::pow(peak_temperature, 4) * factor * factor;
            local_pulse_energy += added_energy_density * tess.GetVolume(index);
        }
    }
    double pulse_energy = local_pulse_energy;
#ifdef RICH_MPI
    MPI_Allreduce(&local_pulse_energy, &pulse_energy, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    if(!(pulse_energy > 0) || !(peak_temperature > 0))
        throw std::runtime_error("Radiation pulse normalization is not positive");
    for(std::size_t index = 0; index < tess.GetPointNo(); ++index)
    {
        double const normalized_radius = abs(tess.GetCellCM(index)) / radius;
        if(normalized_radius >= 1)
            continue;
        double const factor = 1 - normalized_radius * normalized_radius;
        double const added_energy_density = CG::radiation_constant *
            std::pow(peak_temperature, 4) * factor * factor;
        std::vector<double> weights(radiation_grid.centers.size(), 0);
        double weight_sum = 0;
        for(std::size_t group = 0; group < weights.size(); ++group)
        {
            weights[group] =
                planck_integral::planck_energy_density_group_integral(
                    radiation_grid.boundaries[group],
                    radiation_grid.boundaries[group + 1], peak_temperature);
            weight_sum += weights[group];
        }
        if(!(weight_sum > 0))
            throw std::runtime_error("Radiation pulse group normalization failed");
        for(std::size_t group = 0; group < weights.size(); ++group)
            cells[index].Eg[group] += added_energy_density *
                weights[group] / (weight_sum * cells[index].density);
        cells[index].Erad += added_energy_density / cells[index].density;
    }
    return pulse_energy;
}

double RadiationEnergyDensity(ComputationalCell3D const& cell)
{
    return cell.density * std::accumulate(cell.Eg.begin(), cell.Eg.end(), 0.0);
}

struct RefinementIndicators
{
    double pressure_jump = 0;
    double radiation_jump = 0;
    double compression = 0;
};

RefinementIndicators GetRefinementIndicators(
    size_t index,
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    EquationOfState const& eos)
{
    RefinementIndicators result;
    ComputationalCell3D const& cell = cells.at(index);
    double const radiation = RadiationEnergyDensity(cell);
    double const sound = eos.dp2c(cell.density, cell.pressure, cell.tracers,
        ComputationalCell3D::tracerNames);
    Vector3D const point = tess.GetMeshPoint(index);
    std::vector<size_t> const neighbors = tess.GetNeighbors(index);
    for(size_t neighbor : neighbors)
    {
        if(neighbor >= cells.size())
            continue;
        ComputationalCell3D const& other = cells[neighbor];
        double const pressure_scale = std::max(
            std::max(std::abs(cell.pressure), std::abs(other.pressure)),
            std::numeric_limits<double>::min());
        result.pressure_jump = std::max(result.pressure_jump,
            std::abs(cell.pressure - other.pressure) / pressure_scale);

        double const other_radiation = RadiationEnergyDensity(other);
        double const radiation_scale = std::max(
            std::max(std::abs(radiation), std::abs(other_radiation)),
            std::numeric_limits<double>::min());
        result.radiation_jump = std::max(result.radiation_jump,
            std::abs(radiation - other_radiation) / radiation_scale);

        Vector3D const separation = tess.GetMeshPoint(neighbor) - point;
        double const distance = abs(separation);
        if(distance <= 0)
            continue;
        Vector3D const velocity_difference = other.velocity - cell.velocity;
        double const projected_difference =
            (velocity_difference.x * separation.x +
             velocity_difference.y * separation.y +
             velocity_difference.z * separation.z) / distance;
        double const other_sound = eos.dp2c(other.density, other.pressure,
            other.tracers, ComputationalCell3D::tracerNames);
        double const signal_speed = std::max(sound + other_sound,
            std::numeric_limits<double>::min());
        result.compression = std::max(result.compression,
            std::max(0.0, -projected_difference / signal_speed));
    }
    return result;
}

std::vector<size_t> LimitGlobalRefinementCandidates(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    std::vector<size_t> candidates,
    size_t maximum_cells,
    size_t maximum_batch)
{
    unsigned long long local_cell_count = static_cast<unsigned long long>(
        tess.GetPointNo());
    unsigned long long global_cell_count = local_cell_count;
#ifdef RICH_MPI
    MPI_Allreduce(&local_cell_count, &global_cell_count, 1,
        MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(global_cell_count >= maximum_cells)
        return std::vector<size_t>();
    size_t const remaining = static_cast<size_t>(maximum_cells -
        global_cell_count);
    size_t const limit = std::min(remaining, maximum_batch);
    if(limit == 0)
        return std::vector<size_t>();
#ifndef RICH_MPI
    if(candidates.empty())
        return std::vector<size_t>();
#endif

    std::vector<unsigned long long> local_ids;
    local_ids.reserve(candidates.size());
    for(size_t index : candidates)
        local_ids.push_back(static_cast<unsigned long long>(cells.at(index).ID));
    std::vector<unsigned long long> all_ids = local_ids;
#ifdef RICH_MPI
    int const local_count = static_cast<int>(local_ids.size());
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::vector<int> counts(static_cast<size_t>(world_size), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD);
    std::vector<int> displacements(counts.size(), 0);
    for(size_t i = 1; i < counts.size(); ++i)
        displacements[i] = displacements[i - 1] + counts[i - 1];
    all_ids.resize(static_cast<size_t>(displacements.back() + counts.back()));
    MPI_Allgatherv(local_ids.data(), local_count, MPI_UNSIGNED_LONG_LONG,
        all_ids.data(), counts.data(), displacements.data(),
        MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
#endif
    std::sort(all_ids.begin(), all_ids.end());
    all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());
    if(all_ids.size() > limit)
        all_ids.resize(limit);
    std::unordered_set<unsigned long long> selected(all_ids.begin(),
        all_ids.end());
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](size_t index)
        {
            return selected.count(static_cast<unsigned long long>(
                cells.at(index).ID)) == 0;
        }), candidates.end());
    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

unsigned long long GlobalOwnedCellCount(Tessellation3D const& tess)
{
    unsigned long long result = static_cast<unsigned long long>(
        tess.GetPointNo());
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
        MPI_COMM_WORLD);
#endif
    return result;
}

std::vector<size_t> SelectGlobalCandidateCount(
    std::vector<ComputationalCell3D> const& cells,
    std::vector<size_t> candidates,
    size_t limit)
{
    if(limit == 0)
        return std::vector<size_t>();
#ifndef RICH_MPI
    if(candidates.empty())
        return std::vector<size_t>();
#endif
    std::vector<unsigned long long> local_ids;
    local_ids.reserve(candidates.size());
    for(size_t index : candidates)
        local_ids.push_back(static_cast<unsigned long long>(cells.at(index).ID));
    std::vector<unsigned long long> all_ids = local_ids;
#ifdef RICH_MPI
    int const local_count = static_cast<int>(local_ids.size());
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::vector<int> counts(static_cast<size_t>(world_size), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
        MPI_COMM_WORLD);
    std::vector<int> displacements(counts.size(), 0);
    for(size_t i = 1; i < counts.size(); ++i)
        displacements[i] = displacements[i - 1] + counts[i - 1];
    all_ids.resize(static_cast<size_t>(displacements.back() + counts.back()));
    MPI_Allgatherv(local_ids.data(), local_count, MPI_UNSIGNED_LONG_LONG,
        all_ids.data(), counts.data(), displacements.data(),
        MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
#endif
    std::sort(all_ids.begin(), all_ids.end());
    all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());
    if(all_ids.size() > limit)
        all_ids.resize(limit);
    std::unordered_set<unsigned long long> selected(all_ids.begin(),
        all_ids.end());
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [&](size_t index)
        {
            return selected.count(static_cast<unsigned long long>(
                cells.at(index).ID)) == 0;
        }), candidates.end());
    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

class RadiationShockRefine3D : public CellsToRefine3D
{
public:
    RadiationShockRefine3D(EquationOfState const& eos, double radius,
        double target_width, size_t maximum_cells, size_t maximum_batch,
        bool prescribed, double prescribed_time, size_t initial_cells,
        size_t prescribed_count):
        eos_(eos), radius_(radius), target_width_(target_width),
        maximum_cells_(maximum_cells), maximum_batch_(maximum_batch),
        prescribed_(prescribed), prescribed_time_(prescribed_time),
        initial_cells_(initial_cells), prescribed_count_(prescribed_count) {}

    std::pair<std::vector<size_t>, std::vector<Vector3D> > ToRefine(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double time) const override
    {
        std::vector<size_t> candidates;
        if(prescribed_)
        {
            unsigned long long const global_cells = GlobalOwnedCellCount(tess);
            unsigned long long const target = static_cast<unsigned long long>(
                initial_cells_ + prescribed_count_);
            if(time < prescribed_time_ || global_cells >= target)
                return std::make_pair(candidates, std::vector<Vector3D>());
            candidates.resize(tess.GetPointNo());
            std::iota(candidates.begin(), candidates.end(), 0);
            candidates = SelectGlobalCandidateCount(cells,
                std::move(candidates), static_cast<size_t>(target - global_cells));
            return std::make_pair(candidates, std::vector<Vector3D>());
        }
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            if(abs(tess.GetMeshPoint(i)) > 1.5 * radius_ ||
               tess.GetWidth(i) <= target_width_)
                continue;
            RefinementIndicators const indicator = GetRefinementIndicators(
                i, tess, cells, eos_);
            if(indicator.pressure_jump >= 0.20 ||
               indicator.radiation_jump >= 0.20 ||
               indicator.compression >= 0.25)
                candidates.push_back(i);
        }
        candidates = LimitGlobalRefinementCandidates(tess, cells,
            std::move(candidates), maximum_cells_, maximum_batch_);
        return std::make_pair(candidates, std::vector<Vector3D>());
    }

private:
    EquationOfState const& eos_;
    double radius_;
    double target_width_;
    size_t maximum_cells_;
    size_t maximum_batch_;
    bool prescribed_;
    double prescribed_time_;
    size_t initial_cells_;
    size_t prescribed_count_;
};

class RadiationShockRemove3D : public CellsToRemove3D
{
public:
    RadiationShockRemove3D(EquationOfState const& eos, double radius,
        double minimum_age, size_t initial_maximum_id, bool prescribed,
        double prescribed_time, size_t prescribed_final_cells):
        eos_(eos), radius_(radius), minimum_age_(minimum_age),
        initial_maximum_id_(initial_maximum_id), prescribed_(prescribed),
        prescribed_time_(prescribed_time),
        prescribed_final_cells_(prescribed_final_cells) {}

    std::pair<std::vector<size_t>, std::vector<double> > ToRemove(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double time) const override
    {
        std::vector<size_t> candidates;
        std::vector<double> merits;
        if(prescribed_)
        {
            unsigned long long const global_cells = GlobalOwnedCellCount(tess);
            if(time < prescribed_time_ ||
               global_cells <= prescribed_final_cells_)
                return std::make_pair(candidates, merits);
            for(size_t i = 0; i < tess.GetPointNo(); ++i)
                if(cells.at(i).ID > initial_maximum_id_)
                    candidates.push_back(i);
            candidates = SelectGlobalCandidateCount(cells,
                std::move(candidates), static_cast<size_t>(global_cells -
                    prescribed_final_cells_));
            merits.assign(candidates.size(), 1.0);
            return std::make_pair(candidates, merits);
        }
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            size_t const id = cells.at(i).ID;
            if(id <= initial_maximum_id_)
                continue;
            auto const inserted = birth_times_.emplace(id, time);
            if(time - inserted.first->second < minimum_age_ ||
               abs(tess.GetMeshPoint(i)) > 1.5 * radius_)
                continue;
            RefinementIndicators const indicator = GetRefinementIndicators(
                i, tess, cells, eos_);
            if(indicator.pressure_jump <= 0.06 &&
               indicator.radiation_jump <= 0.06 &&
               indicator.compression <= 0.08)
            {
                candidates.push_back(i);
                merits.push_back(1.0 / (1.0 + indicator.pressure_jump +
                    indicator.radiation_jump + indicator.compression));
            }
        }
        return std::make_pair(candidates, merits);
    }

private:
    EquationOfState const& eos_;
    double radius_;
    double minimum_age_;
    size_t initial_maximum_id_;
    bool prescribed_;
    double prescribed_time_;
    size_t prescribed_final_cells_;
    mutable std::unordered_map<size_t, double> birth_times_;
};

struct AMRRunCounters
{
    unsigned long long calls = 0;
    unsigned long long refined = 0;
    unsigned long long derefined = 0;
};

class ControlledAMR
{
public:
    ControlledAMR(Simulation& simulation, AMR3D& amr, double interval):
        simulation_(simulation), amr_(amr), interval_(interval), next_time_(0) {}

    IndividualAMRChangeSet ApplyIndividual(
        IndividualStepContext const& context)
    {
        if(!Due(context.event_time))
            return IndividualAMRChangeSet();
        IndividualAMRChangeSet changes = amr_.ApplyIndividual(simulation_,
            context);
        Record(changes);
        return changes;
    }

    bool ApplyGlobal(double time)
    {
        if(!Due(time))
            return false;
        amr_(simulation_);
        Record(amr_.GetLastChangeSet());
        return true;
    }

    AMRRunCounters const& Counters(void) const {return counters_;}

    double NextTime(void) const {return next_time_;}

    void RestoreCheckpointState(double next_time,
                                AMRRunCounters const& counters)
    {
        if(!std::isfinite(next_time) || next_time < 0)
            throw std::runtime_error(
                "invalid controlled-AMR checkpoint time");
        next_time_ = next_time;
        counters_ = counters;
    }

private:
    bool Due(double time)
    {
        if(interval_ <= 0 || time + 1e-14 * std::max(1.0, time) < next_time_)
            return false;
        do
            next_time_ += interval_;
        while(next_time_ <= time);
        return true;
    }

    void Record(IndividualAMRChangeSet const& changes)
    {
        ++counters_.calls;
        counters_.refined += static_cast<unsigned long long>(
            changes.child_parent_ids.size());
        counters_.derefined += static_cast<unsigned long long>(
            changes.removed_cell_ids.size());
    }

    Simulation& simulation_;
    AMR3D& amr_;
    double interval_;
    double next_time_;
    AMRRunCounters counters_;
};

double WallSeconds(void)
{
#ifdef RICH_MPI
    return MPI_Wtime();
#else
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
#endif
}

void WriteBenchmarkVTK(Simulation const& simulation,
                       std::string const& filename)
{
    Tessellation3D const& tess = simulation.getTessellation();
    std::vector<ComputationalCell3D> const& cells = simulation.getCells();
    size_t const cell_count = tess.GetPointNo();
    if(cells.size() < cell_count)
        throw std::logic_error(
            "VTK snapshot has fewer primitive cells than owned mesh cells");

    std::vector<std::string> scalar_names;
    std::vector<std::vector<double>> scalar_fields;
    auto append_scalar = [&](std::string const& name, auto const& value)
    {
        std::vector<double> field(cell_count);
        for(size_t i = 0; i < cell_count; ++i)
            field[i] = value(i);
        scalar_names.push_back(name);
        scalar_fields.push_back(std::move(field));
    };

    append_scalar("Density", [&](size_t i){return cells[i].density;});
    append_scalar("Pressure", [&](size_t i){return cells[i].pressure;});
    append_scalar("InternalEnergy",
        [&](size_t i){return cells[i].internal_energy;});
    append_scalar("Temperature", [&](size_t i){return cells[i].temperature;});
    append_scalar("ID",
        [&](size_t i){return static_cast<double>(cells[i].ID);});
    append_scalar("Erad", [&](size_t i){return cells[i].Erad;});
    for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
    {
        append_scalar("Eg_" + std::to_string(group),
            [&](size_t i)
            {
                if(cells[i].Eg.size() <= group)
                    throw std::logic_error(
                        "VTK snapshot cell has fewer radiation groups than configured");
                return cells[i].Eg[group];
            });
    }
    for(size_t tracer = 0; tracer < ComputationalCell3D::tracerNames.size();
        ++tracer)
    {
        append_scalar(ComputationalCell3D::tracerNames[tracer],
            [&](size_t i){return cells[i].tracers.at(tracer);});
    }
    for(size_t sticker = 0; sticker < ComputationalCell3D::stickerNames.size();
        ++sticker)
    {
        append_scalar(ComputationalCell3D::stickerNames[sticker],
            [&](size_t i)
            {
                return cells[i].stickers.at(sticker) ? 1.0 : 0.0;
            });
    }
    append_scalar("Volume", [&](size_t i){return tess.GetVolume(i);});

    std::vector<std::string> vector_names{"GeneratorPosition", "Velocity"};
    std::vector<std::vector<Vector3D>> vector_fields(2,
        std::vector<Vector3D>(cell_count));
    for(size_t i = 0; i < cell_count; ++i)
    {
        vector_fields[0][i] = tess.GetMeshPoint(i);
        vector_fields[1][i] = cells[i].velocity;
    }

    std::filesystem::path vtk_path(filename);
    vtk_path.replace_extension("vtu");
    write_vtu3d::write_vtu_3d(vtk_path, scalar_names, scalar_fields,
        vector_names, vector_fields, simulation.GetTime(),
        simulation.GetCycle(), tess);
}

class NullStreamBuffer : public std::streambuf
{
protected:
    int overflow(int character) override {return character;}
};

std::uint64_t MixHash(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void HashCombine(std::uint64_t& seed, double value)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
    std::memcpy(&bits, &value, sizeof(bits));
    seed = MixHash(seed ^ MixHash(bits));
}

std::array<std::uint64_t, 2> InitialStateChecksum(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells)
{
    std::uint64_t local_xor = 0;
    std::uint64_t local_sum = 0;
    for(size_t i = 0; i < tess.GetPointNo(); ++i)
    {
        ComputationalCell3D const& cell = cells.at(i);
        Vector3D const point = tess.GetMeshPoint(i);
        std::uint64_t hash = MixHash(static_cast<std::uint64_t>(cell.ID));
        HashCombine(hash, point.x);
        HashCombine(hash, point.y);
        HashCombine(hash, point.z);
        HashCombine(hash, cell.density);
        HashCombine(hash, cell.pressure);
        HashCombine(hash, cell.internal_energy);
        HashCombine(hash, cell.velocity.x);
        HashCombine(hash, cell.velocity.y);
        HashCombine(hash, cell.velocity.z);
        HashCombine(hash, cell.Erad);
        for(double energy : cell.Eg)
            HashCombine(hash, energy);
        local_xor ^= hash;
        local_sum += hash;
    }
    std::array<std::uint64_t, 2> result{{local_xor, local_sum}};
#ifdef RICH_MPI
    unsigned long long local_values[2] = {
        static_cast<unsigned long long>(local_xor),
        static_cast<unsigned long long>(local_sum)};
    unsigned long long xor_value = 0;
    unsigned long long sum_value = 0;
    MPI_Allreduce(&local_values[0], &xor_value, 1, MPI_UNSIGNED_LONG_LONG,
        MPI_BXOR, MPI_COMM_WORLD);
    MPI_Allreduce(&local_values[1], &sum_value, 1, MPI_UNSIGNED_LONG_LONG,
        MPI_SUM, MPI_COMM_WORLD);
    result[0] = static_cast<std::uint64_t>(xor_value);
    result[1] = static_cast<std::uint64_t>(sum_value);
#endif
    return result;
}

struct EventWorkCounters
{
    unsigned long long events = 0;
    unsigned long long active_cell_updates = 0;
    unsigned long long owned_cell_events = 0;
    unsigned long long active_faces = 0;
    unsigned long long mesh_closure_cells = 0;
    unsigned long long gravity_targets = 0;
    unsigned long long multigroup_rows = 0;
    unsigned long long partial_mesh_events = 0;
    unsigned long long reduced_gravity_events = 0;
    unsigned long long reduced_multigroup_events = 0;
    unsigned int minimum_bin = std::numeric_limits<unsigned int>::max();
    unsigned int maximum_bin = 0;
    unsigned long long time_bin_mask = 0;

    size_t Record(Voronoi3D const& tess, Simulation const& simulation,
        size_t groups)
    {
        size_t const owned = simulation.getExtensives().size();
        size_t active = owned;
        std::vector<unsigned char> active_local(tess.GetPointNo(), 0);
        IndividualTimeStepScheduler const* scheduler =
            simulation.GetIndividualTimeStepScheduler();
        if(scheduler != nullptr && scheduler->initialized())
        {
            active = 0;
            std::vector<CellTimeState> const& states = scheduler->states();
            auto const& local_to_global = tess.GetIndicesInAllPoints();
            std::vector<size_t> global_to_local(states.size(),
                std::numeric_limits<size_t>::max());
            for(auto const& mapping : local_to_global)
                if(mapping.first < active_local.size() &&
                   mapping.second < global_to_local.size())
                    global_to_local[mapping.second] = mapping.first;
            for(size_t global = 0; global < states.size(); ++global)
            {
                CellTimeState const& state = states[global];
                minimum_bin = std::min(minimum_bin,
                    static_cast<unsigned int>(state.time_bin));
                maximum_bin = std::max(maximum_bin,
                    static_cast<unsigned int>(state.time_bin));
                if(state.time_bin < 64)
                    time_bin_mask |= 1ULL << state.time_bin;
                if(state.begin_tick != scheduler->currentTick())
                    continue;
                ++active;
                size_t const local = global_to_local[global];
                if(local < active_local.size())
                    active_local[local] = 1;
            }
        }
        else
        {
            minimum_bin = 0;
            maximum_bin = 0;
            time_bin_mask = 1;
            std::fill(active_local.begin(), active_local.end(), 1);
        }

        size_t event_active_faces = 0;
        for(size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
        {
            std::pair<size_t, size_t> const neighbors =
                tess.GetFaceNeighbors(face);
            bool const left_active = neighbors.first < active_local.size() &&
                active_local[neighbors.first] != 0;
            bool const right_active = neighbors.second < active_local.size() &&
                active_local[neighbors.second] != 0;
            if(left_active || right_active)
                ++event_active_faces;
        }

        ++events;
        active_cell_updates += static_cast<unsigned long long>(active);
        owned_cell_events += static_cast<unsigned long long>(owned);
        active_faces += static_cast<unsigned long long>(event_active_faces);
        mesh_closure_cells += static_cast<unsigned long long>(tess.GetPointNo());
        gravity_targets += static_cast<unsigned long long>(active);
        multigroup_rows += static_cast<unsigned long long>(active * groups);
        if(tess.GetPointNo() < owned)
            ++partial_mesh_events;
        if(active < owned)
        {
            ++reduced_gravity_events;
            ++reduced_multigroup_events;
        }
        return active;
    }
};

struct HistoryRecord
{
    double time = 0;
    unsigned long long cycle = 0;
    unsigned long long cells = 0;
    double active_fraction = 1;
    double mass = 0;
    Vector3D momentum;
    double material_energy = 0;
    double radiation_energy = 0;
    double nongrav_energy = 0;
    Vector3D mass_moment;
    Vector3D center_of_mass;
    double peak_mach = 0;
    double peak_outward_mach = 0;
    double maximum_compression = 0;
    double shock_radius = 0;
};

struct BenchmarkCheckpointState
{
    std::array<std::uint64_t, 2> initial_checksum{{0, 0}};
    HistoryRecord initial_history;
    std::vector<HistoryRecord> history;
    EventWorkCounters work;
    AMRRunCounters amr;
    double next_amr_time = 0;
    double next_history_time = 0;
    double phase_hydro = 0;
    double phase_radiation = 0;
    double phase_amr = 0;
    double evolution_wall = 0;
    unsigned long long radiation_retries = 0;
    double minimum_radiation_candidate_fraction = 1;
};

void WriteHistoryRecord(std::ostream& output, HistoryRecord const& record)
{
    output << std::setprecision(17)
        << record.time << ' ' << record.cycle << ' ' << record.cells << ' '
        << record.active_fraction << ' ' << record.mass << ' '
        << record.momentum.x << ' ' << record.momentum.y << ' '
        << record.momentum.z << ' ' << record.material_energy << ' '
        << record.radiation_energy << ' ' << record.nongrav_energy << ' '
        << record.mass_moment.x << ' ' << record.mass_moment.y << ' '
        << record.mass_moment.z << ' ' << record.center_of_mass.x << ' '
        << record.center_of_mass.y << ' ' << record.center_of_mass.z << ' '
        << record.peak_mach << ' ' << record.peak_outward_mach << ' '
        << record.maximum_compression << ' ' << record.shock_radius;
}

void ReadHistoryRecord(std::istream& input, HistoryRecord& record)
{
    if(!(input >> record.time >> record.cycle >> record.cells
         >> record.active_fraction >> record.mass >> record.momentum.x
         >> record.momentum.y >> record.momentum.z >> record.material_energy
         >> record.radiation_energy >> record.nongrav_energy
         >> record.mass_moment.x >> record.mass_moment.y
         >> record.mass_moment.z >> record.center_of_mass.x
         >> record.center_of_mass.y >> record.center_of_mass.z
         >> record.peak_mach >> record.peak_outward_mach
         >> record.maximum_compression >> record.shock_radius))
        throw std::runtime_error(
            "failed to read benchmark checkpoint history record");
}

void RequireCheckpointKey(std::istream& input, char const* expected)
{
    std::string key;
    if(!(input >> key) || key != expected)
        throw std::runtime_error(std::string(
            "invalid benchmark checkpoint key; expected ") + expected);
}

std::string BenchmarkCheckpointStatePath(std::string const& snapshot,
                                         int rank)
{
    std::ostringstream path;
    path << snapshot << ".benchmark_state_rank_" << rank << ".txt";
    return path.str();
}

void WriteBenchmarkCheckpointState(std::string const& snapshot, int rank,
    Simulation const& simulation, BenchmarkCheckpointState const& state)
{
    std::string const path = BenchmarkCheckpointStatePath(snapshot, rank);
    std::string const temporary_path = path + ".tmp";
    {
        std::ofstream output(temporary_path.c_str(),
            std::ios::out | std::ios::trunc);
        if(!output)
            throw std::runtime_error(
                "failed to create benchmark checkpoint state");
        output << std::setprecision(17)
            << "version 1\n"
            << "rank " << rank << '\n'
            << "cycle " << simulation.GetCycle() << '\n'
            << "time " << simulation.GetTime() << '\n'
            << "initial_checksum " << state.initial_checksum[0] << ' '
            << state.initial_checksum[1] << '\n'
            << "initial_history ";
        WriteHistoryRecord(output, state.initial_history);
        output << "\nhistory_count " << state.history.size() << '\n';
        for(HistoryRecord const& record : state.history)
        {
            output << "history ";
            WriteHistoryRecord(output, record);
            output << '\n';
        }
        output << "work " << state.work.events << ' '
            << state.work.active_cell_updates << ' '
            << state.work.owned_cell_events << ' '
            << state.work.active_faces << ' '
            << state.work.mesh_closure_cells << ' '
            << state.work.gravity_targets << ' '
            << state.work.multigroup_rows << ' '
            << state.work.partial_mesh_events << ' '
            << state.work.reduced_gravity_events << ' '
            << state.work.reduced_multigroup_events << ' '
            << state.work.minimum_bin << ' ' << state.work.maximum_bin << ' '
            << state.work.time_bin_mask << '\n'
            << "amr " << state.amr.calls << ' ' << state.amr.refined << ' '
            << state.amr.derefined << '\n'
            << "next_amr_time " << state.next_amr_time << '\n'
            << "next_history_time " << state.next_history_time << '\n'
            << "phase " << state.phase_hydro << ' ' << state.phase_radiation
            << ' ' << state.phase_amr << '\n'
            << "evolution_wall " << state.evolution_wall << '\n'
            << "radiation_retries " << state.radiation_retries << '\n'
            << "minimum_radiation_candidate_fraction "
            << state.minimum_radiation_candidate_fraction << '\n'
            << "end\n";
        output.flush();
        if(!output)
            throw std::runtime_error(
                "failed to flush benchmark checkpoint state");
    }
    if(std::rename(temporary_path.c_str(), path.c_str()) != 0)
        throw std::runtime_error(
            "failed to publish benchmark checkpoint state");
}

BenchmarkCheckpointState ReadBenchmarkCheckpointState(
    std::string const& snapshot, int rank, Simulation const& simulation)
{
    std::string const path = BenchmarkCheckpointStatePath(snapshot, rank);
    std::ifstream input(path.c_str());
    if(!input)
        throw std::runtime_error(
            "missing benchmark checkpoint state for accumulated restart: " +
            path);
    BenchmarkCheckpointState state;
    unsigned int version = 0;
    int stored_rank = -1;
    unsigned long long stored_cycle = 0;
    double stored_time = 0;
    RequireCheckpointKey(input, "version");
    input >> version;
    RequireCheckpointKey(input, "rank");
    input >> stored_rank;
    RequireCheckpointKey(input, "cycle");
    input >> stored_cycle;
    RequireCheckpointKey(input, "time");
    input >> stored_time;
    if(version != 1 || stored_rank != rank ||
       stored_cycle != static_cast<unsigned long long>(simulation.GetCycle()) ||
       DoubleBits(stored_time) != DoubleBits(simulation.GetTime()))
        throw std::runtime_error(
            "benchmark checkpoint state does not match restart snapshot");
    RequireCheckpointKey(input, "initial_checksum");
    input >> state.initial_checksum[0] >> state.initial_checksum[1];
    RequireCheckpointKey(input, "initial_history");
    ReadHistoryRecord(input, state.initial_history);
    RequireCheckpointKey(input, "history_count");
    size_t history_count = 0;
    input >> history_count;
    state.history.resize(history_count);
    for(HistoryRecord& record : state.history)
    {
        RequireCheckpointKey(input, "history");
        ReadHistoryRecord(input, record);
    }
    RequireCheckpointKey(input, "work");
    input >> state.work.events >> state.work.active_cell_updates
        >> state.work.owned_cell_events >> state.work.active_faces
        >> state.work.mesh_closure_cells >> state.work.gravity_targets
        >> state.work.multigroup_rows >> state.work.partial_mesh_events
        >> state.work.reduced_gravity_events
        >> state.work.reduced_multigroup_events >> state.work.minimum_bin
        >> state.work.maximum_bin >> state.work.time_bin_mask;
    RequireCheckpointKey(input, "amr");
    input >> state.amr.calls >> state.amr.refined >> state.amr.derefined;
    RequireCheckpointKey(input, "next_amr_time");
    input >> state.next_amr_time;
    RequireCheckpointKey(input, "next_history_time");
    input >> state.next_history_time;
    RequireCheckpointKey(input, "phase");
    input >> state.phase_hydro >> state.phase_radiation >> state.phase_amr;
    RequireCheckpointKey(input, "evolution_wall");
    input >> state.evolution_wall;
    RequireCheckpointKey(input, "radiation_retries");
    input >> state.radiation_retries;
    RequireCheckpointKey(input, "minimum_radiation_candidate_fraction");
    input >> state.minimum_radiation_candidate_fraction;
    RequireCheckpointKey(input, "end");
    if(!input || state.history.empty() ||
       !std::isfinite(state.evolution_wall) || state.evolution_wall < 0 ||
       !std::isfinite(state.minimum_radiation_candidate_fraction) ||
       state.minimum_radiation_candidate_fraction <= 0)
        throw std::runtime_error("invalid benchmark checkpoint state");
    return state;
}

HistoryRecord CollectHistory(Voronoi3D const& tess,
    Simulation const& simulation, EquationOfState const& eos,
    size_t active_cells, double radius)
{
    HistoryRecord result;
    result.time = simulation.GetTime();
    result.cycle = static_cast<unsigned long long>(simulation.GetCycle());
    std::vector<ComputationalCell3D> const& cells = simulation.getCells();
    std::vector<Conserved3D> const& extensives = simulation.getExtensives();
    std::vector<Vector3D> points = tess.getAllPoints();
    size_t const owned = extensives.size();
    if(std::getenv("RICH_TEST_DEBUG_COUNTS") != nullptr)
        std::cerr << "CollectHistory sizes: points=" << points.size()
            << " cells=" << cells.size() << " extensives="
            << extensives.size() << " tess_owned=" << tess.GetPointNo()
            << std::endl;
    if(points.size() < owned)
        points.resize(owned, Vector3D());
    double local_shock_score = 0;
    double local_shock_radius = 0;
    for(size_t i = 0; i < owned; ++i)
    {
        ComputationalCell3D const& cell = cells[i];
        Conserved3D const& extensive = extensives.at(i);
        result.mass += extensive.mass;
        result.mass_moment += tess.GetCellCM(static_cast<int>(i)) *
            extensive.mass;
        result.momentum += extensive.momentum;
        result.material_energy += extensive.internal_energy;
        double const radiation_energy = std::accumulate(extensive.Eg.begin(),
            extensive.Eg.end(), 0.0);
        result.radiation_energy += radiation_energy;
        result.nongrav_energy += extensive.energy + radiation_energy;
        double const sound = eos.dp2c(cell.density, cell.pressure,
            cell.tracers, ComputationalCell3D::tracerNames);
        double const r = abs(points[i]);
        double const radial_velocity = r > 0 ?
            (cell.velocity.x * points[i].x +
             cell.velocity.y * points[i].y +
             cell.velocity.z * points[i].z) / r : 0;
        double const outward_mach = sound > 0 ?
            std::max(0.0, radial_velocity / sound) : 0;
        if(sound > 0)
            result.peak_mach = std::max(result.peak_mach,
                abs(cell.velocity) / sound);
        result.peak_outward_mach = std::max(result.peak_outward_mach,
            outward_mach);
        if(tess.GetPointNo() == owned)
        {
            RefinementIndicators const indicator = GetRefinementIndicators(
                i, tess, cells, eos);
            result.maximum_compression = std::max(result.maximum_compression,
                indicator.compression);
            double const shock_score = outward_mach *
                (1.0 + indicator.compression);
            if(r >= 0.10 * radius && r <= 1.5 * radius &&
               shock_score > local_shock_score)
            {
                local_shock_score = shock_score;
                local_shock_radius = r;
            }
        }
    }
    result.cells = static_cast<unsigned long long>(owned);
    unsigned long long global_active = static_cast<unsigned long long>(
        active_cells);
#ifdef RICH_MPI
    double sums[10] = {result.mass, result.momentum.x, result.momentum.y,
        result.momentum.z, result.material_energy, result.radiation_energy,
        result.nongrav_energy, result.mass_moment.x, result.mass_moment.y,
        result.mass_moment.z};
    MPI_Allreduce(MPI_IN_PLACE, sums, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result.mass = sums[0];
    result.momentum = Vector3D(sums[1], sums[2], sums[3]);
    result.material_energy = sums[4];
    result.radiation_energy = sums[5];
    result.nongrav_energy = sums[6];
    result.mass_moment = Vector3D(sums[7], sums[8], sums[9]);
    MPI_Allreduce(MPI_IN_PLACE, &result.peak_mach, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &result.peak_outward_mach, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &result.maximum_compression, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    double global_shock_score = local_shock_score;
    MPI_Allreduce(MPI_IN_PLACE, &global_shock_score, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    double candidate_radius = local_shock_score == global_shock_score ?
        local_shock_radius : 0.0;
    MPI_Allreduce(&candidate_radius, &result.shock_radius, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &result.cells, 1, MPI_UNSIGNED_LONG_LONG,
        MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_active, 1, MPI_UNSIGNED_LONG_LONG,
        MPI_SUM, MPI_COMM_WORLD);
#else
    result.shock_radius = local_shock_radius;
#endif
    if(result.mass > 0)
        result.center_of_mass = result.mass_moment / result.mass;
    result.active_fraction = result.cells > 0 ?
        static_cast<double>(global_active) / static_cast<double>(result.cells) : 0;
    return result;
}
}

int main(void)
{
    int rank = 0;
    int ws = 1;

#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
    double const setup_start = WallSeconds();

    if (rank == 0)
    {
        char cwd_buf[4096];
        if (getcwd(cwd_buf, sizeof(cwd_buf)))
            std::cerr << "CWD: " << cwd_buf << std::endl;
    }

    double const R = 7e10;
    double const M = 2e33;
    double const G = 6.674e-8;

    const double width = 2 * R;
    size_t np = EnvironmentSize("RICH_TEST_POINT_COUNT", static_cast<size_t>(2e6));

    Vector3D ll(-width, -width, -width), ur(width, width, width);
    Voronoi3D tess(ll, ur);

    std::vector<ComputationalCell3D> cells;
    vector<Vector3D> points;

    if (rank == 0)
    {
        size_t np_main = np * 4 / 7;
        size_t np_mid = np * 2 / 7;
        size_t np_far = np - np_main - np_mid;

        points = RandSphereR(np_main, ll, ur, 0, R * 1.1);

        vector<Vector3D> ptemp2 = RandSphereR(np_mid, ll, ur, 0.8 * R, R * 1.05);
        vector<Vector3D> ptemp3 = RandSphereR2(np_far, ll, ur, R, 1.4 * width);

        points.insert(points.end(), ptemp2.begin(), ptemp2.end());
        points.insert(points.end(), ptemp3.begin(), ptemp3.end());

        if(std::getenv("RICH_TEST_FORCE_CENTRAL_POINT") != nullptr &&
           points.size() >= 7)
        {
            double const offset = 0.1 * R;
            points[0] = Vector3D(0, 0, 0);
            points[1] = Vector3D(offset, 0, 0);
            points[2] = Vector3D(-offset, 0, 0);
            points[3] = Vector3D(0, offset, 0);
            points[4] = Vector3D(0, -offset, 0);
            points[5] = Vector3D(0, 0, offset);
            points[6] = Vector3D(0, 0, -offset);
        }

        std::cout << "Total points: " << points.size()
                  << " (main=" << np_main << ", mid=" << np_mid
                  << ", far=" << np_far << ")" << std::endl;
    }

#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    if(std::getenv("RICH_TEST_FORCE_CENTRAL_POINT") == nullptr)
        points = RoundGrid3D(points, ll, ur, 10);

    try {
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
    } catch (UniversalError const& eo) {
        reportError(eo);
        throw;
    }

    if(std::getenv("RICH_TEST_REQUIRE_EMPTY_OWNED_RANK") != nullptr)
    {
#ifdef RICH_MPI
        unsigned long long const local_owned =
            static_cast<unsigned long long>(tess.GetPointNo());
        unsigned long long minimum_owned = local_owned;
        unsigned long long maximum_owned = local_owned;
        unsigned long long empty_ranks = local_owned == 0 ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &minimum_owned, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_owned, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &empty_ranks, 1,
            MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0)
            std::clog << "MPI_OWNED_CELL_RANGE"
                      << " minimum=" << minimum_owned
                      << " maximum=" << maximum_owned
                      << " empty_ranks=" << empty_ranks
                      << " ranks=" << ws << std::endl;
        if(empty_ranks == 0)
            throw std::runtime_error(
                "zero-owned-rank gate did not create an empty MPI rank");
#else
        throw std::runtime_error(
            "zero-owned-rank gate requires an MPI build");
#endif
    }

    if (rank == 0)
        std::cerr << "Finished build" << std::endl;

	double constexpr adiabatic_index = 5.0 / 3.0;
	double constexpr mean_molecular_weight = 0.61;
	double constexpr proton_mass = 1.67262192369e-24;
	double const heat_capacity = CG::boltzmann_constant /
		(mean_molecular_weight * proton_mass * (adiabatic_index - 1));
	IdealGas eos(adiabatic_index, heat_capacity, 1, 0);
	RadiationGrid const radiation_grid = MakeRadiationGrid();
	CG::PreconditionerKind const mg_preconditioner =
		EnvironmentMGPreconditioner();

	LaneEmdenProfile prof(M, R, G);
	double const initial_temperature_floor = 1e4;
	double const pulse_peak_temperature = 3e7;
	double const pulse_radius_fraction = 0.20;
	cells = GetCells(tess, R, eos, prof, radiation_grid,
		initial_temperature_floor);
	double const pulse_energy = AddRadiationPulse(tess, cells, radiation_grid,
		pulse_radius_fraction * R, pulse_peak_temperature);
	if (rank == 0)
		std::cerr << "Finished cells; pulse_energy=" << std::setprecision(17)
		          << pulse_energy << std::endl;

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);
    PointMotion3D const& point_motion =
        std::getenv("RICH_TEST_DISABLE_ROUNDING") != nullptr ?
        static_cast<PointMotion3D const&>(bpm) :
        static_cast<PointMotion3D const&>(pm);

	FreeFreeAbsorptionOpacityMultigroup opacity(1.0, radiation_grid.centers,
		radiation_grid.boundaries, false, true);
	MultigroupDiffusionOpenBoundary radiation_boundary;
	bool const enable_compton =
		std::getenv("RICH_TEST_DISABLE_COMPTON") == nullptr;
	bool const enable_doppler =
		std::getenv("RICH_TEST_DISABLE_DOPPLER") == nullptr;
	RegressionMultigroupDiffusion diffusion(radiation_grid.centers,
		radiation_grid.boundaries, opacity, radiation_boundary, eos,
		std::vector<std::string>(), true, true, enable_compton,
		enable_doppler, -1, true, true,
		mg_preconditioner);
	DefaultCellUpdater cu(false, 0, true, 0, &diffusion);

    RigidWallFlux3D rigidflux(rs);
    RegularFlux3D *regular_flux = new RegularFlux3D(rs);
    IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
    IsBulkFace3D *bulk_face = new IsBulkFace3D();
    vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *,
                          const ConditionActionFlux1::Action3D *>(boundary_face, &rigidflux));
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *,
                          const ConditionActionFlux1::Action3D *>(bulk_face, regular_flux));
    ConditionActionFlux1 fc(flux_vector, interp);

    vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    bool const disable_gravity =
        std::getenv("RICH_TEST_DISABLE_GRAVITY") != nullptr;
    GravityAcceleration3D gravity_acceleration(0.7, true, G);
    ConstantAcceleration3D zero_acceleration(Vector3D(0, 0, 0));
    Acceleration3D const& acceleration = disable_gravity ?
        static_cast<Acceleration3D const&>(zero_acceleration) :
        static_cast<Acceleration3D const&>(gravity_acceleration);
    auto gravity_force = std::make_shared<ConservativeForce3D>(acceleration);
    std::shared_ptr<SourceTerm3D> force = gravity_force;
    std::unique_ptr<PowerLawOpacity> grey_force_opacity;
    std::unique_ptr<DiffusionClosedBox> grey_force_boundary;
    std::unique_ptr<Diffusion> grey_force_diffusion;
    bool const enable_grey_diffusion_force =
        std::getenv("RICH_TEST_ENABLE_GREY_DIFFUSION_FORCE") != nullptr;
    if(enable_grey_diffusion_force)
    {
        grey_force_opacity = std::make_unique<PowerLawOpacity>(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        grey_force_boundary = std::make_unique<DiffusionClosedBox>();
        grey_force_diffusion = std::make_unique<Diffusion>(
            *grey_force_opacity, *grey_force_boundary, eos,
            std::vector<std::string>(), true, true, false, false);
        std::vector<std::shared_ptr<SourceTerm3D>> sources;
        sources.push_back(gravity_force);
        sources.push_back(std::make_shared<DiffusionForce>(
            *grey_force_diffusion, eos, true));
        force = std::make_shared<SeveralSources3D>(sources);
        if(rank == 0)
            std::cout << "GREY_DIFFUSION_FORCE_ZERO_OWNED_TEST enabled=1\n";
    }

    auto tsf = std::make_shared<CourantFriedrichsLewy>(
        0.25, 1, *force, std::vector<std::string>(), false);

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), point_motion, *tsf, fc, cu, eu, *force,
                std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
	simulation.addPhysics(hydroStep);
	auto radiationStep = std::make_shared<RadiationStep>(tess,
		simulation.getCells(), simulation.getExtensives(), simulation.getTracker(),
#ifdef RICH_MPI
		nullptr,
#endif
		diffusion, false);
	simulation.addPhysics(radiationStep);
	double const dynamical_time = std::sqrt(R * R * R / (G * M));
	double const final_time = EnvironmentDouble("RICH_TEST_FINAL_TIME",
		0.20 * dynamical_time);
	unsigned int const requested_initial_bin = static_cast<unsigned int>(EnvironmentSize(
		"RICH_TEST_INITIAL_BIN", 30));
	unsigned int const requested_maximum_bin = static_cast<unsigned int>(EnvironmentSize(
		"RICH_TEST_MAXIMUM_BIN", 40));
	double const time_quantum = EnvironmentDouble("RICH_TEST_TIME_QUANTUM",
		std::ldexp(final_time, -40));
	double const initial_dt = EnvironmentDouble("RICH_TEST_INITIAL_DT",
		std::ldexp(time_quantum, static_cast<int>(requested_initial_bin)));
	double const maximum_individual_dt = std::ldexp(time_quantum,
		static_cast<int>(requested_maximum_bin));
	double const maximum_global_dt = EnvironmentDouble(
		"RICH_TEST_MAX_GLOBAL_DT", std::numeric_limits<double>::infinity());
	if(!(maximum_global_dt > 0))
		throw std::invalid_argument(
			"RICH_TEST_MAX_GLOBAL_DT must be positive when specified");
	simulation.SetTimeStep(initial_dt);

    bool const prescribed_amr = std::getenv("RICH_TEST_PRESCRIBED_AMR") !=
        nullptr;
    size_t const prescribed_refine_count = EnvironmentSize(
        "RICH_TEST_PRESCRIBED_REFINE", 512);
    size_t const prescribed_derefine_count = std::min(
        prescribed_refine_count, EnvironmentSize(
            "RICH_TEST_PRESCRIBED_DEREFINE", 256));
    RadiationShockRefine3D refine(eos, R, R / 256.0,
        EnvironmentSize("RICH_TEST_MAX_CELLS", static_cast<size_t>(3000000)),
        EnvironmentSize("RICH_TEST_AMR_BATCH", static_cast<size_t>(50000)),
        prescribed_amr, EnvironmentDouble("RICH_TEST_PRESCRIBED_REFINE_TIME",
            0.005 * dynamical_time), np, prescribed_refine_count);
    RadiationShockRemove3D remove(eos, R, 0.025 * dynamical_time,
        simulation.GetMaxID(), prescribed_amr,
        EnvironmentDouble("RICH_TEST_PRESCRIBED_DEREFINE_TIME",
            0.01 * dynamical_time),
        np + prescribed_refine_count - prescribed_derefine_count);
    AMR3D amr(eos, refine, remove, interp);
    ControlledAMR controlled_amr(simulation, amr,
        EnvironmentDouble("RICH_TEST_AMR_INTERVAL", 0.01 * dynamical_time));

    char const* restart_input = std::getenv("RICH_TEST_RESTART_INPUT");
    char const* individual_mode = std::getenv("RICH_INDIVIDUAL_MODE");
    bool const individual_enabled = individual_mode != nullptr &&
        individual_mode[0] != '\0';
    if(restart_input != nullptr && restart_input[0] != '\0')
        ReadSimulation(restart_input, simulation);
    else if(individual_mode != nullptr && individual_mode[0] != '\0')
    {
        IndividualTimeStepOptions options;
        std::string const mode(individual_mode);
        bool const synchronized = mode == "full";
        bool const auto_partial = mode == "partial";
        if(!synchronized && !auto_partial && mode != "full-variable")
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_MODE must be 'full', 'full-variable', or 'partial'");
		options.initial_bin = requested_initial_bin;
		options.maximum_bin = requested_maximum_bin;
		options.time_quantum = time_quantum;
		options.force_synchronized = synchronized;
        options.mesh_build_policy = auto_partial ?
            IndividualMeshBuildPolicy::AutoPartial :
            IndividualMeshBuildPolicy::FullReference;
        options.partial_build_fraction = EnvironmentDouble(
            "RICH_TEST_PARTIAL_BUILD_FRACTION",
            options.partial_build_fraction);
        options.verify_partial_build = std::getenv(
            "RICH_VERIFY_PARTIAL_BUILD") != nullptr;
        simulation.EnableIndividualTimeSteps(options);

        if(std::getenv("RICH_TEST_SPARSE_INITIAL_BIN") != nullptr)
        {
            if(synchronized || options.initial_bin < 2)
                throw std::invalid_argument(
                    "RICH_TEST_SPARSE_INITIAL_BIN requires variable individual timesteps");
            IndividualTimeStepScheduler* scheduler =
                simulation.GetIndividualTimeStepScheduler();
            scheduler->initialize(simulation.getCells(), simulation.GetTime(),
                                  simulation.GetTimeStep());
            std::vector<CellTimeState>& states = scheduler->states();
            bool const single_rank = std::getenv(
                "RICH_TEST_SPARSE_SINGLE_RANK") != nullptr;
            bool const all_local_sparse = std::getenv(
                "RICH_TEST_SPARSE_ALL_LOCAL") != nullptr;
            if(all_local_sparse && !single_rank)
                throw std::invalid_argument(
                    "RICH_TEST_SPARSE_ALL_LOCAL requires "
                    "RICH_TEST_SPARSE_SINGLE_RANK");
            bool const activate_reference_maximum = std::getenv(
                "RICH_TEST_SPARSE_MAX_ER_CELL") != nullptr;
            bool const require_cross_rank_face = std::getenv(
                "RICH_TEST_GRAVITY_PREDICTOR_SPEED") != nullptr;
            std::size_t sparse_index = states.size();
            if(activate_reference_maximum)
            {
                double unrestricted_maximum =
                    -std::numeric_limits<double>::infinity();
                for(ComputationalCell3D const& cell : simulation.getCells())
                    unrestricted_maximum = std::max(
                        unrestricted_maximum, cell.Erad * cell.density);
#ifdef RICH_MPI
                MPI_Allreduce(MPI_IN_PLACE, &unrestricted_maximum, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
#ifdef RICH_MPI
                struct
                {
                    double value;
                    int rank;
                } local_maximum = {-std::numeric_limits<double>::infinity(), rank};
#else
                double local_maximum = -std::numeric_limits<double>::infinity();
#endif
                for(std::size_t index = 0; index < states.size(); ++index)
                {
                    if(require_cross_rank_face)
                    {
                        std::vector<std::size_t> neighbors;
                        tess.GetNeighbors(index, neighbors);
                        bool cross_rank_face = false;
                        for(std::size_t const neighbor : neighbors)
                            cross_rank_face = cross_rank_face ||
                                (neighbor >= tess.GetPointNo() &&
                                 !tess.IsPointOutsideBox(neighbor));
                        if(!cross_rank_face)
                            continue;
                    }
                    double const value = simulation.getCells()[index].Erad *
                        simulation.getCells()[index].density;
#ifdef RICH_MPI
                    if(value > local_maximum.value)
                    {
                        local_maximum.value = value;
                        sparse_index = index;
                    }
#else
                    if(value > local_maximum)
                    {
                        local_maximum = value;
                        sparse_index = index;
                    }
#endif
                }
#ifdef RICH_MPI
                auto global_maximum = local_maximum;
                MPI_Allreduce(&local_maximum, &global_maximum, 1,
                    MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
                if(!std::isfinite(global_maximum.value))
                    throw std::logic_error(
                        "gravity sparse test found no cross-rank active/passive face");
                if(rank != global_maximum.rank)
                    sparse_index = states.size();
                if(require_cross_rank_face && sparse_index < states.size())
                {
                    ComputationalCell3D& active_cell =
                        simulation.getCells()[sparse_index];
                    double const current = active_cell.Erad * active_cell.density;
                    double const factor = std::max(
                        2.0, 2.0 * unrestricted_maximum /
                            std::max(current, std::numeric_limits<double>::min()));
                    active_cell.Erad *= factor;
                    for(double& group_energy : active_cell.Eg)
                        group_energy *= factor;
                    simulation.getExtensives()[sparse_index].Erad *= factor;
                    global_maximum.value = active_cell.Erad * active_cell.density;
                }
                MPI_Allreduce(MPI_IN_PLACE, &global_maximum.value, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if(rank == 0)
                    std::clog << "MG_ACTIVE_REFERENCE_MAX_TEST"
                              << " value=" << global_maximum.value
                              << " owner=" << global_maximum.rank << '\n';
#else
                std::clog << "MG_ACTIVE_REFERENCE_MAX_TEST"
                          << " value=" << local_maximum
                          << " owner=0\n";
#endif
            }
            else if(!single_rank || rank == 0)
                sparse_index = states.size() / 2;

            if(all_local_sparse)
            {
                if(rank == 0)
                    for(CellTimeState& state : states)
                    {
                        state.time_bin = static_cast<std::uint8_t>(
                            options.initial_bin - 2);
                        state.end_tick =
                            std::uint64_t(1) << state.time_bin;
                    }
                if(rank == 0)
                    std::clog << "INDIVIDUAL_SPARSE_ALL_LOCAL_TEST"
                              << " active_cells=" << states.size()
                              << " active_rank=0\n";
            }
            else if(sparse_index < states.size())
            {
                CellTimeState& sparse_state = states[sparse_index];
                sparse_state.time_bin = static_cast<std::uint8_t>(
                    options.initial_bin - 2);
                sparse_state.end_tick =
                    std::uint64_t(1) << sparse_state.time_bin;
            }
            else if(!activate_reference_maximum && (!single_rank || rank == 0))
            {
                if(states.empty())
                    throw std::logic_error(
                        "RICH_TEST_SPARSE_INITIAL_BIN requires at least one cell");
            }

            char const* predictor_speed_text = std::getenv(
                "RICH_TEST_GRAVITY_PREDICTOR_SPEED");
            if(predictor_speed_text != nullptr &&
               predictor_speed_text[0] != '\0')
            {
                double const base_speed = EnvironmentDouble(
                    "RICH_TEST_GRAVITY_PREDICTOR_SPEED", 0.0);
                if(!std::isfinite(base_speed) || base_speed <= 0)
                    throw std::invalid_argument(
                        "RICH_TEST_GRAVITY_PREDICTOR_SPEED must be positive");
                if(states.size() != tess.GetPointNo())
                    throw std::logic_error(
                        "gravity predictor test state/mesh size mismatch");
                for(std::size_t index = 0; index < states.size(); ++index)
                {
                    Vector3D const point = tess.GetMeshPoint(index);
                    double const radius = abs(point);
                    Vector3D const direction = radius > 0
                        ? point * (1.0 / radius) : Vector3D(1, 0, 0);
                    double const speed = base_speed *
                        (0.25 + std::min(radius / R, 1.5));
                    states[index].point_velocity = direction * speed;
                }
                if(rank == 0)
                    std::clog << std::setprecision(17)
                              << "GRAVITY_PREDICTOR_MOTION_TEST"
                              << " base_speed=" << base_speed
                              << " profile=radial_radius_dependent\n";
            }
        }
    }
    if(individual_enabled)
    {
        IndividualTimeStepScheduler* scheduler =
            simulation.GetIndividualTimeStepScheduler();
        if(scheduler == nullptr)
            throw std::logic_error(
                "individual benchmark has no timestep scheduler");
        if(!scheduler->initialized())
        {
#ifdef RICH_MPI
            double initial_dt_min = simulation.GetTimeStep();
            double initial_dt_max = initial_dt_min;
            MPI_Allreduce(MPI_IN_PLACE, &initial_dt_min, 1, MPI_DOUBLE,
                          MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &initial_dt_max, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);
            if(initial_dt_min != initial_dt_max)
                throw std::logic_error(
                    "MPI individual benchmark requires one shared initial timestep");
            simulation.SetTimeStep(initial_dt_min);
#endif
            scheduler->initialize(simulation.getCells(), simulation.GetTime(),
                                  simulation.GetTimeStep());
        }
    }
    if(individual_enabled && std::getenv("RICH_TEST_DISABLE_AMR") == nullptr)
        simulation.SetIndividualAMR(
            [&](IndividualStepContext const& context)
            {
                return controlled_amr.ApplyIndividual(context);
            });
#ifdef RICH_MPI
    simulation.PresetLoadBalance("hydro");
#endif
    GravityPolicyBaseline const gravity_policy_baseline =
        CaptureGravityPolicyBaseline(simulation);

    char output_cwd_new[4096];
    if(getcwd(output_cwd_new, sizeof(output_cwd_new)) == nullptr)
        throw std::runtime_error("Failed to resolve current working directory");
    char const* artifact_dir_new = std::getenv("THUNDER_ARTIFACT_DIR");
    std::string const output_dir_new = artifact_dir_new != nullptr &&
        artifact_dir_new[0] != '\0' ? std::string(artifact_dir_new) :
        std::string(output_cwd_new);
    std::string const run_mode = individual_enabled ?
        std::string(individual_mode) : std::string("global");

    size_t const initial_active = simulation.getExtensives().size();
    double const history_interval = EnvironmentDouble(
        "RICH_TEST_HISTORY_INTERVAL", 0.01 * dynamical_time);
    bool const accumulated_restart =
        std::getenv("RICH_TEST_ACCUMULATE_RESTART") != nullptr;
    if(accumulated_restart &&
       (restart_input == nullptr || restart_input[0] == '\0'))
        throw std::invalid_argument(
            "RICH_TEST_ACCUMULATE_RESTART requires RICH_TEST_RESTART_INPUT");
    BenchmarkCheckpointState benchmark_state;
    benchmark_state.initial_checksum =
        InitialStateChecksum(tess, simulation.getCells());
    benchmark_state.initial_history = CollectHistory(tess, simulation,
        eos, initial_active, R);
    benchmark_state.history.push_back(benchmark_state.initial_history);
    benchmark_state.next_history_time = history_interval;
    benchmark_state.next_amr_time = controlled_amr.NextTime();
    if(accumulated_restart)
    {
        benchmark_state = ReadBenchmarkCheckpointState(
            restart_input, rank, simulation);
        controlled_amr.RestoreCheckpointState(
            benchmark_state.next_amr_time, benchmark_state.amr);
        std::clog << std::setprecision(17)
                  << "BENCHMARK_ACCUMULATED_RESTART"
                  << " rank=" << rank
                  << " cycle=" << simulation.GetCycle()
                  << " time=" << simulation.GetTime()
                  << " prior_evolution_wall="
                  << benchmark_state.evolution_wall
                  << std::endl;
    }
    std::array<std::uint64_t, 2>& initial_checksum =
        benchmark_state.initial_checksum;
    HistoryRecord& initial_history = benchmark_state.initial_history;
    std::vector<HistoryRecord>& history = benchmark_state.history;
    EventWorkCounters& work_counters = benchmark_state.work;
    double& next_history_time = benchmark_state.next_history_time;
    double& phase_hydro = benchmark_state.phase_hydro;
    double& phase_radiation = benchmark_state.phase_radiation;
    double& phase_amr = benchmark_state.phase_amr;
    double const evolution_elapsed_offset = benchmark_state.evolution_wall;
    unsigned long long const radiation_retry_offset =
        benchmark_state.radiation_retries;
    double const radiation_candidate_fraction_offset =
        benchmark_state.minimum_radiation_candidate_fraction;
    size_t const max_cycles_new = EnvironmentSize("RICH_TEST_MAX_CYCLES",
        std::numeric_limits<size_t>::max());
    size_t const progress_cycle_interval = EnvironmentSize(
        "RICH_TEST_PROGRESS_CYCLES", 100);
    double const progress_wall_interval = EnvironmentDouble(
        "RICH_TEST_PROGRESS_WALL_SECONDS", 60.0);
    double const checkpoint_wall_interval = EnvironmentDouble(
        "RICH_TEST_CHECKPOINT_WALL_SECONDS", 0.0);
    double const segment_wall_limit = EnvironmentDouble(
        "RICH_TEST_SEGMENT_WALL_SECONDS", 0.0);
    if(segment_wall_limit < 0)
        throw std::invalid_argument(
            "RICH_TEST_SEGMENT_WALL_SECONDS must be nonnegative");

#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double const setup_elapsed_local = WallSeconds() - setup_start;
    PrintIndividualRestartFingerprint(simulation, "initial");
    bool preserve_initial_snapshot = false;
    if(accumulated_restart)
    {
        std::ifstream prior_initial(
            (output_dir_new + "/initial_state.h5").c_str());
        preserve_initial_snapshot = static_cast<bool>(prior_initial);
    }
#ifdef RICH_MPI
    int preserve_initial_local = preserve_initial_snapshot ? 1 : 0;
    int preserve_initial_global = 0;
    MPI_Allreduce(&preserve_initial_local, &preserve_initial_global, 1,
        MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    preserve_initial_snapshot = preserve_initial_global != 0;
#endif
    double initial_snapshot_elapsed_max = 0;
    if(!preserve_initial_snapshot)
    {
        double const initial_snapshot_start = WallSeconds();
        WriteSimulation(simulation, output_dir_new + "/initial_state.h5");
        WriteBenchmarkVTK(simulation, output_dir_new + "/initial_state.vtu");
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        initial_snapshot_elapsed_max = WallSeconds() - initial_snapshot_start;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &initial_snapshot_elapsed_max, 1,
            MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    }
    else if(rank == 0)
        std::clog << "BENCHMARK_INITIAL_SNAPSHOT_PRESERVED" << std::endl;
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    NullStreamBuffer null_stream_buffer;
    std::streambuf* saved_cout_buffer = nullptr;
    if(std::getenv("RICH_QUIET") != nullptr)
        saved_cout_buffer = std::cout.rdbuf(&null_stream_buffer);
    std::ofstream progress_stream;
    if(rank == 0)
    {
        bool append_progress = false;
        if(accumulated_restart)
        {
            std::ifstream prior_progress(
                (output_dir_new + "/progress.tsv").c_str());
            append_progress = static_cast<bool>(prior_progress);
        }
        progress_stream.open((output_dir_new + "/progress.tsv").c_str(),
            std::ios::out | (append_progress ? std::ios::app :
                std::ios::trunc));
        if(!progress_stream)
            std::cerr << "WARNING: failed to open progress.tsv; progress "
                "will remain available in run.log" << std::endl;
        else
        {
            progress_stream << std::setprecision(17);
            if(!append_progress)
                progress_stream
                    << "phase\tmode\tcycle\ttime\tdt\tcompletion"
                    "\tactive_rank0\towned_rank0\twall_seconds\n";
            progress_stream.flush();
        }
    }
    double const evolution_start = WallSeconds();
    double checkpoint_elapsed_local = 0;
    double next_progress_wall = progress_wall_interval > 0 ?
        progress_wall_interval : std::numeric_limits<double>::infinity();
    double next_checkpoint_wall = checkpoint_wall_interval > 0 ?
        checkpoint_wall_interval : std::numeric_limits<double>::infinity();
    size_t checkpoint_slot = 0;
    if(restart_input != nullptr && restart_input[0] != '\0')
    {
        std::string const restart_path(restart_input);
        if(restart_path.size() >= 5 &&
           restart_path.compare(restart_path.size() - 5, 5, "_0.h5") == 0)
            checkpoint_slot = 1;
    }
    size_t last_active = initial_active;
    IndividualTerminalTarget terminal_target;
    bool const has_terminal_target = individual_enabled &&
        GetIndividualTerminalTarget(simulation, final_time, terminal_target);
    if(individual_enabled && !has_terminal_target)
        throw std::logic_error(
            "individual benchmark failed to resolve its terminal tick");
    auto report_progress = [&](char const* phase)
    {
        if(rank != 0)
            return;
        double const wall_elapsed = evolution_elapsed_offset +
            WallSeconds() - evolution_start - checkpoint_elapsed_local;
        double const completion_target = has_terminal_target ?
            terminal_target.represented_time : final_time;
        double const completion = completion_target > 0 ? std::max(0.0,
            std::min(1.0, simulation.GetTime() / completion_target)) : 1.0;
        std::ostringstream record;
        record << std::setprecision(17)
            << "PROGRESS phase=" << phase
            << " mode=" << run_mode
            << " cycle=" << simulation.GetCycle()
            << " time=" << simulation.GetTime()
            << " dt=" << simulation.GetTimeStep()
            << " completion=" << completion
            << " active_rank0=" << last_active
            << " owned_rank0=" << simulation.getExtensives().size()
            << " wall_seconds=" << wall_elapsed;
        std::cerr << record.str() << std::endl;
        if(progress_stream)
        {
            progress_stream << phase << '\t' << run_mode << '\t'
                << simulation.GetCycle() << '\t' << simulation.GetTime()
                << '\t' << simulation.GetTimeStep() << '\t' << completion
                << '\t' << last_active << '\t'
                << simulation.getExtensives().size() << '\t' << wall_elapsed
                << '\n';
            progress_stream.flush();
        }
    };
    report_progress("initial");
    double const final_time_tolerance = 1e-12 * std::max(final_time, 1.0);
    bool segment_stopped = false;
    auto evolution_pending = [&]()
    {
        if(simulation.GetCycle() >= max_cycles_new)
            return false;
        if(!individual_enabled)
            return simulation.GetTime() + final_time_tolerance < final_time;
        IndividualTimeStepScheduler const* scheduler =
            simulation.GetIndividualTimeStepScheduler();
        if(scheduler == nullptr || !scheduler->initialized())
            throw std::logic_error(
                "individual benchmark lost its initialized scheduler");
        return scheduler->currentTick() < terminal_target.tick;
    };
    while(evolution_pending())
    {
        if(!individual_enabled)
        {
            double const remaining = final_time - simulation.GetTime();
            double const permitted_dt = std::min(remaining, maximum_global_dt);
            if(simulation.GetTimeStep() > permitted_dt)
                simulation.SetTimeStep(permitted_dt);
        }
        else if(!PrepareIndividualTerminalEvent(simulation, terminal_target))
            throw std::logic_error(
                "individual terminal tick changed during evolution");

        try
        {
            simulation.step();
        }
        catch(UniversalError const& error)
        {
            report_progress("error");
            if(saved_cout_buffer != nullptr)
                std::cout.rdbuf(saved_cout_buffer);
            reportError(error);
            throw;
        }
        catch(std::exception const& error)
        {
            report_progress("error");
            if(saved_cout_buffer != nullptr)
                std::cout.rdbuf(saved_cout_buffer);
            std::cerr << "Evolution failed at cycle " << simulation.GetCycle()
                << " time " << simulation.GetTime() << ": "
                << error.what() << std::endl;
            throw;
        }
        if(!individual_enabled &&
           std::getenv("RICH_TEST_DISABLE_AMR") == nullptr)
        {
            double const amr_start = WallSeconds();
            controlled_amr.ApplyGlobal(simulation.GetTime());
            phase_amr += WallSeconds() - amr_start;
        }

        size_t const active = work_counters.Record(tess, simulation,
            radiation_grid.centers.size());
        last_active = active;
        std::map<std::string, double> const& physics_times =
            simulation.getLastLocalPhysicsTimes();
        auto const hydro_time = physics_times.find("hydro");
        if(hydro_time != physics_times.end())
            phase_hydro += hydro_time->second;
        auto const radiation_time = physics_times.find("radiation");
        if(radiation_time != physics_times.end())
            phase_radiation += radiation_time->second;
        auto const individual_amr_time = physics_times.find("individual-amr");
        if(individual_amr_time != physics_times.end())
            phase_amr += individual_amr_time->second;

        if(history_interval > 0 &&
           simulation.GetTime() + 1e-13 * final_time >= next_history_time &&
           tess.GetPointNo() == simulation.getExtensives().size())
        {
            history.push_back(CollectHistory(tess, simulation, eos, active, R));
            do
                next_history_time += history_interval;
            while(next_history_time <= simulation.GetTime());
        }

        double const progress_wall = WallSeconds() - evolution_start;
        bool const cycle_due = progress_cycle_interval > 0 &&
            simulation.GetCycle() % progress_cycle_interval == 0;
        bool const wall_due = progress_wall_interval > 0 &&
            progress_wall >= next_progress_wall;
        if(cycle_due || wall_due)
        {
            report_progress("evolution");
            if(wall_due)
            {
                do
                    next_progress_wall += progress_wall_interval;
                while(next_progress_wall <= progress_wall);
            }
        }

        double const raw_progress_wall = WallSeconds() - evolution_start;
        bool checkpoint_due = checkpoint_wall_interval > 0 &&
            raw_progress_wall >= next_checkpoint_wall;
        bool segment_due = segment_wall_limit > 0 &&
            WallSeconds() - setup_start >= segment_wall_limit;
#ifdef RICH_MPI
        int due_flags[2] = {checkpoint_due ? 1 : 0,
                            segment_due ? 1 : 0};
        MPI_Allreduce(MPI_IN_PLACE, due_flags, 2, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD);
        checkpoint_due = due_flags[0] != 0;
        segment_due = due_flags[1] != 0;
#endif
        if(checkpoint_due || segment_due)
        {
            std::ostringstream checkpoint_name;
            checkpoint_name << output_dir_new << "/restart_checkpoint_"
                            << checkpoint_slot << ".h5";
            PrintIndividualRestartFingerprint(simulation, "checkpoint");
            BenchmarkCheckpointState checkpoint_state = benchmark_state;
            checkpoint_state.amr = controlled_amr.Counters();
            checkpoint_state.next_amr_time = controlled_amr.NextTime();
            checkpoint_state.evolution_wall = evolution_elapsed_offset +
                WallSeconds() - evolution_start - checkpoint_elapsed_local;
            checkpoint_state.radiation_retries = radiation_retry_offset +
                static_cast<unsigned long long>(radiationStep->
                    GetCumulativeIndividualRejectedCandidates());
            checkpoint_state.minimum_radiation_candidate_fraction = std::min(
                radiation_candidate_fraction_offset,
                radiationStep->GetSmallestIndividualCandidateFraction());
#ifdef RICH_MPI
            double checkpoint_timing_max[4] = {
                checkpoint_state.evolution_wall,
                checkpoint_state.phase_hydro,
                checkpoint_state.phase_radiation,
                checkpoint_state.phase_amr};
            MPI_Allreduce(MPI_IN_PLACE, checkpoint_timing_max, 4, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);
            checkpoint_state.evolution_wall = checkpoint_timing_max[0];
            checkpoint_state.phase_hydro = checkpoint_timing_max[1];
            checkpoint_state.phase_radiation = checkpoint_timing_max[2];
            checkpoint_state.phase_amr = checkpoint_timing_max[3];
            MPI_Allreduce(MPI_IN_PLACE, &checkpoint_state.radiation_retries,
                1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE,
                &checkpoint_state.minimum_radiation_candidate_fraction, 1,
                MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
            double const checkpoint_start = WallSeconds();
            WriteSimulation(simulation, checkpoint_name.str());
#ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            WriteBenchmarkCheckpointState(checkpoint_name.str(), rank,
                simulation, checkpoint_state);
#ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            double checkpoint_elapsed_max = WallSeconds() - checkpoint_start;
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &checkpoint_elapsed_max, 1,
                MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
            if(rank == 0)
            {
                std::string const pointer_path =
                    output_dir_new + "/restart_checkpoint_latest.txt";
                std::string const temporary_path = pointer_path + ".tmp";
                {
                    std::ofstream pointer(temporary_path.c_str(),
                        std::ios::out | std::ios::trunc);
                    if(!pointer)
                        throw std::runtime_error(
                            "failed to create rolling checkpoint pointer");
                    pointer << std::setprecision(17)
                        << "version 1\n"
                        << "snapshot " << checkpoint_name.str() << '\n'
                        << "benchmark_state_version 1\n"
                        << "slot " << checkpoint_slot << '\n'
                        << "cycle " << simulation.GetCycle() << '\n'
                        << "time " << simulation.GetTime() << '\n'
                        << "dt " << simulation.GetTimeStep() << '\n'
                        << "checkpoint_seconds_max "
                        << checkpoint_elapsed_max << '\n';
                    pointer.flush();
                    if(!pointer)
                        throw std::runtime_error(
                            "failed to flush rolling checkpoint pointer");
                }
                if(std::rename(temporary_path.c_str(),
                               pointer_path.c_str()) != 0)
                    throw std::runtime_error(
                        "failed to publish rolling checkpoint pointer");
                std::clog << std::setprecision(17)
                          << "ROLLING_RESTART_CHECKPOINT"
                          << " slot=" << checkpoint_slot
                          << " cycle=" << simulation.GetCycle()
                          << " time=" << simulation.GetTime()
                          << " seconds_max=" << checkpoint_elapsed_max
                          << " snapshot=" << checkpoint_name.str()
                          << std::endl;
            }
#ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            checkpoint_elapsed_local += WallSeconds() - checkpoint_start;
            checkpoint_slot = 1 - checkpoint_slot;
            double const checkpoint_progress_wall =
                WallSeconds() - evolution_start;
            if(checkpoint_wall_interval > 0)
            {
                do
                    next_checkpoint_wall += checkpoint_wall_interval;
                while(next_checkpoint_wall <= checkpoint_progress_wall);
            }
            report_progress("checkpoint");
            if(segment_due)
            {
                segment_stopped = true;
                break;
            }
        }
    }
    if(segment_stopped)
    {
        report_progress("segment");
        if(saved_cout_buffer != nullptr)
            std::cout.rdbuf(saved_cout_buffer);
        if(rank == 0)
            std::clog << std::setprecision(17)
                      << "BENCHMARK_SEGMENT_COMPLETE"
                      << " cycle=" << simulation.GetCycle()
                      << " time=" << simulation.GetTime()
                      << " target_time=" << final_time
                      << " exit_code=75"
                      << std::endl;
#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return 75;
    }
    report_progress("final");
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double const evolution_elapsed_local = evolution_elapsed_offset +
        WallSeconds() - evolution_start - checkpoint_elapsed_local;
    if(saved_cout_buffer != nullptr)
        std::cout.rdbuf(saved_cout_buffer);

    // Output and comparison are deliberately outside the primary evolution
    // timer.  Restore canonical full geometry once, then write one snapshot.
    size_t const final_owned_count = simulation.getExtensives().size();
    bool identity_mesh = tess.GetPointNo() == final_owned_count;
    auto const& final_map = tess.GetIndicesInAllPoints();
    for(size_t i = 0; identity_mesh && i < final_owned_count; ++i)
    {
        auto const mapped = final_map.find(i);
        identity_mesh = mapped != final_map.end() && mapped->second == i;
    }
    bool restore_full_mesh = !identity_mesh;
#ifdef RICH_MPI
    int restore_full_mesh_local = restore_full_mesh ? 1 : 0;
    int restore_full_mesh_global = 0;
    MPI_Allreduce(&restore_full_mesh_local, &restore_full_mesh_global, 1,
        MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    restore_full_mesh = restore_full_mesh_global != 0;
#endif
    if(restore_full_mesh)
    {
        std::vector<Vector3D> final_points = sim.GetIndividualGeneratorPoints();
        bool final_points_valid = final_points.size() == final_owned_count;
#ifdef RICH_MPI
        int final_points_valid_local = final_points_valid ? 1 : 0;
        int final_points_valid_global = 0;
        MPI_Allreduce(&final_points_valid_local, &final_points_valid_global, 1,
            MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        final_points_valid = final_points_valid_global != 0;
#endif
        if(!final_points_valid)
        {
            if(final_points.size() != final_owned_count)
                std::cerr << "Final mesh restore failed: canonical_points="
                    << final_points.size() << " owned=" << final_owned_count
                    << " tess_cells=" << tess.GetPointNo() << std::endl;
            throw std::runtime_error(
                "Final partial mesh lost canonical generator positions");
        }
        final_points.resize(final_owned_count);
#ifdef RICH_MPI
        tess.BuildParallel(final_points, true, true);
#else
        tess.Build(final_points);
#endif
    }

    HistoryRecord const final_history = CollectHistory(tess, simulation, eos,
        final_owned_count, R);
    if(history.empty() || history.back().time != final_history.time)
        history.push_back(final_history);

    size_t const profile_bins = 512;
    size_t const profile_fields = 7;
    std::vector<double> profile_local(profile_bins * profile_fields, 0.0);
    std::vector<double> spectrum_local(radiation_grid.centers.size(), 0.0);
    double minimum_material_energy = std::numeric_limits<double>::max();
    double minimum_group_energy = std::numeric_limits<double>::max();
    double maximum_compression_ratio = 0;
    std::vector<ComputationalCell3D> const& final_cells = simulation.getCells();
    std::vector<Conserved3D> const& final_extensives =
        simulation.getExtensives();
    for(size_t i = 0; i < tess.GetPointNo(); ++i)
    {
        ComputationalCell3D const& cell = final_cells.at(i);
        Conserved3D const& extensive = final_extensives.at(i);
        Vector3D const cm = tess.GetCellCM(i);
        double const r = abs(cm);
        double const volume = tess.GetVolume(i);
        double const sound = eos.dp2c(cell.density, cell.pressure,
            cell.tracers, ComputationalCell3D::tracerNames);
        double const radial_velocity = r > 0 ?
            (cell.velocity.x * cm.x + cell.velocity.y * cm.y +
             cell.velocity.z * cm.z) / r : 0;
        double const mach = sound > 0 ? abs(cell.velocity) / sound : 0;
        if(r < 1.5 * R)
        {
            size_t bin = static_cast<size_t>(profile_bins * r / (1.5 * R));
            if(bin >= profile_bins)
                bin = profile_bins - 1;
            size_t const offset = bin * profile_fields;
            profile_local[offset] += volume;
            profile_local[offset + 1] += r * volume;
            profile_local[offset + 2] += cell.density * volume;
            profile_local[offset + 3] += cell.pressure * volume;
            profile_local[offset + 4] += radial_velocity * volume;
            profile_local[offset + 5] += cell.temperature * volume;
            profile_local[offset + 6] += RadiationEnergyDensity(cell) * volume;
            double const initial_density = prof.densityAt(r, R);
            if(initial_density > 0)
                maximum_compression_ratio = std::max(maximum_compression_ratio,
                    cell.density / initial_density);
        }
        minimum_material_energy = std::min(minimum_material_energy,
            cell.internal_energy);
        for(size_t group = 0; group < cell.Eg.size(); ++group)
        {
            minimum_group_energy = std::min(minimum_group_energy,
                cell.Eg[group]);
            spectrum_local[group] += extensive.Eg[group];
        }
    }

    std::vector<double> profile_global = profile_local;
    std::vector<double> spectrum_global = spectrum_local;
    MultigroupDiffusion::CoefficientDiagnostics const coefficient_diagnostics =
        final_cells.empty() ?
        MultigroupDiffusion::CoefficientDiagnostics{} :
        diffusion.coefficientDiagnostics();
    double minimum_fleck = coefficient_diagnostics.minimum_fleck_factor;
    double maximum_fleck = coefficient_diagnostics.maximum_fleck_factor;
    unsigned long long fleck_samples = coefficient_diagnostics.fleck_samples;
    double maximum_transport_scattering =
        coefficient_diagnostics.maximum_transport_scattering;

    double setup_elapsed_max = setup_elapsed_local;
    double evolution_elapsed_max = evolution_elapsed_local;
    double phase_hydro_max = phase_hydro;
    double phase_radiation_max = phase_radiation;
    double phase_amr_max = phase_amr;
    EventWorkCounters global_work = work_counters;
    AMRRunCounters global_amr = controlled_amr.Counters();
    unsigned long long radiation_retries = radiation_retry_offset +
        static_cast<unsigned long long>(
            radiationStep->GetCumulativeIndividualRejectedCandidates());
    double minimum_radiation_candidate_fraction = std::min(
        radiation_candidate_fraction_offset,
        radiationStep->GetSmallestIndividualCandidateFraction());
    IndividualRadiationDefectAccounting radiation_defect;
    if(IndividualTimeStepScheduler const* scheduler =
           simulation.GetIndividualTimeStepScheduler())
        radiation_defect = scheduler->radiationDefectAccounting();
#ifdef RICH_MPI
    MPI_Reduce(profile_local.data(), profile_global.data(),
        static_cast<int>(profile_global.size()), MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD);
    MPI_Reduce(spectrum_local.data(), spectrum_global.data(),
        static_cast<int>(spectrum_global.size()), MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_material_energy, 1, MPI_DOUBLE,
        MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_group_energy, 1, MPI_DOUBLE,
        MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_fleck, 1, MPI_DOUBLE, MPI_MIN,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_fleck, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_transport_scattering, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_compression_ratio, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fleck_samples, 1, MPI_UNSIGNED_LONG_LONG,
        MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &setup_elapsed_max, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &evolution_elapsed_max, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &phase_hydro_max, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &phase_radiation_max, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &phase_amr_max, 1, MPI_DOUBLE, MPI_MAX,
        MPI_COMM_WORLD);

    unsigned long long work_sum[9] = {
        work_counters.active_cell_updates, work_counters.owned_cell_events,
        work_counters.active_faces, work_counters.mesh_closure_cells,
        work_counters.gravity_targets, work_counters.multigroup_rows,
        work_counters.partial_mesh_events,
        work_counters.reduced_gravity_events,
        work_counters.reduced_multigroup_events};
    MPI_Allreduce(MPI_IN_PLACE, work_sum, 9, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
        MPI_COMM_WORLD);
    global_work.active_cell_updates = work_sum[0];
    global_work.owned_cell_events = work_sum[1];
    global_work.active_faces = work_sum[2];
    global_work.mesh_closure_cells = work_sum[3];
    global_work.gravity_targets = work_sum[4];
    global_work.multigroup_rows = work_sum[5];
    global_work.partial_mesh_events = work_sum[6];
    global_work.reduced_gravity_events = work_sum[7];
    global_work.reduced_multigroup_events = work_sum[8];
    MPI_Allreduce(MPI_IN_PLACE, &global_work.events, 1,
        MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_work.minimum_bin, 1, MPI_UNSIGNED,
        MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_work.maximum_bin, 1, MPI_UNSIGNED,
        MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_work.time_bin_mask, 1,
        MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
    unsigned long long amr_sum[2] = {global_amr.refined,
        global_amr.derefined};
    MPI_Allreduce(MPI_IN_PLACE, amr_sum, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
        MPI_COMM_WORLD);
    global_amr.refined = amr_sum[0];
    global_amr.derefined = amr_sum[1];
    MPI_Allreduce(MPI_IN_PLACE, &global_amr.calls, 1,
        MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &radiation_retries, 1,
        MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif

    PrintIndividualRestartFingerprint(simulation, "final");
    PrintGravityPolicyDiagnostic(simulation, gravity_policy_baseline);
    double const postprocess_start = WallSeconds();
    WriteSimulation(simulation, output_dir_new + "/final_state.h5");
    WriteBenchmarkVTK(simulation, output_dir_new + "/final_state.vtu");
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    double postprocess_elapsed_max = WallSeconds() - postprocess_start;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &postprocess_elapsed_max, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD);
#endif

    if(rank == 0)
    {
        std::ofstream profile_out(output_dir_new + "/radial_profile.txt");
        profile_out << "# radius density pressure radial_velocity temperature radiation_energy_density\n";
        for(size_t bin = 0; bin < profile_bins; ++bin)
        {
            size_t const offset = bin * profile_fields;
            double const volume = profile_global[offset];
            if(volume <= 0)
                continue;
            profile_out << std::setprecision(17)
                << profile_global[offset + 1] / volume << ' '
                << profile_global[offset + 2] / volume << ' '
                << profile_global[offset + 3] / volume << ' '
                << profile_global[offset + 4] / volume << ' '
                << profile_global[offset + 5] / volume << ' '
                << profile_global[offset + 6] / volume << '\n';
        }

        std::ofstream spectrum_out(output_dir_new + "/spectrum.txt");
        spectrum_out << "# group lower_eV center_eV upper_eV integrated_energy\n";
        for(size_t group = 0; group < spectrum_global.size(); ++group)
            spectrum_out << std::setprecision(17) << group << ' '
                << radiation_grid.boundaries[group] << ' '
                << radiation_grid.centers[group] << ' '
                << radiation_grid.boundaries[group + 1] << ' '
                << spectrum_global[group] << '\n';

        std::ofstream history_out(output_dir_new + "/history.txt");
        history_out << "# time cycle cells active_fraction mass px py pz material_energy radiation_energy peak_mach peak_outward_mach maximum_compression shock_radius nongrav_energy center_of_mass_x center_of_mass_y center_of_mass_z\n";
        for(HistoryRecord const& record : history)
            history_out << std::setprecision(17) << record.time << ' '
                << record.cycle << ' ' << record.cells << ' '
                << record.active_fraction << ' ' << record.mass << ' '
                << record.momentum.x << ' ' << record.momentum.y << ' '
                << record.momentum.z << ' ' << record.material_energy << ' '
                << record.radiation_energy << ' ' << record.peak_mach << ' '
                << record.peak_outward_mach << ' '
                << record.maximum_compression << ' '
                << record.shock_radius << ' ' << record.nongrav_energy << ' '
                << record.center_of_mass.x << ' ' << record.center_of_mass.y
                << ' ' << record.center_of_mass.z << '\n';

        std::ofstream timing_out(output_dir_new + "/timing.txt");
        timing_out << std::setprecision(17)
            << "setup_wall_max " << setup_elapsed_max << '\n'
            << "initial_snapshot_wall_max "
            << initial_snapshot_elapsed_max << '\n'
            << "evolution_wall_max " << evolution_elapsed_max << '\n'
            << "hydro_phase_max " << phase_hydro_max << '\n'
            << "radiation_phase_max " << phase_radiation_max << '\n'
            << "amr_phase_max " << phase_amr_max << '\n'
            << "postprocess_wall_max " << postprocess_elapsed_max << '\n';

        std::ofstream counters_out(output_dir_new + "/counters.txt");
        unsigned int distinct_time_bins = 0;
        for(unsigned long long mask = global_work.time_bin_mask; mask != 0;
            mask >>= 1)
            distinct_time_bins += static_cast<unsigned int>(mask & 1ULL);
        counters_out << std::setprecision(17)
            << "events " << global_work.events << '\n'
            << "active_cell_updates " << global_work.active_cell_updates << '\n'
            << "owned_cell_events " << global_work.owned_cell_events << '\n'
            << "active_faces " << global_work.active_faces << '\n'
            << "mesh_closure_cells " << global_work.mesh_closure_cells << '\n'
            << "gravity_targets " << global_work.gravity_targets << '\n'
            << "multigroup_rows " << global_work.multigroup_rows << '\n'
            << "partial_mesh_rank_events " << global_work.partial_mesh_events << '\n'
            << "reduced_gravity_rank_events " << global_work.reduced_gravity_events << '\n'
            << "reduced_multigroup_rank_events " << global_work.reduced_multigroup_events << '\n'
            << "minimum_time_bin " << global_work.minimum_bin << '\n'
            << "maximum_time_bin " << global_work.maximum_bin << '\n'
            << "distinct_time_bins " << distinct_time_bins << '\n'
            << "amr_calls " << global_amr.calls << '\n'
            << "refined_cells " << global_amr.refined << '\n'
            << "derefined_cells " << global_amr.derefined << '\n'
            << "radiation_retries " << radiation_retries << '\n'
            << "minimum_radiation_candidate_fraction "
            << minimum_radiation_candidate_fraction << '\n'
            << std::setprecision(
                   std::numeric_limits<long double>::max_digits10)
            << "radiation_defect_cumulative_signed_extent "
            << radiation_defect.cumulative_signed_extent << '\n'
            << "radiation_defect_cumulative_absolute_extent "
            << radiation_defect.cumulative_absolute_extent << '\n'
            << "radiation_defect_initial_positive_global_extent "
            << radiation_defect.initial_positive_global_extent << '\n'
            << "radiation_defect_last_normalization_scale "
            << radiation_defect.last_normalization_scale << '\n'
            << "radiation_defect_maximum_event_absolute_fraction "
            << radiation_defect.maximum_event_absolute_fraction << '\n'
            << "radiation_defect_maximum_local_fraction "
            << radiation_defect.maximum_local_fraction << '\n'
            << "radiation_defect_maximum_local_tolerance_ratio "
            << radiation_defect.maximum_local_tolerance_ratio << '\n'
            << "radiation_defect_accepted_dirichlet_candidates "
            << radiation_defect.accepted_dirichlet_candidates << '\n'
            << "radiation_defect_rejections "
            << radiation_defect.defect_rejections << '\n'
            << "radiation_defect_retry_substeps "
            << radiation_defect.defect_retry_substeps << '\n'
            << "radiation_defect_config_version "
            << radiation_defect.config_version << '\n'
            << "radiation_defect_local_withdrawal_limit "
            << radiation_defect.local_withdrawal_limit << '\n'
            << "radiation_defect_local_absolute_limit "
            << radiation_defect.local_absolute_limit << '\n'
            << "radiation_defect_event_absolute_target "
            << radiation_defect.event_absolute_target << '\n'
            << "radiation_defect_cumulative_signed_limit "
            << radiation_defect.cumulative_signed_limit << '\n'
            << "radiation_defect_cumulative_absolute_limit "
            << radiation_defect.cumulative_absolute_limit << '\n'
            << "radiation_defect_cooldown_accepted_candidates "
            << radiation_defect.cooldown_accepted_candidates << '\n'
            << "radiation_defect_cooldown_required_candidates "
            << radiation_defect.cooldown_required_candidates << '\n'
            << "radiation_defect_cooldown_fraction_ceiling "
            << radiation_defect.cooldown_fraction_ceiling << '\n'
            << "radiation_defect_history_complete "
            << (radiation_defect.history_complete ? 1 : 0) << '\n';

        double peak_mach = 0;
        double peak_outward_mach = 0;
        double peak_compression = 0;
        double maximum_shock_radius = 0;
        std::vector<double> active_fractions;
        for(HistoryRecord const& record : history)
        {
            peak_mach = std::max(peak_mach, record.peak_mach);
            peak_outward_mach = std::max(peak_outward_mach,
                record.peak_outward_mach);
            peak_compression = std::max(peak_compression,
                record.maximum_compression);
            maximum_shock_radius = std::max(maximum_shock_radius,
                record.shock_radius);
            if(record.time > 0)
                active_fractions.push_back(record.active_fraction);
        }
        std::sort(active_fractions.begin(), active_fractions.end());
        double const median_active_fraction = active_fractions.empty() ? 1.0 :
            active_fractions[active_fractions.size() / 2];
        double const mean_event_active_fraction =
            global_work.owned_cell_events > 0 ?
            static_cast<double>(global_work.active_cell_updates) /
                static_cast<double>(global_work.owned_cell_events) : 1.0;
        double const mass_drift = initial_history.mass > 0 ?
            std::abs(final_history.mass - initial_history.mass) /
                initial_history.mass : 0;
        double const momentum_scale = M * R / dynamical_time;
        double const momentum_drift = momentum_scale > 0 ?
            abs(final_history.momentum - initial_history.momentum) /
                momentum_scale : 0;
        double const nongrav_energy_drift = initial_history.nongrav_energy != 0 ?
            std::abs(final_history.nongrav_energy -
                     initial_history.nongrav_energy) /
                std::abs(initial_history.nongrav_energy) : 0;
        double const normalized_center_of_mass = R > 0 ?
            abs(final_history.center_of_mass) / R : 0;
        size_t significant_groups = 0;
        double const spectrum_total = std::accumulate(spectrum_global.begin(),
            spectrum_global.end(), 0.0);
        for(double energy : spectrum_global)
            if(energy > 1e-12 * spectrum_total)
                ++significant_groups;
        double shock_compression_ratio = 0;
        if(final_history.shock_radius > 0)
        {
            size_t shock_bin = static_cast<size_t>(profile_bins *
                final_history.shock_radius / (1.5 * R));
            shock_bin = std::min(shock_bin, profile_bins - 1);
            double peak_density = 0;
            size_t const peak_begin = shock_bin > 4 ? shock_bin - 4 : 0;
            size_t const peak_end = std::min(profile_bins - 1, shock_bin + 4);
            for(size_t bin = peak_begin; bin <= peak_end; ++bin)
            {
                size_t const offset = bin * profile_fields;
                if(profile_global[offset] > 0)
                    peak_density = std::max(peak_density,
                        profile_global[offset + 2] / profile_global[offset]);
            }
            double upstream_density = 0;
            size_t upstream_samples = 0;
            for(size_t bin = shock_bin + 5;
                bin < std::min(profile_bins, shock_bin + 13); ++bin)
            {
                size_t const offset = bin * profile_fields;
                if(profile_global[offset] > 0)
                {
                    upstream_density += profile_global[offset + 2] /
                        profile_global[offset];
                    ++upstream_samples;
                }
            }
            if(upstream_samples > 0 && upstream_density > 0)
                shock_compression_ratio = peak_density /
                    (upstream_density / upstream_samples);
        }

        double const effective_final_time = has_terminal_target ?
            terminal_target.represented_time : final_time;
        std::ofstream metrics_out(output_dir_new + "/metrics.txt");
        metrics_out << std::setprecision(17)
            << "final_time " << final_history.time << '\n'
            << "target_final_time " << final_time << '\n'
            << "effective_target_final_time " << effective_final_time << '\n'
            << "requested_final_time_error "
            << final_history.time - final_time << '\n'
            << "terminal_tick "
            << (has_terminal_target ? terminal_target.tick : 0) << '\n'
            << "terminal_time_origin "
            << (has_terminal_target ? terminal_target.time_origin : 0) << '\n'
            << "terminal_time_quantum "
            << (has_terminal_target ? terminal_target.time_quantum : 0) << '\n'
            << "final_cells " << final_history.cells << '\n'
            << "mass_drift " << mass_drift << '\n'
            << "normalized_momentum_drift " << momentum_drift << '\n'
            << "nongrav_energy_drift " << nongrav_energy_drift << '\n'
            << "normalized_center_of_mass " << normalized_center_of_mass
            << '\n'
            << "minimum_material_energy " << minimum_material_energy << '\n'
            << "minimum_group_energy " << minimum_group_energy << '\n'
            << "minimum_fleck_factor " << minimum_fleck << '\n'
            << "maximum_fleck_factor " << maximum_fleck << '\n'
            << "fleck_samples " << fleck_samples << '\n'
            << "maximum_transport_scattering "
            << maximum_transport_scattering << '\n'
            << "significant_groups " << significant_groups << '\n'
            << "peak_mach " << peak_mach << '\n'
            << "peak_outward_mach " << peak_outward_mach << '\n'
            << "peak_compression_indicator " << peak_compression << '\n'
            << "maximum_compression_ratio " << maximum_compression_ratio << '\n'
            << "shock_compression_ratio " << shock_compression_ratio << '\n'
            << "shock_radius " << final_history.shock_radius << '\n'
            << "maximum_shock_radius " << maximum_shock_radius << '\n'
            << "median_active_fraction " << median_active_fraction << '\n'
            << "mean_event_active_fraction " << mean_event_active_fraction
            << '\n'
            << "time_bin_ratio "
            << std::ldexp(1.0, static_cast<int>(global_work.maximum_bin) -
                static_cast<int>(global_work.minimum_bin)) << '\n';

        std::ofstream manifest_out(output_dir_new + "/manifest.txt");
        manifest_out << std::setprecision(17)
            << "case lane_radiation_shock_individual\n"
            << "mode " << run_mode << '\n'
            << "mpi_ranks " << ws << '\n'
            << "requested_initial_cells " << np << '\n'
            << "energy_groups " << radiation_grid.centers.size() << '\n'
            << "mg_preconditioner "
            << CG::PreconditionerKindLabel(mg_preconditioner) << '\n'
            << "energy_min_eV " << radiation_grid.boundaries.front() << '\n'
            << "energy_max_eV " << radiation_grid.boundaries.back() << '\n'
            << "mass_g " << M << '\n'
            << "radius_cm " << R << '\n'
            << "initial_temperature_floor_K " << initial_temperature_floor << '\n'
            << "pulse_peak_temperature_K " << pulse_peak_temperature << '\n'
            << "pulse_radius_fraction " << pulse_radius_fraction << '\n'
            << "pulse_energy_erg " << pulse_energy << '\n'
            << "initial_dt " << initial_dt << '\n'
            << "time_quantum " << time_quantum << '\n'
            << "maximum_global_dt " << maximum_global_dt << '\n'
            << "maximum_individual_dt " << maximum_individual_dt << '\n'
            << "requested_initial_bin " << requested_initial_bin << '\n'
            << "requested_maximum_bin " << requested_maximum_bin << '\n'
            << "gravity barnes_hut_quadrupole_theta_0.7\n"
            << "compton 1\n"
            << "doppler 1\n"
            << "flux_limiter 1\n"
            << "hydro_feedback 1\n"
            << "protections 1\n"
            << "cooling_limiter 1\n"
            << "radiation_defect_config_version "
            << radiation_defect.config_version << '\n'
            << "radiation_defect_local_withdrawal_limit "
            << radiation_defect.local_withdrawal_limit << '\n'
            << "radiation_defect_local_absolute_limit "
            << radiation_defect.local_absolute_limit << '\n'
            << "radiation_defect_event_absolute_target "
            << radiation_defect.event_absolute_target << '\n'
            << "radiation_defect_cumulative_signed_limit "
            << radiation_defect.cumulative_signed_limit << '\n'
            << "radiation_defect_cumulative_absolute_limit "
            << radiation_defect.cumulative_absolute_limit << '\n'
            << "radiation_defect_cooldown_required_candidates "
            << radiation_defect.cooldown_required_candidates << '\n'
            << "initial_checksum_xor " << std::hex << initial_checksum[0]
            << '\n' << "initial_checksum_sum " << initial_checksum[1]
            << std::dec << '\n';

        std::cout << "BENCHMARK_RESULT mode=" << run_mode
            << " mg_preconditioner="
            << CG::PreconditionerKindLabel(mg_preconditioner)
            << " evolution_wall_max=" << evolution_elapsed_max
            << " final_time=" << final_history.time
            << " cells=" << final_history.cells << std::endl;
    }


#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
