#include "IndividualTimeStep.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace
{
std::uint64_t checked_add(std::uint64_t left, std::uint64_t right)
{
    if(right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("Individual timestep timeline overflow");
    return left + right;
}

#ifdef RICH_MPI
struct NeighborBinRequest : public Serializable
{
    std::size_t cell_id = 0;
    std::uint8_t maximum_bin = 0;
    std::uint64_t maximum_end_tick = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(cell_id);
        bytes += serializer->insert(maximum_bin);
        bytes += serializer->insert(maximum_end_tick);
        return bytes;
    }

    force_inline std::size_t load(const Serializer* serializer,
                                  std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(cell_id, byte_offset + bytes);
        bytes += serializer->extract(maximum_bin, byte_offset + bytes);
        bytes += serializer->extract(maximum_end_tick, byte_offset + bytes);
        return bytes;
    }
};

template<typename T>
std::vector<T> SyncCanonicalDataToMesh(
    const Tessellation3D& tess,
    const Tessellation3D::AllPointsMap& local_to_global,
    const std::vector<T>& canonical)
{
    std::vector<T> local(tess.getMeshPoints().size());
    bool mapping_valid = true;
    for(auto const& mapping : local_to_global)
    {
        if(mapping.first >= local.size() || mapping.second >= canonical.size())
        {
            mapping_valid = false;
            continue;
        }
        local[mapping.first] = canonical[mapping.second];
    }
    int mapping_valid_int = mapping_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &mapping_valid_int, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(mapping_valid_int == 0)
        throw std::logic_error(
            "Individual timestep canonical-to-mesh map is invalid on at least one rank");
    std::vector<T> canonical_copy = canonical;
    tess.SyncPartialBuildData(local, canonical_copy);
    return local;
}
#endif
}

#ifdef RICH_MPI
std::size_t CellTimeState::dump(Serializer* serializer) const
{
    std::size_t bytes = 0;
    bytes += serializer->insert(cell_id);
    bytes += serializer->insert(begin_tick);
    bytes += serializer->insert(end_tick);
    bytes += serializer->insert(last_primitive_tick);
    bytes += serializer->insert(time_bin);
    bytes += serializer->insert(point_velocity);
    bytes += serializer->insert(cached_acceleration);
    const std::uint8_t gravity_phase = gravity_half_kick_pending ? 1 : 0;
    bytes += serializer->insert(gravity_phase);
    return bytes;
}

std::size_t CellTimeState::load(const Serializer* serializer,
                                std::size_t byte_offset)
{
    std::size_t bytes = 0;
    bytes += serializer->extract(cell_id, byte_offset + bytes);
    bytes += serializer->extract(begin_tick, byte_offset + bytes);
    bytes += serializer->extract(end_tick, byte_offset + bytes);
    bytes += serializer->extract(last_primitive_tick, byte_offset + bytes);
    bytes += serializer->extract(time_bin, byte_offset + bytes);
    bytes += serializer->extract(point_velocity, byte_offset + bytes);
    bytes += serializer->extract(cached_acceleration, byte_offset + bytes);
    std::uint8_t gravity_phase = 0;
    bytes += serializer->extract(gravity_phase, byte_offset + bytes);
    gravity_half_kick_pending = gravity_phase != 0;
    return bytes;
}
#endif

bool AdvanceRadiationRepairWarning(
    RadiationRepairAccounting& accounting, double const injection_fraction)
{
    bool const warn = std::isfinite(injection_fraction) &&
        injection_fraction >= accounting.next_warning_fraction;
    if(!warn)
        return false;
    while(accounting.next_warning_fraction <= injection_fraction) {
        double const next = 2 * accounting.next_warning_fraction;
        if(!std::isfinite(next) || next <= accounting.next_warning_fraction) {
            accounting.next_warning_fraction =
                std::numeric_limits<double>::max();
            break;
        }
        accounting.next_warning_fraction = next;
    }
    return true;
}

bool IndividualStepContext::isActive(std::size_t index) const
{
    return index < active_mask.size() && active_mask[index] != 0;
}

double IndividualStepContext::cellTimeStep(std::size_t index) const
{
    if(index >= cell_time_steps.size())
        throw std::out_of_range("IndividualStepContext cell index is out of range");
    return cell_time_steps[index];
}

double IndividualStepContext::faceTimeStep(std::size_t left, std::size_t right) const
{
    return std::min(cellTimeStep(left), cellTimeStep(right));
}

IndividualTimeStepScheduler::IndividualTimeStepScheduler(IndividualTimeStepOptions options)
    : options_(options)
{
    if(options_.initial_bin > options_.maximum_bin)
        throw std::invalid_argument("Individual timestep initial bin exceeds maximum bin");
    if(options_.maximum_bin > 62)
        throw std::invalid_argument("Individual timestep maximum bin must not exceed 62");
    if(!(options_.partial_build_fraction > 0 && options_.partial_build_fraction <= 1))
        throw std::invalid_argument("Partial build fraction must be in (0, 1]");
}

void IndividualTimeStepScheduler::initialize(const std::vector<ComputationalCell3D> &cells,
                                             double current_time,
                                             double initial_time_step)
{
    if(initialized_)
        throw std::logic_error("Individual timestep scheduler is already initialized");
    if(!(std::isfinite(initial_time_step) && initial_time_step > 0))
        throw std::invalid_argument("Individual timestep initialization requires a positive finite timestep");

    time_quantum_ = options_.time_quantum;
    if(!(time_quantum_ > 0))
        time_quantum_ = std::ldexp(initial_time_step, -static_cast<int>(options_.initial_bin));
    if(!(std::isfinite(time_quantum_) && time_quantum_ > 0))
        throw std::invalid_argument("Individual timestep quantum must be positive and finite");

    time_origin_ = current_time;
    current_tick_ = 0;
    states_.resize(cells.size());
    const std::uint64_t initial_ticks = ticksForBin(options_.initial_bin);
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        CellTimeState &state = states_[i];
        state.cell_id = cells[i].ID;
        state.begin_tick = 0;
        state.end_tick = initial_ticks;
        state.last_primitive_tick = 0;
        state.time_bin = options_.initial_bin;
        state.point_velocity = Vector3D();
        state.cached_acceleration = Vector3D();
        state.gravity_half_kick_pending = false;
    }
    rebuildIndex(cells);
    initialized_ = true;
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells) const
{
    if(!initialized_)
        throw std::logic_error("Individual timestep scheduler is not initialized");
    validateCells(cells);
    if(states_.empty())
        throw std::logic_error("Individual timestep scheduler has no cells");

    return prepareEvent(cells, nextEventTick());
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick) const
{
    return prepareEvent(cells, event_tick, false);
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick,
    bool force_all_active) const
{
    if(!initialized_)
        throw std::logic_error("Individual timestep scheduler is not initialized");
    validateCells(cells);
    if(event_tick <= current_tick_)
        throw std::logic_error("Individual timestep scheduler produced a non-advancing event");
    const std::uint64_t local_next = nextEventTick();
    if(event_tick > local_next)
        throw std::logic_error("Collective individual event skipped a local cell end tick");

    IndividualStepContext result;
    result.previous_event_tick = current_tick_;
    result.event_tick = event_tick;
    result.time_origin = time_origin_;
    result.previous_event_time = time_origin_ + time_quantum_ * static_cast<double>(current_tick_);
    result.event_time = time_origin_ + time_quantum_ * static_cast<double>(event_tick);
    result.time_quantum = time_quantum_;
    result.mesh_build_policy = options_.mesh_build_policy;
    result.partial_build_fraction = options_.partial_build_fraction;
    result.verify_partial_build = options_.verify_partial_build;
    result.radiation_repair_accounting = &radiation_repair_accounting_;
    result.radiation_defect_accounting = &radiation_defect_accounting_;
    result.active_mask.assign(states_.size(), 0);
    result.cell_time_steps.resize(states_.size());
    result.primitive_ticks.resize(states_.size());
    result.point_velocities.resize(states_.size());
    result.cached_accelerations.resize(states_.size());
    result.gravity_half_kick_pending.resize(states_.size());

    for(std::size_t i = 0; i < states_.size(); ++i)
    {
        const CellTimeState &state = states_[i];
        result.point_velocities[i] = state.point_velocity;
        result.cached_accelerations[i] = state.cached_acceleration;
        result.gravity_half_kick_pending[i] = state.gravity_half_kick_pending ? 1 : 0;
        if(force_all_active &&
           !(state.begin_tick < event_tick && event_tick <= state.end_tick))
            throw std::logic_error(
                "Forced individual event lies outside a cell's scheduled interval");
        const std::uint64_t step_end_tick = force_all_active ?
            event_tick : state.end_tick;
        result.cell_time_steps[i] = time_quantum_ *
            static_cast<double>(step_end_tick - state.begin_tick);
        result.primitive_ticks[i] = state.last_primitive_tick;
        if(force_all_active || state.end_tick == event_tick)
        {
            result.active_indices.push_back(i);
            result.active_mask[i] = 1;
        }
    }
    return result;
}

void IndividualTimeStepScheduler::commitEvent(
    const IndividualStepContext &context,
    const Tessellation3D &tess,
    const std::vector<ComputationalCell3D> &cells,
    const std::vector<double> &time_step_limits)
{
    validateCells(cells);
    if(context.previous_event_tick != current_tick_)
        throw std::logic_error("Individual timestep event was prepared from stale scheduler state");
    if(time_step_limits.size() != states_.size())
        throw std::invalid_argument("Individual timestep limit vector has the wrong size");
    if(context.point_velocities.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].point_velocity = context.point_velocities[i];
    if(context.cached_accelerations.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].cached_acceleration = context.cached_accelerations[i];
    if(context.gravity_half_kick_pending.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].gravity_half_kick_pending =
                context.gravity_half_kick_pending[i] != 0;

    if(options_.force_synchronized)
    {
        if(context.active_indices.size() != states_.size())
            throw std::logic_error(
                "Synchronized individual event did not activate every cell");
        double shared_limit = std::numeric_limits<double>::infinity();
        for(double limit : time_step_limits)
            shared_limit = std::min(shared_limit, limit);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &shared_limit, 1, MPI_DOUBLE, MPI_MIN,
            MPI_COMM_WORLD);
#endif
        std::uint8_t shared_bin = options_.maximum_bin;
        if(!states_.empty())
            shared_bin = chooseNextBin(states_.front(), context.event_tick,
                shared_limit);
#ifdef RICH_MPI
        unsigned int shared_bin_value = shared_bin;
        MPI_Allreduce(MPI_IN_PLACE, &shared_bin_value, 1, MPI_UNSIGNED,
            MPI_MIN, MPI_COMM_WORLD);
        shared_bin = static_cast<std::uint8_t>(shared_bin_value);
#endif
        for(CellTimeState& state : states_)
        {
            state.begin_tick = context.event_tick;
            state.last_primitive_tick = context.event_tick;
            state.time_bin = shared_bin;
            state.end_tick = checked_add(context.event_tick,
                ticksForBin(shared_bin));
        }
        current_tick_ = context.event_tick;
        return;
    }

    for(std::size_t index : context.active_indices)
    {
        if(index >= states_.size())
            throw std::out_of_range("Active individual timestep cell is out of range");
        CellTimeState &state = states_[index];
        CellTimeState completed_state = state;
        if(state.end_tick != context.event_tick)
        {
            if(state.begin_tick >= context.event_tick ||
               context.event_tick > state.end_tick)
                throw std::logic_error(
                    "Forced individual cell committed outside its scheduled interval");
            std::uint64_t const completed_ticks =
                context.event_tick - state.begin_tick;
            std::uint8_t completed_bin = 0;
            while(completed_bin < options_.maximum_bin &&
                  ticksForBin(static_cast<std::uint8_t>(completed_bin + 1)) <=
                      completed_ticks)
                ++completed_bin;
            completed_state.time_bin = completed_bin;
        }
        std::uint8_t const next_bin = chooseNextBin(
            completed_state, context.event_tick, time_step_limits[index]);
        state.begin_tick = context.event_tick;
        state.last_primitive_tick = context.event_tick;
        state.time_bin = next_bin;
        state.end_tick = checked_add(context.event_tick, ticksForBin(state.time_bin));
    }

    for(std::size_t index = 0; index < states_.size(); ++index)
    {
        if(context.isActive(index))
            continue;
        double const limit = time_step_limits[index];
        if(std::isinf(limit) && limit > 0)
            continue;
        std::uint8_t const limited_bin = quantizeTimeStep(limit);
        std::uint64_t const limited_end = nextAlignedTick(
            context.event_tick, limited_bin);
        if(limited_end < states_[index].end_tick)
        {
            states_[index].time_bin = std::min(states_[index].time_bin,
                                               limited_bin);
            states_[index].end_tick = limited_end;
        }
    }

    limitNeighborBins(tess, cells, context.event_tick);
    current_tick_ = context.event_tick;
}

std::size_t IndividualTimeStepScheduler::clampToTerminalTick(
    std::uint64_t terminal_tick)
{
    if(!initialized_)
        throw std::logic_error(
            "Cannot clamp an uninitialized individual timestep scheduler");
    if(terminal_tick <= current_tick_)
        throw std::invalid_argument(
            "Individual timestep terminal tick must follow the current tick");

    std::size_t clamped = 0;
    for(CellTimeState& state : states_)
    {
        if(state.end_tick > terminal_tick)
        {
            state.end_tick = terminal_tick;
            ++clamped;
        }
    }
    return clamped;
}

double IndividualTimeStepScheduler::nextEventTimeStep(void) const
{
    if(!initialized_)
        return 0;
    const std::uint64_t next_tick = nextEventTick();
    if(next_tick == std::numeric_limits<std::uint64_t>::max())
        return std::numeric_limits<double>::infinity();
    if(next_tick <= current_tick_)
        return 0;
    return time_quantum_ * static_cast<double>(next_tick - current_tick_);
}

std::uint64_t IndividualTimeStepScheduler::nextEventTick(void) const
{
    if(!initialized_)
        return std::numeric_limits<std::uint64_t>::max();
    std::uint64_t next_tick = std::numeric_limits<std::uint64_t>::max();
    for(const CellTimeState &state : states_)
        next_tick = std::min(next_tick, state.end_tick);
    return next_tick;
}

void IndividualTimeStepScheduler::resetTimeOrigin(double current_time)
{
    if(initialized_ && current_tick_ != 0)
        throw std::logic_error("Cannot reset an active individual timestep timeline");
    time_origin_ = current_time;
}

void IndividualTimeStepScheduler::rebuildIndex(const std::vector<ComputationalCell3D> &cells)
{
    if(states_.size() != cells.size())
        throw std::invalid_argument("Individual timestep state and cell counts differ");
    id_to_index_.clear();
    id_to_index_.reserve(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        if(!id_to_index_.emplace(cells[i].ID, i).second)
            throw std::invalid_argument("Duplicate cell ID in individual timestep scheduler");
        states_[i].cell_id = cells[i].ID;
    }
}

void IndividualTimeStepScheduler::restore(
    const std::vector<ComputationalCell3D>& cells,
    double time_origin,
    double time_quantum,
    std::uint64_t current_tick,
    std::vector<CellTimeState> states,
    RadiationRepairAccounting repair_accounting,
    IndividualRadiationDefectAccounting defect_accounting,
    bool force_all_active_latched)
{
    if(initialized_)
        throw std::logic_error("Cannot restore over an initialized individual timestep scheduler");
    if(!(std::isfinite(time_origin) && std::isfinite(time_quantum) &&
         time_quantum > 0))
        throw std::invalid_argument("Invalid individual timestep restart clock");
    if(states.size() != cells.size())
        throw std::invalid_argument("Individual timestep restart state count differs from cells");
    bool const residual_bias_vectors_valid =
        repair_accounting.residual_correction_signed_bias_by_group.size() ==
            repair_accounting.
                residual_correction_absolute_bias_by_group.size() &&
        std::all_of(
            repair_accounting.residual_correction_signed_bias_by_group.begin(),
            repair_accounting.residual_correction_signed_bias_by_group.end(),
            [](double const value) { return std::isfinite(value); }) &&
        std::all_of(
            repair_accounting.
                residual_correction_absolute_bias_by_group.begin(),
            repair_accounting.
                residual_correction_absolute_bias_by_group.end(),
            [](double const value)
            { return std::isfinite(value) && value >= 0; });
    if(!std::isfinite(repair_accounting.cumulative_injected_energy) ||
       repair_accounting.cumulative_injected_energy < 0 ||
       !std::isfinite(repair_accounting.maximum_relative_deficit) ||
       repair_accounting.maximum_relative_deficit < 0 ||
       !std::isfinite(repair_accounting.maximum_global_radiation_energy) ||
       repair_accounting.maximum_global_radiation_energy < 0 ||
       !std::isfinite(repair_accounting.next_warning_fraction) ||
       repair_accounting.next_warning_fraction < 1e-4 ||
       !std::isfinite(
           repair_accounting.residual_correction_signed_energy_bias) ||
       !std::isfinite(
           repair_accounting.residual_correction_absolute_energy_bias) ||
       repair_accounting.residual_correction_absolute_energy_bias < 0 ||
       !std::isfinite(
           repair_accounting.residual_correction_minimum_scale) ||
       repair_accounting.residual_correction_minimum_scale < 0 ||
       repair_accounting.residual_correction_minimum_scale > 1 ||
       !residual_bias_vectors_valid ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_cumulative_injected_energy) ||
       repair_accounting.
           residual_positive_floor_cumulative_injected_energy < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_cell_injection_ratio) ||
       repair_accounting.
           residual_positive_floor_maximum_cell_injection_ratio < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_global_injection_ratio) ||
       repair_accounting.
           residual_positive_floor_maximum_global_injection_ratio < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_post_true_residual_error) ||
       repair_accounting.
           residual_positive_floor_maximum_post_true_residual_error < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_initial_global_radiation_energy) ||
       repair_accounting.
           residual_positive_floor_initial_global_radiation_energy < 0)
        throw std::invalid_argument(
            "Invalid individual timestep radiation repair accounting");

    long double const defect_balance_scale = std::max(
        std::abs(defect_accounting.cumulative_signed_extent),
        defect_accounting.cumulative_absolute_extent);
    long double const defect_balance_slack =
        64 * std::numeric_limits<long double>::epsilon() *
        defect_balance_scale;
    bool const defect_accounting_valid =
        defect_accounting.config_version == 1 &&
        std::isfinite(
            static_cast<double>(defect_accounting.cumulative_signed_extent)) &&
        std::isfinite(
            static_cast<double>(defect_accounting.cumulative_absolute_extent)) &&
        defect_accounting.cumulative_absolute_extent >= 0 &&
        std::abs(defect_accounting.cumulative_signed_extent) <=
            defect_accounting.cumulative_absolute_extent +
                defect_balance_slack &&
        std::isfinite(defect_accounting.initial_positive_global_extent) &&
        defect_accounting.initial_positive_global_extent >= 0 &&
        std::isfinite(defect_accounting.last_normalization_scale) &&
        defect_accounting.last_normalization_scale >= 0 &&
        std::isfinite(defect_accounting.maximum_event_absolute_fraction) &&
        defect_accounting.maximum_event_absolute_fraction >= 0 &&
        std::isfinite(defect_accounting.maximum_local_fraction) &&
        defect_accounting.maximum_local_fraction >= 0 &&
        std::isfinite(defect_accounting.local_withdrawal_limit) &&
        defect_accounting.local_withdrawal_limit > 0 &&
        std::isfinite(defect_accounting.event_absolute_target) &&
        defect_accounting.event_absolute_target > 0 &&
        std::isfinite(defect_accounting.cumulative_signed_limit) &&
        defect_accounting.cumulative_signed_limit > 0 &&
        std::isfinite(defect_accounting.cumulative_absolute_limit) &&
        defect_accounting.cumulative_absolute_limit > 0 &&
        defect_accounting.cooldown_required_candidates > 0 &&
        std::isfinite(defect_accounting.cooldown_fraction_ceiling) &&
        defect_accounting.cooldown_fraction_ceiling > 0 &&
        defect_accounting.cooldown_fraction_ceiling <= 1;
    if(!defect_accounting_valid)
        throw std::invalid_argument(
            "Invalid individual timestep radiation defect accounting");
    for(std::size_t i = 0; i < states.size(); ++i)
    {
        if(states[i].cell_id != cells[i].ID)
            throw std::invalid_argument("Individual timestep restart IDs are not in canonical cell order");
        if(states[i].time_bin > options_.maximum_bin ||
           states[i].begin_tick > current_tick ||
           states[i].end_tick <= current_tick ||
           states[i].last_primitive_tick > current_tick)
            throw std::invalid_argument("Invalid individual timestep restart tick/bin state");
    }
    time_origin_ = time_origin;
    time_quantum_ = time_quantum;
    current_tick_ = current_tick;
    states_ = std::move(states);
    radiation_repair_accounting_ = repair_accounting;
    radiation_defect_accounting_ = defect_accounting;
    setForceAllActiveLatched(force_all_active_latched);
    rebuildIndex(cells);
    initialized_ = true;
}

void IndividualTimeStepScheduler::applyAMRChangeSet(
    const std::vector<ComputationalCell3D>& cells,
    const IndividualAMRChangeSet& changes,
    const Tessellation3D* tess)
{
    if(!initialized_)
        throw std::logic_error("Cannot remap AMR before individual timestep initialization");

    std::unordered_map<std::size_t, CellTimeState> old_states;
    old_states.reserve(states_.size());
    for(CellTimeState const& state : states_)
        old_states.emplace(state.cell_id, state);
    std::unordered_map<std::size_t, std::size_t> child_to_parent;
    child_to_parent.reserve(changes.child_parent_ids.size());
    for(auto const& entry : changes.child_parent_ids)
        if(!child_to_parent.emplace(entry.first, entry.second).second)
            throw std::invalid_argument("AMR change set contains a duplicate child ID");
    std::unordered_set<std::size_t> removed(
        changes.removed_cell_ids.begin(), changes.removed_cell_ids.end());

    std::vector<CellTimeState> remapped(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        std::size_t const id = cells[i].ID;
        auto existing = old_states.find(id);
        if(existing != old_states.end())
            remapped[i] = existing->second;
        else
        {
            auto parent_id = child_to_parent.find(id);
            if(parent_id == child_to_parent.end())
                throw std::invalid_argument("New AMR cell has no parent timestep state");
            auto parent = old_states.find(parent_id->second);
            if(parent == old_states.end())
                throw std::invalid_argument("AMR child references an unknown parent ID");
            remapped[i] = parent->second;
            remapped[i].cell_id = id;
        }
        if(removed.count(id) != 0)
            throw std::invalid_argument("AMR change set keeps an ID marked as removed");
    }
    states_ = std::move(remapped);
    rebuildIndex(cells);
    if(tess != nullptr)
        limitNeighborBins(*tess, cells, current_tick_);
}

std::uint64_t IndividualTimeStepScheduler::ticksForBin(std::uint8_t bin) const
{
    if(bin > options_.maximum_bin || bin > 62)
        throw std::out_of_range("Individual timestep bin is out of range");
    return std::uint64_t(1) << bin;
}

std::uint64_t IndividualTimeStepScheduler::nextAlignedTick(
    std::uint64_t event_tick, std::uint8_t bin) const
{
    const std::uint64_t interval = ticksForBin(bin);
    const std::uint64_t remainder = event_tick % interval;
    const std::uint64_t increment = remainder == 0 ? interval : interval - remainder;
    return checked_add(event_tick, increment);
}

std::uint8_t IndividualTimeStepScheduler::quantizeTimeStep(double time_step) const
{
    if(!(std::isfinite(time_step) && time_step > 0))
    {
        if(std::isinf(time_step) && time_step > 0)
            return options_.maximum_bin;
        throw std::invalid_argument("Individual timestep limit must be positive and finite");
    }

    const double tick_limit = std::floor(time_step / time_quantum_);
    if(tick_limit < 1)
    {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Individual timestep limit " << time_step
                << " is below the configured time quantum " << time_quantum_;
        throw std::runtime_error(message.str());
    }

    const std::uint64_t ticks = tick_limit >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
        ? std::numeric_limits<std::uint64_t>::max()
        : static_cast<std::uint64_t>(tick_limit);
    std::uint8_t bin = 0;
    while(bin < options_.maximum_bin && ticksForBin(static_cast<std::uint8_t>(bin + 1)) <= ticks)
        ++bin;
    return bin;
}

std::uint8_t IndividualTimeStepScheduler::chooseNextBin(const CellTimeState &state,
                                                        std::uint64_t event_tick,
                                                        double time_step_limit) const
{
    const std::uint8_t desired = quantizeTimeStep(time_step_limit);
    if(desired <= state.time_bin)
        return desired;
    if(state.time_bin >= options_.maximum_bin)
        return state.time_bin;

    const std::uint8_t candidate = static_cast<std::uint8_t>(state.time_bin + 1);
    const std::uint64_t candidate_ticks = ticksForBin(candidate);
    if(event_tick % candidate_ticks == 0)
        return candidate;
    return state.time_bin;
}

void IndividualTimeStepScheduler::validateCells(const std::vector<ComputationalCell3D> &cells) const
{
    if(cells.size() != states_.size())
        throw std::logic_error("Cell count changed without remapping individual timestep state");
    for(std::size_t i = 0; i < cells.size(); ++i)
        if(cells[i].ID != states_[i].cell_id)
            throw std::logic_error("Cell order changed without remapping individual timestep state");
}

void IndividualTimeStepScheduler::limitNeighborBins(
    const Tessellation3D &tess,
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick)
{
#ifdef RICH_MPI
    int any_rank_has_cells = states_.empty() ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &any_rank_has_cells, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if(any_rank_has_cells == 0)
        return;
#else
    if(states_.empty())
        return;
#endif

    std::vector<std::size_t> neighbors;
    Tessellation3D::AllPointsMap const& local_to_global =
        tess.GetIndicesInAllPoints();
#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    const std::size_t mesh_size = tess.getMeshPoints().size();
    std::vector<unsigned char> required_mesh_point(mesh_size, 0);
    for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
    {
        if(local_i < mesh_size)
            required_mesh_point[local_i] = 1;
        neighbors.clear();
        tess.GetNeighbors(local_i, neighbors);
        for(std::size_t neighbor : neighbors)
            if(neighbor < mesh_size && !tess.IsPointOutsideBox(neighbor))
                required_mesh_point[neighbor] = 1;
    }
    std::vector<int> point_owner(mesh_size, -1);
    unsigned local_mapping_issues =
        local_to_global.size() < tess.GetPointNo() ? 1u : 0u;
    for(auto const& mapping : local_to_global)
    {
        if(mapping.first >= mesh_size || mapping.second >= states_.size())
        {
            local_mapping_issues |= 2u;
            continue;
        }
        point_owner[mapping.first] = rank;
    }
    const std::vector<int> duplicated_procs = tess.GetDuplicatedProcs();
    const std::vector<std::vector<std::size_t> >& ghost_indices =
        tess.GetGhostIndeces();
    if(duplicated_procs.size() != ghost_indices.size())
        local_mapping_issues |= 4u;
    for(std::size_t peer = 0;
        peer < duplicated_procs.size() && peer < ghost_indices.size(); ++peer)
    {
        if(duplicated_procs[peer] < 0 || duplicated_procs[peer] >= rank_count)
            local_mapping_issues |= 8u;
        for(std::size_t ghost : ghost_indices[peer])
        {
            if(ghost >= point_owner.size())
            {
                local_mapping_issues |= 16u;
                continue;
            }
            // A mesh slot can occur in more than one incoming list after a
            // many-rank rebuild.  SyncPartialBuildData writes those lists in
            // this same order, so the final peer is the owner of the value
            // currently stored in the synchronized mesh slot.
            point_owner[ghost] = duplicated_procs[peer];
        }
    }
    for(std::size_t local = 0; local < mesh_size; ++local)
        if(required_mesh_point[local] != 0 && point_owner[local] < 0)
            local_mapping_issues |= 32u;

    std::vector<std::size_t> canonical_ids(states_.size());
    for(std::size_t i = 0; i < states_.size(); ++i)
        canonical_ids[i] = states_[i].cell_id;
    const std::vector<std::size_t> mesh_ids = SyncCanonicalDataToMesh(
        tess, local_to_global, canonical_ids);
    const std::vector<unsigned char> canonical_valid(states_.size(), 1);
    const std::vector<unsigned char> mesh_valid = SyncCanonicalDataToMesh(
        tess, local_to_global, canonical_valid);
    for(std::size_t local = 0; local < mesh_size; ++local)
        if(required_mesh_point[local] != 0 && mesh_valid[local] == 0)
            local_mapping_issues |= 64u;

    unsigned mapping_issues = local_mapping_issues;
    MPI_Allreduce(MPI_IN_PLACE, &mapping_issues, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if(mapping_issues != 0)
    {
        throw std::logic_error(
            "Individual timestep MPI neighbor ownership mapping is inconsistent "
            "(issue mask " + std::to_string(mapping_issues) + ")");
    }

    bool globally_changed = true;
    unsigned iteration = 0;
    while(globally_changed)
    {
        if(iteration++ > static_cast<unsigned>(options_.maximum_bin) + 2)
            throw std::logic_error(
                "Individual timestep MPI neighbor-bin limiter failed to converge");

        std::vector<std::uint8_t> canonical_bins(states_.size());
        for(std::size_t i = 0; i < states_.size(); ++i)
            canonical_bins[i] = states_[i].time_bin;
        const std::vector<std::uint8_t> mesh_bins = SyncCanonicalDataToMesh(
            tess, local_to_global, canonical_bins);

        bool locally_changed = false;
        bool local_request_error = false;
        std::vector<std::vector<NeighborBinRequest> > outgoing(rank_count);
        for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
        {
            auto const mapped_i = local_to_global.find(local_i);
            if(mapped_i == local_to_global.end() ||
               mapped_i->second >= states_.size())
            {
                local_request_error = true;
                continue;
            }
            const std::size_t i = mapped_i->second;
            neighbors.clear();
            tess.GetNeighbors(local_i, neighbors);
            for(std::size_t local_neighbor : neighbors)
            {
                if(local_neighbor >= mesh_size ||
                   tess.IsPointOutsideBox(local_neighbor))
                    continue;
                const std::uint8_t fine_bin = std::min(
                    states_[i].time_bin, mesh_bins[local_neighbor]);
                const unsigned limited = std::min<unsigned>(
                    options_.maximum_bin,
                    static_cast<unsigned>(fine_bin) +
                        options_.maximum_neighbor_bin_difference);
                const std::uint8_t limited_bin =
                    static_cast<std::uint8_t>(limited);
                const std::uint64_t limited_end = nextAlignedTick(
                    event_tick, limited_bin);

                CellTimeState& local_state = states_[i];
                if(local_state.time_bin > limited_bin)
                {
                    local_state.time_bin = limited_bin;
                    local_state.end_tick = std::min(local_state.end_tick,
                                                    limited_end);
                    locally_changed = true;
                }

                auto const mapped_neighbor =
                    local_to_global.find(local_neighbor);
                if(mapped_neighbor != local_to_global.end())
                {
                    if(mapped_neighbor->second >= states_.size())
                    {
                        local_request_error = true;
                        continue;
                    }
                    CellTimeState& neighbor_state =
                        states_[mapped_neighbor->second];
                    if(neighbor_state.time_bin > limited_bin)
                    {
                        neighbor_state.time_bin = limited_bin;
                        neighbor_state.end_tick = std::min(
                            neighbor_state.end_tick, limited_end);
                        locally_changed = true;
                    }
                    continue;
                }

                const int owner = point_owner[local_neighbor];
                if(owner < 0 || owner >= rank_count ||
                   mesh_valid[local_neighbor] == 0)
                {
                    local_request_error = true;
                    continue;
                }
                if(mesh_bins[local_neighbor] > limited_bin)
                {
                    NeighborBinRequest& request =
                        outgoing[owner].emplace_back();
                    request.cell_id = mesh_ids[local_neighbor];
                    request.maximum_bin = limited_bin;
                    request.maximum_end_tick = limited_end;
                }
            }
        }

        const std::vector<std::vector<NeighborBinRequest> > incoming =
            MPI_Exchange_all_to_all(outgoing, MPI_COMM_WORLD);
        for(auto const& rank_requests : incoming)
            for(NeighborBinRequest const& request : rank_requests)
            {
                auto const found = id_to_index_.find(request.cell_id);
                if(found == id_to_index_.end())
                {
                    local_request_error = true;
                    continue;
                }
                CellTimeState& state = states_[found->second];
                if(state.time_bin > request.maximum_bin)
                {
                    state.time_bin = request.maximum_bin;
                    state.end_tick = std::min(state.end_tick,
                                              request.maximum_end_tick);
                    locally_changed = true;
                }
            }

        int request_error = local_request_error ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &request_error, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if(request_error != 0)
            throw std::logic_error(
                "Individual timestep MPI neighbor-bin request mapping failed");
        int changed_int = locally_changed ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &changed_int, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        globally_changed = changed_int != 0;
    }
#else
    bool changed = true;
    while(changed)
    {
        changed = false;
        for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
        {
            auto const mapped_i = local_to_global.find(local_i);
            if(mapped_i == local_to_global.end() || mapped_i->second >= cells.size())
                continue;
            std::size_t const i = mapped_i->second;
            neighbors.clear();
            tess.GetNeighbors(local_i, neighbors);
            for(std::size_t local_neighbor : neighbors)
            {
                auto const mapped_neighbor = local_to_global.find(local_neighbor);
                if(mapped_neighbor == local_to_global.end() ||
                   mapped_neighbor->second >= states_.size())
                    continue;
                std::size_t const neighbor = mapped_neighbor->second;
                const std::uint8_t fine_bin = std::min(states_[i].time_bin, states_[neighbor].time_bin);
                const unsigned limited = std::min<unsigned>(
                    options_.maximum_bin,
                    static_cast<unsigned>(fine_bin) + options_.maximum_neighbor_bin_difference);
                const std::uint8_t limited_bin = static_cast<std::uint8_t>(limited);

                for(std::size_t coarse : {i, neighbor})
                {
                    CellTimeState &state = states_[coarse];
                    if(state.time_bin <= limited_bin)
                        continue;
                    state.time_bin = limited_bin;
                    const std::uint64_t limited_end = nextAlignedTick(
                        event_tick, limited_bin);
                    state.end_tick = std::min(state.end_tick, limited_end);
                    changed = true;
                }
            }
        }
    }
#endif
}
