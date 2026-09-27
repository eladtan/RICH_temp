#include "Simulation.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include "misc/mpi_wait_profiler.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <spatial_ds/OctTree/OctTree.hpp>
#ifdef RICH_MPI
#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#include <mpi_utils/mpi_alltoall.hpp>
#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace
{
// RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN = m in [0, 1].  Bins are powers of two of
// their anchor and a cell takes the coarsest bin within its limit.  Anchored
// at the global step of the switch (m = 0, the behaviour before 2026-09-27),
// the grid inherits that step's ramp after a restart or a switch, so the
// finest cells step at the ramped step for the whole period.  m > 0 anchors
// bin initial_bin at m times the uncapped CFL/source suggestion of the last
// global step instead (m < 1 leaves room for dips of that limit); the first
// interval still takes at most the global step.  Default 0.8: on the TDE it
// took the 83-event window from 126 s to 99-103 s and the adaptive span to
// t = 21 from 309 s to 280 s, closer to the global run (jobs 10222173,
// 10222175, 10222210, 10222217); 0.9 brought back finest-bin dips (bin-29
// events).  Collective on first use; must agree on every rank.
double IndividualBinAnchorMargin()
{
    static double const margin = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN");
        double parsed = 0.8;
        if(value != nullptr && value[0] != '\0')
        {
            char* end = nullptr;
            parsed = std::strtod(value, &end);
            if(end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0 || parsed > 1)
                parsed = -1;
        }
        double extrema[2] = {parsed, -parsed};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[0] < 0 || extrema[0] != -extrema[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN must be one number in [0, 1] on every rank");
        return parsed;
    }();
    return margin;
}

void requireIndividualTimeStepSupport(PhysicsStep const& physics_step)
{
    if(physics_step.supportsIndividualTimeSteps())
        return;

    std::string message = "Physics step '" + physics_step.getName() +
        "' does not support individual timesteps";
    std::string const reason = physics_step.individualTimeStepUnsupportedReason();
    if(!reason.empty())
        message += ": " + reason;
    throw std::invalid_argument(message);
}

struct IndividualSignalTreePoint
{
    using coord_type = double;
    using Raw_type = Vector3D;

    std::array<double, 3> position = {{0, 0, 0}};
    std::size_t source_index =
        std::numeric_limits<std::size_t>::max();

    IndividualSignalTreePoint() = default;

    IndividualSignalTreePoint(
        Vector3D const& position_, std::size_t const source_index_)
        : position{{position_.x, position_.y, position_.z}},
          source_index(source_index_)
    {}

    double& operator[](std::size_t const dimension)
    {return position[dimension];}

    double operator[](std::size_t const dimension) const
    {return position[dimension];}

    bool operator==(IndividualSignalTreePoint const& other) const
    {
        return Vector3D(position[0], position[1], position[2]) ==
            Vector3D(other[0], other[1], other[2]);
    }

    bool operator!=(IndividualSignalTreePoint const& other) const
    {return !(*this == other);}

    friend IndividualSignalTreePoint operator+(
        IndividualSignalTreePoint const& left,
        IndividualSignalTreePoint const& right)
    {
        return IndividualSignalTreePoint(
            Vector3D(left[0] + right[0],
                     left[1] + right[1],
                     left[2] + right[2]),
            std::numeric_limits<std::size_t>::max());
    }

    friend IndividualSignalTreePoint operator/(
        IndividualSignalTreePoint const& point,
        double const divisor)
    {
        return IndividualSignalTreePoint(
            Vector3D(point[0] / divisor,
                     point[1] / divisor,
                     point[2] / divisor),
            std::numeric_limits<std::size_t>::max());
    }

    friend IndividualSignalTreePoint operator-(
        IndividualSignalTreePoint const& left,
        IndividualSignalTreePoint const& right)
    {
        return IndividualSignalTreePoint(
            Vector3D(left[0] - right[0],
                     left[1] - right[1],
                     left[2] - right[2]),
            std::numeric_limits<std::size_t>::max());
    }

    friend IndividualSignalTreePoint operator*(
        IndividualSignalTreePoint const& point,
        double const factor)
    {
        return IndividualSignalTreePoint(
            Vector3D(point[0] * factor,
                     point[1] * factor,
                     point[2] * factor),
            std::numeric_limits<std::size_t>::max());
    }

    friend std::ostream& operator<<(
        std::ostream& output,
        IndividualSignalTreePoint const& point)
    {
        return output << Vector3D(
            point[0], point[1], point[2]);
    }
};

static_assert(std::is_trivially_copyable_v<IndividualSignalTreePoint>,
              "MPI tree points must be trivially copyable");

struct IndividualSignalSource
{
    Vector3D position;
    Vector3D velocity;
    std::size_t cell_id = std::numeric_limits<std::size_t>::max();
    double sound_speed = 0;
};

using IndividualSignalTree = OctTree<IndividualSignalTreePoint>;
using IndividualSignalTreeNode = IndividualSignalTree::OctTreeNode;

struct IndividualSignalNodeSummary
{
    bool has_source = false;
    double maximum_sound_speed = 0;
    std::array<double, 3> minimum_velocity = {{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()}};
    std::array<double, 3> maximum_velocity = {{
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max()}};
};

IndividualSignalNodeSummary individualSignalLeafSummary(
    IndividualSignalSource const& source)
{
    IndividualSignalNodeSummary result;
    result.has_source = true;
    result.maximum_sound_speed = source.sound_speed;
    result.minimum_velocity = {{
        source.velocity.x, source.velocity.y, source.velocity.z}};
    result.maximum_velocity = result.minimum_velocity;
    return result;
}

void mergeIndividualSignalNodeSummary(
    IndividualSignalNodeSummary& destination,
    IndividualSignalNodeSummary const& source)
{
    if(!source.has_source)
        return;
    destination.has_source = true;
    destination.maximum_sound_speed = std::max(
        destination.maximum_sound_speed, source.maximum_sound_speed);
    for(std::size_t dimension = 0; dimension < 3; ++dimension) {
        destination.minimum_velocity[dimension] = std::min(
            destination.minimum_velocity[dimension],
            source.minimum_velocity[dimension]);
        destination.maximum_velocity[dimension] = std::max(
            destination.maximum_velocity[dimension],
            source.maximum_velocity[dimension]);
    }
}

struct IndividualWakeNodeSummary
{
    IndividualSignalNodeSummary target_bounds;
    double maximum_remaining_sleep = 0;
};

using IndividualWakeNodeSummaries =
    std::unordered_map<IndividualSignalTreeNode const*,
                       IndividualWakeNodeSummary>;

void mergeIndividualWakeNodeSummary(
    IndividualWakeNodeSummary& destination,
    IndividualWakeNodeSummary const& source)
{
    mergeIndividualSignalNodeSummary(
        destination.target_bounds, source.target_bounds);
    destination.maximum_remaining_sleep = std::max(
        destination.maximum_remaining_sleep,
        source.maximum_remaining_sleep);
}

IndividualWakeNodeSummary buildIndividualWakeNodeSummaries(
    IndividualSignalTreeNode const* const node,
    std::vector<IndividualSignalSource> const& sources,
    std::vector<double> const& remaining_sleep,
    IndividualWakeNodeSummaries& summaries)
{
    IndividualWakeNodeSummary result;
    if(node == nullptr)
        return result;

    if(node->isLeaf) {
        std::size_t const target_index = node->value.source_index;
        if(target_index >= sources.size() ||
           target_index >= remaining_sleep.size())
            throw std::out_of_range(
                "invalid target index in individual wake tree");
        if(remaining_sleep[target_index] > 0) {
            result.target_bounds =
                individualSignalLeafSummary(sources[target_index]);
            result.maximum_remaining_sleep =
                remaining_sleep[target_index];
        }
    }
    else
        for(IndividualSignalTreeNode const* const child : node->children)
            mergeIndividualWakeNodeSummary(
                result,
                buildIndividualWakeNodeSummaries(
                    child, sources, remaining_sleep, summaries));

    summaries.emplace(node, result);
    return result;
}

struct IndividualSignalQuery
{
    std::size_t cell_id = std::numeric_limits<std::size_t>::max();
    double position[3] = {0, 0, 0};
    double velocity[3] = {0, 0, 0};
    double sound_speed = 0;
    double current_limit = std::numeric_limits<double>::max();
};

struct IndividualWakeSource
{
    std::size_t cell_id = std::numeric_limits<std::size_t>::max();
    double position[3] = {0, 0, 0};
    double velocity[3] = {0, 0, 0};
    double sound_speed = 0;
};

static_assert(std::is_trivially_copyable_v<IndividualWakeSource>,
              "MPI wake sources must be trivially copyable");

struct IndividualFullSourceSweepReport
{
    bool performed = false;
    std::uint8_t minimum_occupied_bin = 0;
    std::uint64_t elapsed_minimum_steps = 0;
    std::uint64_t global_source_count = 0;
    std::uint64_t global_target_count = 0;
    std::uint64_t mpi_exchange_rounds = 0;
    std::uint64_t mpi_source_records = 0;
    std::uint64_t maximum_exchange_buffer_bytes = 0;
    double seconds_max = 0;
    // This rank's own sweep time (seconds_max is the MPI maximum).
    double seconds_local = 0;
    // This rank's wake-limiter profile at every event, sweep or not (S5):
    // deadline preparation, source states (centroids, EOS), local tree and
    // node summaries plus the distributed tree, local evaluation with the
    // destination routing, the sparse exchange (includes waiting for the
    // slowest rank), and evaluation of the received sources.
    double prepare_seconds = 0;
    double sources_seconds = 0;
    double tree_seconds = 0;
    // Wake routing tree built afresh / reused this call.
    std::uint64_t routing_rebuilds = 0;
    std::uint64_t routing_reuses = 0;
    double local_route_seconds = 0;
    double exchange_seconds = 0;
    double remote_seconds = 0;
    std::uint64_t local_signal_sources = 0;
    std::uint64_t local_outgoing_records = 0;
    std::uint64_t local_incoming_records = 0;
    // local_route_seconds split: evaluating this rank's own sources against
    // its local tree, and finding plus filling their destination ranks.
    double local_eval_seconds = 0;
    double route_seconds = 0;
    // Own sources whose query radius reached the domain diagonal.
    std::uint64_t local_saturated_sources = 0;
};


double individualPairSignalTimeStep(
    IndividualSignalQuery const& target,
    IndividualSignalSource const& source,
    double const minimum_wake_time)
{
    if(target.cell_id == source.cell_id)
        return target.current_limit;

    double const dx = target.position[0] - source.position.x;
    double const dy = target.position[1] - source.position.y;
    double const dz = target.position[2] - source.position.z;
    double const distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    if(!std::isfinite(distance))
        throw std::runtime_error(
            "invalid distance in individual tree signal");
    double closing_speed = 0;
    if(distance > 0)
    {
        closing_speed = (
            (source.velocity.x - target.velocity[0]) * dx +
            (source.velocity.y - target.velocity[1]) * dy +
            (source.velocity.z - target.velocity[2]) * dz) /
            distance;
        closing_speed = std::max(0.0, closing_speed);
    }

    double const combined_signal_speed =
        target.sound_speed + source.sound_speed + closing_speed;
    if(!std::isfinite(combined_signal_speed) ||
       combined_signal_speed < 0)
        throw std::runtime_error(
            "invalid hydrodynamic signal speed in individual wake tree");
    if(combined_signal_speed == 0)
        return target.current_limit;

    double const pair_time_step = std::max(
        minimum_wake_time,
        distance / combined_signal_speed);
    if(std::isnan(pair_time_step) || pair_time_step <= 0)
        throw std::runtime_error(
            "invalid pair timestep in individual tree signal");
    if(std::isinf(pair_time_step))
        return target.current_limit;
    return std::min(target.current_limit, pair_time_step);
}

IndividualSignalQuery makeIndividualSignalQuery(
    IndividualSignalSource const& source,
    double const current_limit)
{
    IndividualSignalQuery result;
    result.cell_id = source.cell_id;
    result.position[0] = source.position.x;
    result.position[1] = source.position.y;
    result.position[2] = source.position.z;
    result.velocity[0] = source.velocity.x;
    result.velocity[1] = source.velocity.y;
    result.velocity[2] = source.velocity.z;
    result.sound_speed = source.sound_speed;
    result.current_limit = current_limit;
    return result;
}

IndividualWakeSource makeIndividualWakeSource(
    IndividualSignalSource const& source)
{
    IndividualWakeSource result;
    result.cell_id = source.cell_id;
    result.position[0] = source.position.x;
    result.position[1] = source.position.y;
    result.position[2] = source.position.z;
    result.velocity[0] = source.velocity.x;
    result.velocity[1] = source.velocity.y;
    result.velocity[2] = source.velocity.z;
    result.sound_speed = source.sound_speed;
    return result;
}

IndividualSignalSource individualWakeSourceState(
    IndividualWakeSource const& source)
{
    IndividualSignalSource result;
    result.cell_id = source.cell_id;
    result.position = Vector3D(
        source.position[0], source.position[1], source.position[2]);
    result.velocity = Vector3D(
        source.velocity[0], source.velocity[1], source.velocity[2]);
    result.sound_speed = source.sound_speed;
    return result;
}

IndividualSignalTreePoint individualWakeSourcePoint(
    IndividualWakeSource const& source)
{
    return IndividualSignalTreePoint(
        Vector3D(source.position[0], source.position[1], source.position[2]),
        0);
}

double minimumDistanceToIndividualSignalNode(
    Vector3D const& position,
    IndividualSignalTreeNode const& node)
{
    Vector3D const lower = node.boundingBox.getLL();
    Vector3D const upper = node.boundingBox.getUR();
    double distance_squared = 0;
    for(std::size_t dimension = 0; dimension < 3; ++dimension) {
        double separation = 0;
        if(position[dimension] < lower[dimension])
            separation = lower[dimension] - position[dimension];
        else if(position[dimension] > upper[dimension])
            separation = position[dimension] - upper[dimension];
        distance_squared += separation * separation;
    }
    return std::sqrt(distance_squared);
}

double maximumIndividualWakeSignalSpeedInNode(
    IndividualWakeSource const& source,
    IndividualSignalNodeSummary const& target_bounds)
{
    double relative_speed_squared = 0;
    for(std::size_t dimension = 0; dimension < 3; ++dimension)
    {
        double const maximum_difference = std::max(
            std::abs(target_bounds.minimum_velocity[dimension] -
                     source.velocity[dimension]),
            std::abs(target_bounds.maximum_velocity[dimension] -
                     source.velocity[dimension]));
        relative_speed_squared += maximum_difference * maximum_difference;
    }
    double const combined_signal_speed =
        source.sound_speed + target_bounds.maximum_sound_speed +
        std::sqrt(relative_speed_squared);
    if(std::isnan(combined_signal_speed) || combined_signal_speed < 0)
        throw std::runtime_error(
            "invalid node hydrodynamic signal speed");
    return combined_signal_speed;
}

bool individualWakeNodeCanBeReached(
    IndividualWakeSource const& source,
    IndividualSignalTreeNode const& node,
    IndividualWakeNodeSummary const& summary)
{
    if(!summary.target_bounds.has_source ||
       summary.maximum_remaining_sleep <= 0)
        return false;
    Vector3D const source_position(
        source.position[0], source.position[1], source.position[2]);
    double const minimum_center_distance =
        minimumDistanceToIndividualSignalNode(source_position, node);
    double const maximum_signal_speed =
        maximumIndividualWakeSignalSpeedInNode(
            source, summary.target_bounds);
    if(maximum_signal_speed <= 0)
        return false;
    double const minimum_arrival_time =
        minimum_center_distance / maximum_signal_speed;
    return minimum_arrival_time < summary.maximum_remaining_sleep;
}

void evaluateIndividualWakeNode(
    IndividualWakeSource const& source,
    IndividualSignalSource const& source_state,
    IndividualSignalTreeNode const* const node,
    IndividualWakeNodeSummaries const& summaries,
    std::vector<IndividualSignalSource> const& targets,
    std::vector<double> const& remaining_sleep,
    double const minimum_wake_time,
    std::vector<double>& signal_wake_deadlines)
{
    if(node == nullptr)
        return;
    auto const summary_iterator = summaries.find(node);
    if(summary_iterator == summaries.end())
        throw std::logic_error(
            "missing node summary in individual wake tree");
    if(!individualWakeNodeCanBeReached(
           source, *node, summary_iterator->second))
        return;

    if(node->isLeaf)
    {
        std::size_t const target_index = node->value.source_index;
        if(target_index >= targets.size() ||
           target_index >= remaining_sleep.size() ||
           target_index >= signal_wake_deadlines.size())
            throw std::out_of_range(
                "invalid target index in individual wake tree");
        if(remaining_sleep[target_index] <= 0)
            return;
        if(targets[target_index].cell_id == source.cell_id)
            return;

        double target_deadline = remaining_sleep[target_index];
        if(std::isfinite(signal_wake_deadlines[target_index]))
            target_deadline = std::min(
                target_deadline,
                signal_wake_deadlines[target_index]);
        IndividualSignalQuery const target =
            makeIndividualSignalQuery(
                targets[target_index], target_deadline);
        double const arrival_time = individualPairSignalTimeStep(
            target, source_state, minimum_wake_time);
        if(arrival_time < target_deadline)
            signal_wake_deadlines[target_index] = std::min(
                signal_wake_deadlines[target_index], arrival_time);
        return;
    }

    for(IndividualSignalTreeNode const* const child : node->children)
        evaluateIndividualWakeNode(
            source, source_state, child, summaries, targets,
            remaining_sleep, minimum_wake_time, signal_wake_deadlines);
}

void evaluateIndividualWakeSource(
    IndividualWakeSource const& source,
    IndividualSignalTree const& tree,
    IndividualWakeNodeSummaries const& summaries,
    std::vector<IndividualSignalSource> const& targets,
    std::vector<double> const& remaining_sleep,
    double const minimum_wake_time,
    std::vector<double>& signal_wake_deadlines)
{
    IndividualSignalSource const source_state =
        individualWakeSourceState(source);
    evaluateIndividualWakeNode(
        source, source_state, tree.getRoot(), summaries, targets,
        remaining_sleep, minimum_wake_time, signal_wake_deadlines);
}

#ifdef RICH_MPI
double individualWakeQueryRadius(
    IndividualWakeSource const& source,
    double const maximum_remaining_sleep,
    double const maximum_sound_speed,
    double const maximum_velocity,
    double const domain_diagonal)
{
    double const source_velocity = std::sqrt(
        source.velocity[0] * source.velocity[0] +
        source.velocity[1] * source.velocity[1] +
        source.velocity[2] * source.velocity[2]);
    double const upper_signal_speed =
        source.sound_speed + maximum_sound_speed +
        source_velocity + maximum_velocity;
    double radius = maximum_remaining_sleep * upper_signal_speed;
    if(!std::isfinite(radius) || radius > domain_diagonal)
        radius = domain_diagonal;
    return std::max(0.0, radius);
}
#endif

IndividualFullSourceSweepReport limitIndividualTreeWakeTimeSteps(
    IndividualStepContext const& context,
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    EquationOfState const& eos,
    std::vector<std::shared_ptr<PhysicsStep>> const& physics,
    std::vector<CellTimeState> const& scheduler_states,
    std::vector<double> const& physical_time_step_limits,
    std::vector<double>& signal_wake_deadlines,
    std::uint64_t const last_full_source_sweep_tick,
    std::uint64_t const full_source_sweep_interval_minimum_steps)
{
    auto const wake_start = std::chrono::steady_clock::now();
    IndividualFullSourceSweepReport report;
    bool hydrodynamic_signal = false;
    std::vector<Vector3D> cell_centroids;
    std::vector<IndividualSignalSource> sources;
    std::vector<double> remaining_sleep;
    std::vector<std::size_t> signal_source_indices;
    signal_source_indices.reserve(context.active_indices.size());
    std::uint64_t local_minimum_occupied_bin =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t local_passive_target_count = 0;
    std::string local_failure;

    try
    {
        if(scheduler_states.size() != cells.size() ||
           context.active_mask.size() != cells.size() ||
           physical_time_step_limits.size() < cells.size() ||
           signal_wake_deadlines.size() < cells.size())
            throw std::logic_error(
                "individual wake canonical state size mismatch");
        if(!std::isfinite(context.time_quantum) ||
           context.time_quantum <= 0)
            throw std::runtime_error(
                "invalid time quantum in individual wake tree");
        if(full_source_sweep_interval_minimum_steps == 0 ||
           last_full_source_sweep_tick > context.event_tick)
            throw std::logic_error(
                "invalid individual full-source sweep cadence");

        for(std::shared_ptr<PhysicsStep> const& step : physics)
            hydrodynamic_signal = hydrodynamic_signal ||
                step->contributesIndividualHydrodynamicSignal();

        // Only a cell that completed its retained physical bin starts a new
        // tree wake event. A short signal catch-up is not a new source.
        for(std::size_t const source_index : context.active_indices)
        {
            if(source_index >= scheduler_states.size())
                throw std::out_of_range(
                    "individual wake source index is out of range");
            CellTimeState const& source_state =
                scheduler_states[source_index];
            if(source_state.time_bin > 62 ||
               source_state.begin_tick > context.event_tick)
                throw std::logic_error(
                    "invalid individual wake source scheduler state");
            std::uint64_t const completed_ticks =
                context.event_tick - source_state.begin_tick;
            std::uint64_t const physical_ticks =
                std::uint64_t(1) << source_state.time_bin;
            if(source_state.end_tick == context.event_tick &&
               completed_ticks == physical_ticks)
                signal_source_indices.push_back(source_index);
        }

        remaining_sleep.assign(cells.size(), 0.0);
        for(std::size_t target = 0; target < cells.size(); ++target)
        {
            if(scheduler_states[target].cell_id != cells[target].ID)
                throw std::logic_error(
                    "individual wake scheduler state order mismatch");
            if(scheduler_states[target].time_bin > 62)
                throw std::logic_error(
                    "invalid occupied bin in individual wake tree");
            local_minimum_occupied_bin = std::min(
                local_minimum_occupied_bin,
                static_cast<std::uint64_t>(
                    scheduler_states[target].time_bin));
            if(context.isActive(target))
                continue;
            ++local_passive_target_count;
            if(scheduler_states[target].end_tick <= context.event_tick)
                throw std::logic_error(
                    "inactive individual cell has no future wake tick");
            double deadline = context.time_quantum *
                static_cast<double>(
                    scheduler_states[target].end_tick -
                    context.event_tick);
            double const proposed_limit =
                physical_time_step_limits[target];
            if(!(std::isinf(proposed_limit) && proposed_limit > 0))
            {
                if(!std::isfinite(proposed_limit) ||
                   proposed_limit <= 0)
                    throw std::runtime_error(
                        "invalid inactive timestep limit in wake tree");
                deadline = std::min(deadline, proposed_limit);
            }
            if(!std::isfinite(deadline) || deadline <= 0)
                throw std::runtime_error(
                    "invalid inactive deadline in wake tree");
            remaining_sleep[target] = deadline;
        }
    }
    catch(std::exception const& error)
    {
        local_failure = error.what();
    }
    catch(...)
    {
        local_failure =
            "unknown failure preparing individual wake deadlines";
    }

#ifdef RICH_MPI
    int wake_local_valid = local_failure.empty() ? 1 : 0;
    int wake_collective_valid = wake_local_valid;
    MPI_Allreduce(MPI_IN_PLACE, &wake_collective_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    if(wake_collective_valid == 0)
    {
        if(local_failure.empty())
            local_failure =
                "individual wake deadlines failed on another MPI rank";
        throw std::runtime_error(local_failure);
    }
#else
    if(!local_failure.empty())
        throw std::runtime_error(local_failure);
#endif

#ifdef RICH_MPI
    int local_mode = hydrodynamic_signal ? 1 : 0;
    int minimum_mode = local_mode;
    int maximum_mode = local_mode;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_mode, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_mode, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(minimum_mode != maximum_mode)
        throw std::runtime_error(
            "individual hydro wake mode differs across MPI ranks");
#endif
    report.local_signal_sources =
        static_cast<std::uint64_t>(signal_source_indices.size());
    auto wake_phase_start = std::chrono::steady_clock::now();
    report.prepare_seconds = std::chrono::duration<double>(
        wake_phase_start - wake_start).count();
    if(!hydrodynamic_signal)
        return report;

    // Cadence uses the committed bins entering this event. A bin reduction
    // selected by this event becomes visible at the following event.
    std::uint64_t minimum_occupied_bin = local_minimum_occupied_bin;
#ifdef RICH_MPI
    std::uint64_t cadence_state[5] = {
        local_minimum_occupied_bin,
        last_full_source_sweep_tick,
        std::numeric_limits<std::uint64_t>::max() -
            last_full_source_sweep_tick,
        full_source_sweep_interval_minimum_steps,
        std::numeric_limits<std::uint64_t>::max() -
            full_source_sweep_interval_minimum_steps};
    MPI_Allreduce(MPI_IN_PLACE, cadence_state, 5, MPI_UINT64_T,
                  MPI_MIN, MPI_COMM_WORLD);
    minimum_occupied_bin = cadence_state[0];
    std::uint64_t const maximum_last_sweep_tick =
        std::numeric_limits<std::uint64_t>::max() - cadence_state[2];
    std::uint64_t const maximum_interval =
        std::numeric_limits<std::uint64_t>::max() - cadence_state[4];
    if(cadence_state[1] != maximum_last_sweep_tick ||
       cadence_state[3] != maximum_interval)
        throw std::runtime_error(
            "individual full-source sweep cadence differs across MPI ranks");
#endif
    if(minimum_occupied_bin > 62)
        throw std::logic_error(
            "individual full-source sweep found no occupied time bin");

    std::uint64_t const elapsed_ticks =
        context.event_tick - last_full_source_sweep_tick;
    std::uint64_t const minimum_bin_ticks =
        std::uint64_t(1) << minimum_occupied_bin;
    std::uint64_t const elapsed_minimum_steps =
        elapsed_ticks / minimum_bin_ticks;
    bool const full_source_sweep =
        elapsed_minimum_steps >=
            full_source_sweep_interval_minimum_steps;

    unsigned long long source_and_target_counts[2] = {
        static_cast<unsigned long long>(full_source_sweep ?
            cells.size() : signal_source_indices.size()),
        static_cast<unsigned long long>(local_passive_target_count)};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, source_and_target_counts, 2,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(source_and_target_counts[0] == 0)
    {
        report.prepare_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wake_phase_start).count();
        return report;
    }

    if(full_source_sweep)
    {
        report.performed = true;
        report.minimum_occupied_bin =
            static_cast<std::uint8_t>(minimum_occupied_bin);
        report.elapsed_minimum_steps = elapsed_minimum_steps;
        report.global_source_count = source_and_target_counts[0];
        report.global_target_count = source_and_target_counts[1];
    }

    double maximum_remaining_sleep = 0;
    for(double const deadline : remaining_sleep)
        maximum_remaining_sleep =
            std::max(maximum_remaining_sleep, deadline);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maximum_remaining_sleep, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    if(maximum_remaining_sleep <= 0)
    {
        report.prepare_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wake_phase_start).count();
        if(report.performed)
        {
            report.seconds_max = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wake_start).count();
            report.seconds_local = report.seconds_max;
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &report.seconds_max, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        }
        return report;
    }

    report.prepare_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wake_phase_start).count();
    wake_phase_start = std::chrono::steady_clock::now();
    local_failure.clear();
    try
    {
        for(std::shared_ptr<PhysicsStep> const& step : physics)
        {
            std::vector<Vector3D> step_centroids;
            if(step->getIndividualCellCentroids(step_centroids))
            {
                if(step_centroids.size() != cells.size())
                    throw std::logic_error(
                        "individual cell centroid count mismatch");
                if(cell_centroids.empty())
                    cell_centroids = std::move(step_centroids);
                else if(cell_centroids != step_centroids)
                    throw std::logic_error(
                        "individual cell centroid providers disagree");
            }
        }

        if(hydrodynamic_signal)
        {
            if(cell_centroids.empty())
            {
                if(tess.GetPointNo() != cells.size())
                    throw std::logic_error(
                        "individual cell centroids unavailable on partial mesh");
                cell_centroids.resize(cells.size());
                for(std::size_t cell = 0; cell < cells.size(); ++cell)
                    cell_centroids[cell] = tess.GetCellCM(cell);
            }

            sources.resize(cells.size());
            for(std::size_t cell_index = 0;
                cell_index < cells.size(); ++cell_index)
            {
                ComputationalCell3D const& cell = cells[cell_index];
                IndividualSignalSource& source = sources[cell_index];
                source.position = cell_centroids[cell_index];
                source.velocity = cell.velocity;
                source.cell_id = cell.ID;

                if(!std::isfinite(source.position.x) ||
                   !std::isfinite(source.position.y) ||
                   !std::isfinite(source.position.z) ||
                   !std::isfinite(source.velocity.x) ||
                   !std::isfinite(source.velocity.y) ||
                   !std::isfinite(source.velocity.z) ||
                   !std::isfinite(cell.density) || cell.density <= 0)
                    throw std::runtime_error(
                        "invalid cell state in individual signal tree");

                source.sound_speed = eos.dp2c(
                    cell.density, cell.pressure, cell.tracers,
                    ComputationalCell3D::tracerNames);
                if(!std::isfinite(source.sound_speed) ||
                   source.sound_speed < 0)
                    throw std::runtime_error(
                        "invalid sound speed in individual signal tree");
            }
        }
    }
    catch(std::exception const& error)
    {
        local_failure = error.what();
    }
    catch(...)
    {
        local_failure =
            "unknown failure preparing individual signal tree";
    }

#ifdef RICH_MPI
    int local_valid = local_failure.empty() ? 1 : 0;
    int collective_valid = local_valid;
    MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    if(collective_valid == 0)
    {
        if(local_failure.empty())
            local_failure =
                "individual signal tree failed on another MPI rank";
        throw std::runtime_error(local_failure);
    }

#else
    if(!local_failure.empty())
        throw std::runtime_error(local_failure);
#endif

    report.sources_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wake_phase_start).count();
    wake_phase_start = std::chrono::steady_clock::now();
    double lower[3] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    double upper[3] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
#ifdef RICH_MPI
    double maximum_sound_speed = 0;
    double maximum_velocity = 0;
#endif
    for(IndividualSignalSource const& source : sources)
    {
        for(std::size_t dimension = 0; dimension < 3; ++dimension)
        {
            lower[dimension] =
                std::min(lower[dimension], source.position[dimension]);
            upper[dimension] =
                std::max(upper[dimension], source.position[dimension]);
        }
#ifdef RICH_MPI
        maximum_sound_speed =
            std::max(maximum_sound_speed, source.sound_speed);
        maximum_velocity = std::max(maximum_velocity, std::sqrt(
            source.velocity.x * source.velocity.x +
            source.velocity.y * source.velocity.y +
            source.velocity.z * source.velocity.z));
#endif
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, lower, 3, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, upper, 3, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_sound_speed, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_velocity, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
#endif

    double diagonal_squared = 0;
    for(std::size_t dimension = 0; dimension < 3; ++dimension)
    {
        double const magnitude = std::max(
            1.0, std::max(std::abs(lower[dimension]),
                          std::abs(upper[dimension])));
        double const padding =
            64 * std::numeric_limits<double>::epsilon() * magnitude;
        lower[dimension] -= padding;
        upper[dimension] += padding;
        double const extent = upper[dimension] - lower[dimension];
        diagonal_squared += extent * extent;
    }
    double const domain_diagonal = std::sqrt(diagonal_squared);
    if(!std::isfinite(domain_diagonal) || domain_diagonal <= 0)
        throw std::runtime_error(
            "invalid domain extent in individual signal tree");

    // The trees' root box is padded by a fraction of the extent, so the
    // sources' extreme positions stay inside an older root for a while and
    // the routing tree below can be reused (a root recomputed tight every
    // event would move under the extreme sources every event).  Queries and
    // the domain diagonal above use the current extent, so the padding only
    // changes the index, not the results.
    for(std::size_t dimension = 0; dimension < 3; ++dimension)
    {
        double const pad = 0.02 * (upper[dimension] - lower[dimension]);
        lower[dimension] -= pad;
        upper[dimension] += pad;
    }
    Vector3D const lower_point(lower[0], lower[1], lower[2]);
    Vector3D const upper_point(upper[0], upper[1], upper[2]);
    IndividualSignalTree local_tree(
        IndividualSignalTreePoint(lower_point, 0),
        IndividualSignalTreePoint(upper_point, 0));
    for(std::size_t source_index = 0;
        source_index < sources.size(); ++source_index)
        local_tree.insert(IndividualSignalTreePoint(
            sources[source_index].position, source_index));
    IndividualWakeNodeSummaries local_wake_node_summaries;
    buildIndividualWakeNodeSummaries(
        local_tree.getRoot(), sources, remaining_sleep,
        local_wake_node_summaries);

#ifdef RICH_MPI
    int rank_count = 1;
    int local_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    // The routing tree (which ranks hold sources near a point) costs one
    // MPI_Allgather per tree node.  It is reused while every rank's sources
    // lie within a distance delta of the regions the cached tree attributes
    // to that rank, and every query then widens its radius by delta: a target
    // within the query radius R of a source is within R + delta of its rank's
    // cached regions, so the widened query still reaches its rank.  delta is
    // the largest such escape over all ranks (MPI_MAX); the tree is rebuilt
    // when it exceeds a quarter of the smallest query radius of this call
    // (every query is at least maximum_remaining_sleep times the largest
    // sound speed plus velocity, or the domain diagonal), which bounds the
    // widening.  RICH_INDIVIDUAL_WAKE_ROUTING_REUSE=0 rebuilds every call.
    static int const routing_reuse_enabled = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_WAKE_ROUTING_REUSE");
        int enabled = 1;
        if(value != nullptr && value[0] != '\0')
        {
            if(std::strcmp(value, "0") == 0)
                enabled = 0;
            else if(std::strcmp(value, "1") == 0)
                enabled = 1;
            else
                enabled = -1;
        }
        int extrema[2] = {enabled, -enabled};
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(extrema[0] != -extrema[1] || extrema[0] < 0)
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_WAKE_ROUTING_REUSE must be 0 or 1 on every rank");
        return extrema[0];
    }();
    // Heap-held and never destroyed: a static tree's destructor would run
    // after MPI_Finalize, in static destruction order.
    static DistributedOctTree<IndividualSignalTreePoint>* routing_tree = nullptr;
    static std::vector<BoundingBox<IndividualSignalTreePoint>>* routing_regions_store =
        new std::vector<BoundingBox<IndividualSignalTreePoint>>();
    std::vector<BoundingBox<IndividualSignalTreePoint>>& routing_regions =
        *routing_regions_store;
    double const minimum_query_radius = std::min(domain_diagonal,
        maximum_remaining_sleep * (maximum_sound_speed + maximum_velocity));
    double const escape_limit = 0.25 * minimum_query_radius;
    // Local escape: the largest distance of an own source from the own
    // cached regions (infinite without a tree or with reuse off).
    double escape = routing_reuse_enabled != 0 && routing_tree != nullptr &&
        !routing_regions.empty() ? 0.0 : std::numeric_limits<double>::infinity();
    if(std::isfinite(escape))
    {
        auto box_distance = [](BoundingBox<IndividualSignalTreePoint> const& box,
                               Vector3D const& point)
        {
            IndividualSignalTreePoint const& ll = box.getLL();
            IndividualSignalTreePoint const& ur = box.getUR();
            double const coordinates[3] = {point.x, point.y, point.z};
            double squared = 0;
            for(std::size_t d = 0; d < 3; ++d)
            {
                double const outside = std::max({0.0, ll[d] - coordinates[d],
                    coordinates[d] - ur[d]});
                squared += outside * outside;
            }
            return squared;
        };
        std::size_t last_box = 0;
        for(IndividualSignalSource const& source : sources)
        {
            std::size_t const box_count = routing_regions.size();
            double nearest = std::numeric_limits<double>::infinity();
            for(std::size_t offset = 0; offset < box_count && nearest > 0; ++offset)
            {
                std::size_t const index = (last_box + offset) % box_count;
                double const distance = box_distance(routing_regions[index],
                    source.position);
                if(distance < nearest)
                {
                    nearest = distance;
                    if(distance == 0)
                        last_box = index;
                }
            }
            escape = std::max(escape, nearest);
            if(!(escape <= escape_limit * escape_limit))
                break;
        }
    }
    // Squared distances above; the margin is a distance.
    escape = std::sqrt(escape);
    MPI_Allreduce(MPI_IN_PLACE, &escape, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    double routing_margin = 0;
    if(escape <= escape_limit)
    {
        routing_margin = escape;
        ++report.routing_reuses;
    }
    else
    {
        delete routing_tree;
        routing_tree = new DistributedOctTree<IndividualSignalTreePoint>(
            &local_tree, false, MPI_COMM_WORLD);
        routing_regions = routing_tree->getMyBoundingBoxes();
        ++report.routing_rebuilds;
    }
    DistributedOctTree<IndividualSignalTreePoint> const& distributed_tree =
        *routing_tree;
    std::uint64_t const local_source_count = full_source_sweep ?
        static_cast<std::uint64_t>(sources.size()) :
        static_cast<std::uint64_t>(signal_source_indices.size());
    std::uint64_t maximum_local_source_count = local_source_count;
    MPI_Allreduce(MPI_IN_PLACE, &maximum_local_source_count, 1,
                  MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

    // Bound both the worst-case outgoing replication and the worst-case
    // incoming payload to roughly this many bytes per rank and round.  The
    // source count is divided by rank count because a domain-wide query can
    // send every source to every rank.
    constexpr std::uint64_t exchange_payload_budget =
        std::uint64_t(16) * 1024 * 1024;
    std::uint64_t const payload_per_source_across_ranks =
        std::max<std::uint64_t>(1,
            static_cast<std::uint64_t>(rank_count) *
            sizeof(IndividualWakeSource));
    std::uint64_t const sources_per_round =
        std::max<std::uint64_t>(
            1, exchange_payload_budget /
                payload_per_source_across_ranks);
    std::uint64_t const exchange_rounds =
        (maximum_local_source_count + sources_per_round - 1) /
            sources_per_round;
    std::uint64_t local_exchanged_source_records = 0;
    std::uint64_t local_peak_exchange_records = 0;
    report.tree_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wake_phase_start).count();

    for(std::uint64_t round = 0; round < exchange_rounds; ++round)
    {
        auto round_phase_start = std::chrono::steady_clock::now();
        std::uint64_t const first_source = round * sources_per_round;
        std::uint64_t const last_source = std::min(
            local_source_count, first_source + sources_per_round);
        std::vector<std::vector<IndividualWakeSource>>
            outgoing_wake_sources(static_cast<std::size_t>(rank_count));
        std::uint64_t outgoing_record_count = 0;
        for(std::uint64_t ordinal = first_source;
            ordinal < last_source; ++ordinal)
        {
            std::size_t const source_index = full_source_sweep ?
                static_cast<std::size_t>(ordinal) :
                signal_source_indices.at(
                    static_cast<std::size_t>(ordinal));
            IndividualWakeSource const wake_source =
                makeIndividualWakeSource(sources[source_index]);
            auto const eval_start = std::chrono::steady_clock::now();
            evaluateIndividualWakeSource(
                wake_source, local_tree, local_wake_node_summaries,
                sources, remaining_sleep, context.time_quantum,
                signal_wake_deadlines);
            auto const route_start = std::chrono::steady_clock::now();
            report.local_eval_seconds +=
                std::chrono::duration<double>(route_start - eval_start).count();
            double const query_radius = individualWakeQueryRadius(
                wake_source, maximum_remaining_sleep,
                maximum_sound_speed, maximum_velocity, domain_diagonal);
            if(!(query_radius < domain_diagonal))
                ++report.local_saturated_sources;
            for(int const destination :
                distributed_tree.getIntersectingRanks(
                    individualWakeSourcePoint(wake_source),
                    query_radius + routing_margin))
                if(destination != local_rank)
                {
                    outgoing_wake_sources.at(
                        static_cast<std::size_t>(destination)).
                        push_back(wake_source);
                    ++outgoing_record_count;
                }
            report.route_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - route_start).count();
        }

        report.local_route_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - round_phase_start).count();
        round_phase_start = std::chrono::steady_clock::now();
        std::vector<std::vector<IndividualWakeSource>> const
            incoming_wake_sources = MPI_Exchange_all_to_all_sparse(
                outgoing_wake_sources, MPI_COMM_WORLD);
        report.exchange_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - round_phase_start).count();
        round_phase_start = std::chrono::steady_clock::now();
        std::uint64_t incoming_record_count = 0;
        for(std::vector<IndividualWakeSource> const& rank_sources :
            incoming_wake_sources)
        {
            incoming_record_count += rank_sources.size();
            for(IndividualWakeSource const& wake_source : rank_sources)
                evaluateIndividualWakeSource(
                    wake_source, local_tree, local_wake_node_summaries,
                    sources, remaining_sleep, context.time_quantum,
                    signal_wake_deadlines);
        }
        report.remote_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - round_phase_start).count();
        report.local_outgoing_records += outgoing_record_count;
        report.local_incoming_records += incoming_record_count;
        local_exchanged_source_records += outgoing_record_count;
        local_peak_exchange_records = std::max(
            local_peak_exchange_records,
            outgoing_record_count + incoming_record_count);
    }

    if(report.performed)
    {
        report.mpi_exchange_rounds = exchange_rounds;
        report.mpi_source_records = local_exchanged_source_records;
        MPI_Allreduce(MPI_IN_PLACE, &report.mpi_source_records, 1,
                      MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        std::uint64_t maximum_exchange_records =
            local_peak_exchange_records;
        MPI_Allreduce(MPI_IN_PLACE, &maximum_exchange_records, 1,
                      MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        report.maximum_exchange_buffer_bytes =
            maximum_exchange_records * sizeof(IndividualWakeSource);
    }
#else
    if(full_source_sweep)
        for(IndividualSignalSource const& source : sources)
            evaluateIndividualWakeSource(
                makeIndividualWakeSource(source), local_tree,
                local_wake_node_summaries, sources, remaining_sleep,
                context.time_quantum, signal_wake_deadlines);
    else
        for(std::size_t const source_index : signal_source_indices)
            evaluateIndividualWakeSource(
                makeIndividualWakeSource(sources[source_index]),
                local_tree, local_wake_node_summaries, sources,
                remaining_sleep, context.time_quantum,
                signal_wake_deadlines);
#endif

    if(report.performed)
    {
        report.seconds_max = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wake_start).count();
        report.seconds_local = report.seconds_max;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &report.seconds_max, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    }
    return report;
}

bool parseEnvironmentToggle(char const* const name, bool const fallback,
                            bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    if(std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
       std::strcmp(value, "on") == 0 || std::strcmp(value, "yes") == 0)
        return true;
    if(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
       std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0)
        return false;
    valid = false;
    return fallback;
}

// Numeric environment parsers, shared by the MPI runtime options and the
// adaptive integration options (which serial builds also compile).
double parseEnvironmentDouble(char const* const name, double const fallback,
                               double const minimum, bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    errno = 0;
    char* end = nullptr;
    double const parsed = std::strtod(value, &end);
    if(errno != 0 || end == value || *end != '\0' ||
       !std::isfinite(parsed) || parsed < minimum)
    {
        valid = false;
        return fallback;
    }
    return parsed;
}

std::size_t parseEnvironmentSize(char const* const name,
                                 std::size_t const fallback, bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    errno = 0;
    char* end = nullptr;
    unsigned long long const parsed = std::strtoull(value, &end, 10);
    if(value[0] == '-' || errno != 0 || end == value || *end != '\0' ||
       parsed == 0 ||
       parsed > static_cast<unsigned long long>(
           std::numeric_limits<std::size_t>::max()))
    {
        valid = false;
        return fallback;
    }
    return static_cast<std::size_t>(parsed);
}

#ifndef RICH_MPI
void requireSerialIndividualActiveHilbertCompatibility()
{
    char const* const configured = std::getenv(
        "RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE");
    if(configured == nullptr || configured[0] == '\0')
        return;

    bool valid = true;
    bool const enabled = parseEnvironmentToggle(
        "RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE", true, valid);
    if(!valid)
        throw std::invalid_argument(
            "Invalid individual active Hilbert environment option");
    if(enabled)
        throw std::logic_error(
            "Active Hilbert balancing requires an MPI build");
}
#endif

#ifdef RICH_MPI
struct IndividualRebalanceRuntimeOptions
{
    bool enabled = false;
    bool trace = false;
    bool before_first_event = false;
    double threshold = 1.25;
    double immediate_amr_threshold = 1.5;
    std::size_t cooldown_events = 8;
    double amortization_factor = 2;
};

struct IndividualActiveHilbertRuntimeOptions
{
    bool enabled = true;
    bool explicitly_configured = false;
    bool trace = false;
    // RICH_INDIVIDUAL_ACTIVE_HILBERT_BOUND_CHECK=1: still build the active-only
    // cut the bound predicts to be rejected, and fail if it is accepted
    // (validation of the skip; costs what the skip saves).
    bool bound_check = false;
    double active_threshold = 1.25;
    // Active-only cuts must not concentrate so many passive cells that the
    // canonical state becomes a memory hazard.
    double maximum_owned_cell_skew = 2;
    // RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS=S (default 4; 1 = one range per
    // rank, the active-only single-range cut): spread the cells over
    // S * ranks equal-count curve pieces dealt out in turn
    // (HilbertLoadBalancer::rebalanceInterleaved), adopted and kept or
    // reverted by the measured-probe ledger.  TDE, 256 ranks, 83 events:
    // S=1 170 s, S=4 126 s.
    int segments_per_rank = 4;
    // RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_BINS (default off): bin-balanced
    // pieces and assignment (buildIndividualBinBalancedSegments), re-planned
    // on accumulated mesh excess.  Off (measured better on the TDE: the bins
    // the plan balances change within ~10 events) deals equal-count pieces
    // out in turn, which spreads every region over S ranks whatever is
    // active, and re-plans only when the owned-cell skew drifts.
    bool segment_bins = false;
    // RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_WORK = beta >= 0 (default 0):
    // interleaved pieces of equal weight 1 + beta * work, work being a
    // cell's update frequency 2^(coarsest bin - bin) normalized to one per
    // cell on average (AREPO's doubling weight), so busy regions are cut
    // into finer pieces that the round-robin spreads.
    double segment_work = 0;
};

struct IndividualForceAllActiveRuntimeOptions
{
    bool enabled = false;
    bool latch_enabled = false;
    std::size_t minimum_bin = 63;
};

#endif

std::uint64_t individualFullSourceSweepRuntimeInterval(
    std::uint64_t const fallback)
{
    bool locally_valid = fallback > 0;
    std::uint64_t interval = fallback;
    char const* const value = std::getenv(
        "RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL");
    if(value != nullptr && value[0] != '\0')
    {
        errno = 0;
        char* end = nullptr;
        unsigned long long const parsed = std::strtoull(value, &end, 10);
        if(value[0] < '0' || value[0] > '9' || errno != 0 || end == value ||
           *end != '\0' || parsed == 0)
            locally_valid = false;
        else
        {
            locally_valid = true;
            interval = static_cast<std::uint64_t>(parsed);
        }
    }

#ifdef RICH_MPI
    int collective_valid = locally_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    unsigned long long minimum_interval = interval;
    unsigned long long maximum_interval = interval;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_interval, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_interval, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    if(collective_valid == 0 || minimum_interval != maximum_interval)
        throw std::invalid_argument(
            "Invalid or MPI-inconsistent "
            "RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL");
    interval = static_cast<std::uint64_t>(minimum_interval);
#else
    if(!locally_valid)
        throw std::invalid_argument(
            "RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL must be a "
            "positive integer");
#endif
    return interval;
}

#ifdef RICH_MPI
IndividualRebalanceRuntimeOptions const& individualRebalanceRuntimeOptions()
{
    static IndividualRebalanceRuntimeOptions const options = []()
    {
        IndividualRebalanceRuntimeOptions result;
        bool locally_valid = true;
        result.enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_AUTO_REBALANCE", false, locally_valid);
        result.trace = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_PERF_TRACE", false, locally_valid);
        result.before_first_event = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_REBALANCE_BEFORE_FIRST_EVENT", false,
            locally_valid);
        result.threshold = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_THRESHOLD", 1.25, 1.0,
            locally_valid);
        result.immediate_amr_threshold = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_AMR_THRESHOLD", 1.5,
            result.threshold, locally_valid);
        result.cooldown_events = parseEnvironmentSize(
            "RICH_INDIVIDUAL_REBALANCE_COOLDOWN", 8, locally_valid);
        result.amortization_factor = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_AMORTIZATION", 2.0, 0.0,
            locally_valid);

        int local_mask = (result.enabled ? 1 : 0) |
            (result.trace ? 2 : 0) |
            (result.before_first_event ? 4 : 0);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        double minimum_values[3] = {result.threshold,
                                    result.immediate_amr_threshold,
                                    result.amortization_factor};
        double maximum_values[3] = {result.threshold,
                                    result.immediate_amr_threshold,
                                    result.amortization_factor};
        unsigned long long minimum_cooldown = result.cooldown_events;
        unsigned long long maximum_cooldown = result.cooldown_events;
        int collective_valid = locally_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minimum_values, 3, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maximum_values, 3, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_cooldown, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_cooldown, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(collective_valid == 0 || minimum_mask != maximum_mask ||
           !std::equal(std::begin(minimum_values), std::end(minimum_values),
                       std::begin(maximum_values)) ||
           minimum_cooldown != maximum_cooldown)
            throw std::invalid_argument(
                "Invalid or inconsistent individual rebalance environment options");
        return result;
    }();
    return options;
}

IndividualActiveHilbertRuntimeOptions const&
individualActiveHilbertRuntimeOptions()
{
    static IndividualActiveHilbertRuntimeOptions const options = []()
    {
        IndividualActiveHilbertRuntimeOptions result;
        bool locally_valid = true;
        char const* const configured_cache = std::getenv(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE");
        result.explicitly_configured = configured_cache != nullptr &&
            configured_cache[0] != '\0';
        result.enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE", true,
            locally_valid);
        result.trace = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_PERF_TRACE", false, locally_valid);
        result.bound_check = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_BOUND_CHECK", false, locally_valid);
        result.active_threshold = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD", 1.25, 1.0,
            locally_valid);
        result.maximum_owned_cell_skew = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW", 2.0, 1.0,
            locally_valid);
        std::size_t const segments = parseEnvironmentSize(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS", 4, locally_valid);
        if(segments < 1 || segments > 64)
            locally_valid = false;
        result.segments_per_rank = static_cast<int>(
            std::min<std::size_t>(std::max<std::size_t>(segments, 1), 64));
        result.segment_bins = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_BINS", false, locally_valid);
        result.segment_work = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_WORK", 0.0, 0.0,
            locally_valid);
        // Weights stay finite: each is at most 1 + beta * cells.
        if(result.segment_work > 1e6)
            locally_valid = false;

        int const local_mask = (result.enabled ? 1 : 0) |
            (result.trace ? 2 : 0) |
            (result.explicitly_configured ? 4 : 0) |
            (result.bound_check ? 8 : 0) |
            (result.segment_bins ? 16 : 0);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        double minimum_values[4] = {result.active_threshold,
                                    result.maximum_owned_cell_skew,
                                    static_cast<double>(result.segments_per_rank),
                                    result.segment_work};
        double maximum_values[4] = {result.active_threshold,
                                    result.maximum_owned_cell_skew,
                                    static_cast<double>(result.segments_per_rank),
                                    result.segment_work};
        int collective_valid = locally_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minimum_values, 4, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maximum_values, 4, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        if(collective_valid == 0 || minimum_mask != maximum_mask ||
           !std::equal(std::begin(minimum_values), std::end(minimum_values),
                       std::begin(maximum_values)))
            throw std::invalid_argument(
                "Invalid or inconsistent active Hilbert cache options");
        return result;
    }();
    return options;
}

IndividualForceAllActiveRuntimeOptions const&
individualForceAllActiveRuntimeOptions()
{
    static IndividualForceAllActiveRuntimeOptions const options = []()
    {
        IndividualForceAllActiveRuntimeOptions result;
        bool locally_valid = true;
        char const* const raw = std::getenv(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN");
        result.enabled = raw != nullptr && raw[0] != '\0';
        result.latch_enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH", false,
            locally_valid);
        result.minimum_bin = parseEnvironmentSize(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN", 63,
            locally_valid);
        if(result.minimum_bin >= 64)
            locally_valid = false;
        if(result.latch_enabled && !result.enabled)
            locally_valid = false;

        int const local_mask = (result.enabled ? 1 : 0) |
            (result.latch_enabled ? 2 : 0);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        unsigned long long minimum_bin = result.minimum_bin;
        unsigned long long maximum_bin = result.minimum_bin;
        int collective_valid = locally_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_bin, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_bin, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(collective_valid == 0 || minimum_mask != maximum_mask ||
           minimum_bin != maximum_bin)
            throw std::invalid_argument(
                "Invalid or inconsistent forced-active event environment option");
        return result;
    }();
    return options;
}

struct IndividualHilbertPartitionLoad
{
    std::uint64_t active_cells = 0;
    std::uint64_t maximum_active_cells = 0;
    std::uint64_t owned_cells = 0;
    std::uint64_t minimum_owned_cells = 0;
    std::uint64_t maximum_owned_cells = 0;
    int ranks = 1;

    double activeMaxMean(void) const
    {
        return active_cells > 0 ?
            static_cast<double>(maximum_active_cells) /
                (static_cast<double>(active_cells) / ranks) : 1;
    }

    double ownedMaxMean(void) const
    {
        return owned_cells > 0 ?
            static_cast<double>(maximum_owned_cells) /
                (static_cast<double>(owned_cells) / ranks) : 1;
    }

    bool ownsEveryRank(void) const
    {
        return minimum_owned_cells > 0;
    }

    bool activeBalanced(double const threshold) const
    {
        double const achievable_mean = std::max(
            1.0, static_cast<double>(active_cells) / ranks);
        return static_cast<double>(maximum_active_cells) <=
            threshold * achievable_mean;
    }
};

IndividualHilbertPartitionLoad measureIndividualCurrentLoad(
    std::size_t const local_owned_cells,
    std::size_t const local_active_cells, int const ranks)
{
    unsigned long long totals[2] = {
        static_cast<unsigned long long>(local_owned_cells),
        static_cast<unsigned long long>(local_active_cells)};
    unsigned long long maxima[2] = {totals[0], totals[1]};
    unsigned long long minimum_owned = totals[0];
    MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minimum_owned, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);

    IndividualHilbertPartitionLoad result;
    result.owned_cells = totals[0];
    result.active_cells = totals[1];
    result.maximum_owned_cells = maxima[0];
    result.minimum_owned_cells = minimum_owned;
    result.maximum_active_cells = maxima[1];
    result.ranks = ranks;
    return result;
}

// Owned cells (entries [0, ranks)) and active cells (entries [ranks,
// 2 ranks)) in each segment of a cut, from the current generator positions;
// segment r holds the curve keys the cut gives to rank r.  Collective.
std::vector<unsigned long long> countIndividualHilbertSegments(
    HilbertLoadBalancer<Vector3D> const& load_balance,
    std::vector<Vector3D> const& points,
    std::vector<unsigned char> const& active_mask, int const ranks)
{
    if(points.size() != active_mask.size())
        throw std::logic_error(
            "Active Hilbert measurement requires aligned points and activity");

    std::vector<unsigned long long> counts(
        static_cast<std::size_t>(2 * ranks), 0);
    int local_owner_valid = 1;
    for(std::size_t i = 0; i < points.size(); ++i)
    {
        int const owner = load_balance.getOwner(points[i]);
        if(owner < 0 || owner >= ranks)
        {
            local_owner_valid = 0;
            continue;
        }
        ++counts[static_cast<std::size_t>(owner)];
        if(active_mask[i] != 0)
            ++counts[static_cast<std::size_t>(ranks + owner)];
    }
    MPI_Allreduce(MPI_IN_PLACE, &local_owner_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_owner_valid == 0)
        throw std::logic_error(
            "Active Hilbert boundary selected an invalid MPI owner");
    MPI_Allreduce(MPI_IN_PLACE, counts.data(),
                  static_cast<int>(counts.size()), MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    return counts;
}

IndividualHilbertPartitionLoad individualHilbertLoadFromSegments(
    std::vector<unsigned long long> const& counts, int const ranks)
{
    IndividualHilbertPartitionLoad result;
    result.ranks = ranks;
    result.minimum_owned_cells = counts[0];
    for(int rank = 0; rank < ranks; ++rank)
    {
        std::uint64_t const owned = counts[static_cast<std::size_t>(rank)];
        std::uint64_t const active =
            counts[static_cast<std::size_t>(ranks + rank)];
        result.owned_cells += owned;
        result.active_cells += active;
        result.maximum_owned_cells = std::max(
            result.maximum_owned_cells, owned);
        result.minimum_owned_cells = std::min(
            result.minimum_owned_cells, owned);
        result.maximum_active_cells = std::max(
            result.maximum_active_cells, active);
    }
    return result;
}

IndividualHilbertPartitionLoad measureIndividualHilbertPartition(
    HilbertLoadBalancer<Vector3D> const& load_balance,
    std::vector<Vector3D> const& points,
    std::vector<unsigned char> const& active_mask, int const ranks)
{
    return individualHilbertLoadFromSegments(countIndividualHilbertSegments(
        load_balance, points, active_mask, ranks), ranks);
}

// Per-bin occupancy of a partition: for each time bin, the global cell count
// and the largest count on one rank.  A bin is "constrained" when it holds
// enough cells that ranks * ceil(count / ranks) / count <= threshold (an even
// split can meet the threshold); the partition's bin skew is the largest
// max/mean over constrained bins.  Collective.
struct IndividualBinOccupancy
{
    std::array<unsigned long long, 64> total{};
    std::array<unsigned long long, 64> maximum{};

    bool constrained(int const bin, int const ranks, double const threshold) const
    {
        unsigned long long const count = this->total[static_cast<std::size_t>(bin)];
        if(count == 0)
            return false;
        unsigned long long const per_rank =
            (count + static_cast<unsigned long long>(ranks) - 1) /
            static_cast<unsigned long long>(ranks);
        return static_cast<double>(ranks) * static_cast<double>(per_rank) /
            static_cast<double>(count) <= threshold;
    }

    double skew(int const ranks, double const threshold) const
    {
        double worst = 1;
        for(int bin = 0; bin < 64; ++bin)
            if(this->constrained(bin, ranks, threshold))
                worst = std::max(worst, static_cast<double>(
                    this->maximum[static_cast<std::size_t>(bin)]) /
                    (static_cast<double>(this->total[static_cast<std::size_t>(bin)]) / ranks));
        return worst;
    }
};

// Active-fraction class of an event for the segmented-ownership ledger:
// below 0.1 %, 1 %, 10 %, 50 %, or above.
int IndividualSegmentEventClass(std::uint64_t const active, std::uint64_t const owned)
{
    double const fraction = owned > 0 ?
        static_cast<double>(active) / static_cast<double>(owned) : 1;
    return fraction < 1e-3 ? 0 : fraction < 1e-2 ? 1 : fraction < 1e-1 ? 2 :
        fraction < 0.5 ? 3 : 4;
}

template<typename StateVector>
IndividualBinOccupancy measureIndividualBinOccupancy(StateVector const& states)
{
    IndividualBinOccupancy result;
    std::array<unsigned long long, 64> local{};
    for(auto const& state : states)
        ++local[std::min<std::size_t>(state.time_bin, 63)];
    MPI_Allreduce(local.data(), result.total.data(), 64, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), result.maximum.data(), 64, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    return result;
}

// Bin-balanced segmented partition (plan stage A2).  Collective.
// 1. Cut the curve into piecesPerRank * ranks pieces of equal weight, where a
//    cell weighs 1 plus, for a constrained bin b, cells / count_b: every
//    constrained bin carries as much weight as all cells together, so pieces
//    are fine exactly where a bin is dense (a top-down refinement by cost in
//    one weighted cut, as AREPO refines its top leaves by cost).
// 2. Count cells and constrained-bin cells per piece (one Allreduce).
// 3. Assign pieces greedily (largest first by AREPO's doubling priority,
//    sum over bins of 2^(coarsest - b) cells) to the rank that minimizes its
//    worst normalized load over cells and constrained bins.  Every rank runs
//    the same deterministic assignment on the same reduced data.
// 4. Merge consecutive pieces with the same owner.
struct IndividualSegmentPlanStats
{
    int pieces = 0;
    int segments = 0;
    int constrained_bins = 0;
    double planned_bin_skew = 0;
    double planned_owned_skew = 0;
};

template<typename StateVector>
std::shared_ptr<HilbertLoadBalancer<Vector3D>> buildIndividualBinBalancedSegments(
    HilbertLoadBalancer<Vector3D> const& current, std::vector<Vector3D> const& points,
    StateVector const& states, IndividualBinOccupancy const& occupancy,
    int const piecesPerRank, int const ranks, double const threshold,
    IndividualSegmentPlanStats& stats)
{
    if(points.size() != states.size())
        throw std::logic_error("Bin-balanced segments need aligned points and scheduler states");
    std::vector<int> constrained_index(64, -1);
    std::vector<int> constrained_bins;
    for(int bin = 0; bin < 64; ++bin)
        if(occupancy.constrained(bin, ranks, threshold))
        {
            constrained_index[static_cast<std::size_t>(bin)] =
                static_cast<int>(constrained_bins.size());
            constrained_bins.push_back(bin);
        }
    std::size_t const bins = constrained_bins.size();
    unsigned long long global_cells = 0;
    for(unsigned long long const count : occupancy.total)
        global_cells += count;

    std::vector<curve_index_t> indices(points.size());
    std::vector<double> weights(points.size(), 1.0);
    for(std::size_t i = 0; i < points.size(); ++i)
    {
        indices[i] = current.getCurveIndex(points[i]);
        int const k = constrained_index[std::min<std::size_t>(states[i].time_bin, 63)];
        if(k >= 0)
            weights[i] += static_cast<double>(global_cells) /
                static_cast<double>(occupancy.total[static_cast<std::size_t>(constrained_bins[static_cast<std::size_t>(k)])]);
    }
    int const pieces = std::max(1, piecesPerRank) * ranks;
    std::vector<curve_index_t> borders = getWeightedBorders3(indices, weights,
        std::less<curve_index_t>{}, MPI_COMM_WORLD, pieces);
    if(borders.size() != static_cast<std::size_t>(pieces))
        throw std::logic_error("Bin-balanced segments got an unexpected piece count");

    std::size_t const width = bins + 1;
    std::vector<unsigned long long> histogram(static_cast<std::size_t>(pieces) * width, 0);
    for(std::size_t i = 0; i < points.size(); ++i)
    {
        std::size_t const piece = std::min<std::size_t>(static_cast<std::size_t>(std::distance(
            borders.cbegin(), std::upper_bound(borders.cbegin(), borders.cend(), indices[i]))),
            static_cast<std::size_t>(pieces - 1));
        ++histogram[piece * width];
        int const k = constrained_index[std::min<std::size_t>(states[i].time_bin, 63)];
        if(k >= 0)
            ++histogram[piece * width + 1 + static_cast<std::size_t>(k)];
    }
    MPI_Allreduce(MPI_IN_PLACE, histogram.data(), static_cast<int>(histogram.size()),
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    int const coarsest = constrained_bins.empty() ? 0 : constrained_bins.back();
    std::vector<double> priority(static_cast<std::size_t>(pieces), 0);
    for(std::size_t piece = 0; piece < static_cast<std::size_t>(pieces); ++piece)
    {
        priority[piece] = static_cast<double>(histogram[piece * width]);
        for(std::size_t k = 0; k < bins; ++k)
            priority[piece] += std::ldexp(static_cast<double>(histogram[piece * width + 1 + k]),
                coarsest - constrained_bins[k]);
    }
    std::vector<std::size_t> order(static_cast<std::size_t>(pieces));
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&priority](std::size_t a, std::size_t b)
        { return priority[a] > priority[b]; });

    std::vector<double> mean(width, 0);
    mean[0] = static_cast<double>(global_cells) / ranks;
    for(std::size_t k = 0; k < bins; ++k)
        mean[1 + k] = static_cast<double>(occupancy.total[static_cast<std::size_t>(constrained_bins[k])]) / ranks;
    std::vector<double> load(static_cast<std::size_t>(ranks) * width, 0);
    std::vector<double> worst(static_cast<std::size_t>(ranks), 0);
    std::vector<int> owners(static_cast<std::size_t>(pieces), 0);
    for(std::size_t const piece : order)
    {
        int best_rank = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        double best_tie = std::numeric_limits<double>::infinity();
        for(int rank = 0; rank < ranks; ++rank)
        {
            double cost = 0;
            for(std::size_t c = 0; c < width; ++c)
                if(mean[c] > 0)
                    cost = std::max(cost, (load[static_cast<std::size_t>(rank) * width + c] +
                        static_cast<double>(histogram[piece * width + c])) / mean[c]);
            // Ties: the rank with the smaller current worst load.
            if(cost < best_cost || (cost == best_cost && worst[static_cast<std::size_t>(rank)] < best_tie))
            {
                best_cost = cost;
                best_tie = worst[static_cast<std::size_t>(rank)];
                best_rank = rank;
            }
        }
        owners[piece] = best_rank;
        for(std::size_t c = 0; c < width; ++c)
            load[static_cast<std::size_t>(best_rank) * width + c] +=
                static_cast<double>(histogram[piece * width + c]);
        worst[static_cast<std::size_t>(best_rank)] = best_cost;
    }

    stats.pieces = pieces;
    stats.constrained_bins = static_cast<int>(bins);
    stats.planned_owned_skew = 0;
    stats.planned_bin_skew = 1;
    for(int rank = 0; rank < ranks; ++rank)
    {
        stats.planned_owned_skew = std::max(stats.planned_owned_skew,
            load[static_cast<std::size_t>(rank) * width] / mean[0]);
        for(std::size_t k = 0; k < bins; ++k)
            if(mean[1 + k] > 0)
                stats.planned_bin_skew = std::max(stats.planned_bin_skew,
                    load[static_cast<std::size_t>(rank) * width + 1 + k] / mean[1 + k]);
    }

    std::vector<curve_index_t> merged_borders;
    std::vector<int> merged_owners;
    for(std::size_t piece = 0; piece < static_cast<std::size_t>(pieces); ++piece)
    {
        if(!merged_owners.empty() && merged_owners.back() == owners[piece])
            merged_borders.back() = borders[piece];
        else
        {
            merged_borders.push_back(borders[piece]);
            merged_owners.push_back(owners[piece]);
        }
    }
    stats.segments = static_cast<int>(merged_borders.size());
    std::shared_ptr<HilbertLoadBalancer<Vector3D>> proposal = current.clone();
    proposal->setSegments(merged_borders, merged_owners);
    return proposal;
}

// A lower bound on the most cells one rank owns after the active-only
// rebalance (HilbertLoadBalancer::rebalance, weight 1 on active and 0 on
// passive cells).  Both paths of getWeightedBorders3 put each of the first
// P - 1 borders on the key of an active cell (the first point whose prefix
// weight exceeds the target); the appended last border (the last selected
// one + 1) cannot matter because getOwner clamps to rank P - 1.  A key's owner
// is the number of borders at or below it, so the cells whose keys lie
// strictly between two consecutive active keys (or before the first, or after
// the last) end on one rank; equal keys stay together and duplicate borders
// only leave ranks empty.  The cells of consecutive segments of any cut that
// hold no active cell form such a set; the largest run is the bound.
struct IndividualActiveCutBound
{
    std::uint64_t cells = 0;
    int first_segment = -1;
    int segments = 0;
};

IndividualActiveCutBound individualActiveCutOwnedLowerBound(
    std::vector<unsigned long long> const& counts, int const ranks)
{
    IndividualActiveCutBound best;
    std::uint64_t run_cells = 0;
    int run_first = -1;
    for(int segment = 0; segment < ranks; ++segment)
    {
        if(counts[static_cast<std::size_t>(ranks + segment)] != 0)
        {
            run_cells = 0;
            run_first = -1;
            continue;
        }
        if(run_first < 0)
            run_first = segment;
        run_cells += counts[static_cast<std::size_t>(segment)];
        if(run_cells > best.cells)
        {
            best.cells = run_cells;
            best.first_segment = run_first;
            best.segments = segment - run_first + 1;
        }
    }
    return best;
}

void reportIndividualRankDistribution(int const rank, int const rank_count,
                                      std::size_t const cycle,
                                      std::string const& phase,
                                      double const local_value,
                                      char const* const unit)
{
    std::vector<double> values(rank == 0 ? rank_count : 0);
    MPI_Gather(&local_value, 1, MPI_DOUBLE,
               rank == 0 ? values.data() : nullptr, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    if(rank != 0)
        return;
    auto const minimum_iterator =
        std::min_element(values.begin(), values.end());
    auto const maximum_iterator =
        std::max_element(values.begin(), values.end());
    int const minimum_rank =
        static_cast<int>(std::distance(values.begin(), minimum_iterator));
    int const maximum_rank =
        static_cast<int>(std::distance(values.begin(), maximum_iterator));
    std::sort(values.begin(), values.end());
    std::size_t const median_index = (values.size() - 1) / 2;
    std::size_t const p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    double const mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    std::cout << std::setprecision(17)
              << "RICH_STEP_DETAIL mode=individual cycle=" << cycle
              << " phase=" << phase
               << " unit=" << unit
               << " min=" << values.front()
               << " min_rank=" << minimum_rank
               << " median=" << values[median_index]
               << " mean=" << mean
               << " p95=" << values[p95_index]
               << " max=" << values.back()
               << " max_rank=" << maximum_rank << std::endl;
}

double peakResidentSetKiB()
{
    struct rusage usage;
    return getrusage(RUSAGE_SELF, &usage) == 0 ?
        static_cast<double>(usage.ru_maxrss) : -1.0;
}

double currentResidentSetKiB()
{
    std::ifstream statm("/proc/self/statm");
    unsigned long long virtual_pages = 0;
    unsigned long long resident_pages = 0;
    long const page_size = sysconf(_SC_PAGESIZE);
    if(page_size <= 0 || !(statm >> virtual_pages >> resident_pages))
        return -1.0;
    return static_cast<double>(resident_pages) *
        static_cast<double>(page_size) / 1024.0;
}
#endif

void ValidateRuntimeLogConfiguration(void)
{
    RuntimeLogConfiguration const log_configuration =
        GetRuntimeLogConfiguration();
    RuntimeColorConfiguration const color_configuration =
        GetRuntimeColorConfiguration();
    int valid = log_configuration.valid && color_configuration.valid ? 1 : 0;
    int minimum_configuration[2] = {
        static_cast<int>(log_configuration.level),
        static_cast<int>(color_configuration.mode)};
    int maximum_configuration[2] = {
        minimum_configuration[0], minimum_configuration[1]};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, minimum_configuration, 2, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, maximum_configuration, 2, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if(valid == 0 || minimum_configuration[0] != maximum_configuration[0] ||
       minimum_configuration[1] != maximum_configuration[1])
        throw std::invalid_argument(
            "Invalid or MPI-inconsistent runtime logging environment; "
            "RICH_RUNTIME_LOG expects summary|detailed and "
            "RICH_RUNTIME_COLOR expects auto|always|never");
}

double RuntimeMaximum(double value)
{
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    return value;
}

double RuntimeMinimum(double value)
{
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    return value;
}

std::uint64_t RuntimeSum(std::uint64_t value)
{
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    return value;
}

std::uint64_t RuntimeMaximum(std::uint64_t value)
{
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    return value;
}

std::uint64_t RuntimeMinimum(std::uint64_t value)
{
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    return value;
}

std::string RuntimeToken(std::string value)
{
    if(value.empty())
        return "unknown";
    for(char& character : value)
        if(std::isspace(static_cast<unsigned char>(character)) ||
           character == '=' || character == ',')
            character = '_';
    return value;
}

double LocalRuntimePhase(
    std::map<std::string, double> const& phases,
    std::string const& phase_name)
{
    auto const exact = phases.find(phase_name);
    if(exact != phases.end())
        return exact->second;
    double seconds = 0;
    for(auto const& phase : phases)
    {
        std::string lowercase = phase.first;
        std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        if(lowercase.find(phase_name) != std::string::npos)
            seconds += phase.second;
    }
    return seconds;
}

struct RuntimeStepOutput
{
    std::string mode;
    std::size_t cycle = 0;
    double start_time = 0;
    double end_time = 0;
    double event_dt = 0;
    double applied_dt_min = 0;
    double applied_dt_max = 0;
    double next_event_dt = 0;
    std::uint64_t active_cells = 0;
    std::uint64_t total_cells = 0;
    std::string active_bins;
    std::uint64_t source_calls = 0;
    std::uint64_t mesh_builds = 0;
    double hydro_seconds = 0;
    double gravity_seconds = 0;
    double radiation_seconds = 0;
    double amr_seconds = 0;
    double source_seconds = 0;
    double source_first_seconds = 0;
    double source_second_seconds = 0;
    double mesh_seconds = 0;
    double step_seconds = 0;
    // Individual mode only: exclusive scheduler-side sections outside the
    // physics steps (rank maxima), and the per-rank residual, rank max.
    double sync_seconds = 0;
    double prepare_seconds = 0;
    double closure_seconds = 0;
    double suggest_seconds = 0;
    double wake_seconds = 0;
    double sweep_seconds = 0;
    double commit_seconds = 0;
    double other_seconds = 0;
    std::uint64_t full_builds = 0;
};

struct RuntimeAMROutput
{
    std::string mode;
    std::size_t cycle = 0;
    double time = 0;
    std::uint64_t cells_before = 0;
    std::uint64_t added_cells = 0;
    std::uint64_t removed_cells = 0;
    std::uint64_t cells_after = 0;
};

constexpr int runtime_simulation_digits = 12;

std::string RuntimeLabel(
    char const* const label, char const* const ansi_style,
    bool const color_enabled)
{
    if(!color_enabled)
        return label;
    return std::string(ansi_style) + label + "\033[0m";
}

void WriteRuntimeStep(
    int const rank, bool const detailed, bool const color_enabled,
    RuntimeStepOutput const& step)
{
    if(rank != 0)
        return;

    std::ostringstream header_line;
    header_line << RuntimeLabel("RICH_STEP", "\033[1;36m", color_enabled)
                << " mode=" << step.mode
                << " cycle=" << step.cycle;

    std::ostringstream time_line;
    time_line << "  " << RuntimeLabel("time  ", "\033[36m", color_enabled)
              << " | " << std::setprecision(runtime_simulation_digits)
              << "t_start=" << step.start_time
              << " | t_end=" << step.end_time
              << " | event_dt=" << step.event_dt
              << " | applied_dt_min=" << step.applied_dt_min
              << " | applied_dt_max=" << step.applied_dt_max
              << " | next_event_dt=" << step.next_event_dt;

    std::ostringstream work_line;
    work_line << "  " << RuntimeLabel("work  ", "\033[34m", color_enabled)
              << " | active_cells=" << step.active_cells
              << " | total_cells=" << step.total_cells
              << " | active_bins=" << step.active_bins;

    std::ostringstream phases_line;
    phases_line << "  " << RuntimeLabel("phases", "\033[35m", color_enabled)
                << std::fixed << std::setprecision(6)
                << " | step_s=" << step.step_seconds
                << " | hydro_s=" << step.hydro_seconds
                << " | gravity_s=" << step.gravity_seconds
                << " | radiation_s=" << step.radiation_seconds
                << " | amr_s=" << step.amr_seconds;
    if(step.mode == "individual")
        phases_line << " | sync_s=" << step.sync_seconds
                    << " | prepare_s=" << step.prepare_seconds
                    << " | closure_s=" << step.closure_seconds
                    << " | suggest_s=" << step.suggest_seconds
                    << " | wake_s=" << step.wake_seconds
                    << " | sweep_s=" << step.sweep_seconds
                    << " | commit_s=" << step.commit_seconds
                    << " | other_s=" << step.other_seconds;

    std::ostringstream mesh_line;
    mesh_line << "  " << RuntimeLabel("mesh  ", "\033[33m", color_enabled)
              << std::fixed << std::setprecision(6)
              << " | mesh_s=" << step.mesh_seconds
              << " | mesh_builds=" << step.mesh_builds;
    if(step.mode == "individual")
        mesh_line << " | full_builds=" << step.full_builds;

    std::ostringstream source_line;
    source_line << "  " << RuntimeLabel("source", "\033[32m", color_enabled)
                << std::fixed << std::setprecision(6)
                << " | source_s=" << step.source_seconds;
    if(detailed)
        source_line << " | source_first_s=" << step.source_first_seconds
                    << " | source_second_s=" << step.source_second_seconds;
    source_line << " | source_pct="
                << (step.step_seconds > 0 ?
                    100 * step.source_seconds / step.step_seconds : 0)
                << " | source_calls=" << step.source_calls;

    std::cout << header_line.str() << std::endl;
    std::cout << time_line.str() << std::endl;
    std::cout << work_line.str() << std::endl;
    std::cout << phases_line.str() << std::endl;
    std::cout << mesh_line.str() << std::endl;
    std::cout << source_line.str() << std::endl;
    std::cout << std::endl;
}

void WriteRuntimeAMR(
    int const rank, bool const color_enabled, RuntimeAMROutput const& amr)
{
    if(rank != 0)
        return;
    std::cout << RuntimeLabel("RICH_AMR", "\033[1;33m", color_enabled)
              << " mode=" << amr.mode
              << " cycle=" << amr.cycle
              << " time=" << std::setprecision(runtime_simulation_digits)
              << amr.time
              << " cells_before=" << amr.cells_before
              << " added_cells=" << amr.added_cells
              << " removed_cells=" << amr.removed_cells
              << " cells_after=" << amr.cells_after
              << std::endl << std::endl;
}

void ReportRuntimeRetry(
    int const rank, bool const color_enabled,
    RuntimeStepOutput const& step, std::string const& physics,
    std::size_t const attempt, StepRetryRecord retry)
{
    retry.attempted_dt_min = RuntimeMinimum(retry.attempted_dt_min);
    retry.attempted_dt_max = RuntimeMaximum(retry.attempted_dt_max);
    retry.elapsed_seconds = RuntimeMaximum(retry.elapsed_seconds);
    std::uint64_t const local_cell =
        retry.representative_cell ==
            std::numeric_limits<std::size_t>::max() ?
        std::numeric_limits<std::uint64_t>::max() :
        static_cast<std::uint64_t>(retry.representative_cell);
    std::uint64_t const global_cell = RuntimeMinimum(local_cell);
    retry.representative_cell = global_cell ==
        std::numeric_limits<std::uint64_t>::max() ?
        std::numeric_limits<std::size_t>::max() :
        static_cast<std::size_t>(global_cell);

    if(rank != 0)
        return;

    std::ostringstream header_line;
    header_line << RuntimeLabel("RICH_RETRY", "\033[1;33m", color_enabled)
                << " mode=" << step.mode
                << " cycle=" << step.cycle
                << " physics=" << RuntimeToken(physics)
                << " attempt=" << attempt;

    std::ostringstream attempt_line;
    attempt_line << "  "
                 << RuntimeLabel("attempt", "\033[33m", color_enabled)
                 << " | active_cells=" << step.active_cells
                 << " | active_bins=" << step.active_bins
                 << std::setprecision(runtime_simulation_digits)
                 << " | attempted_dt_min=" << retry.attempted_dt_min
                 << " | attempted_dt_max=" << retry.attempted_dt_max;

    std::ostringstream failure_line;
    failure_line << "  "
                 << RuntimeLabel("failure", "\033[31m", color_enabled)
                 << std::fixed << std::setprecision(6)
                 << " | retry_s=" << retry.elapsed_seconds
                 << " | reason=" << RuntimeToken(retry.reason)
                 << " | cell=";
    if(retry.representative_cell ==
       std::numeric_limits<std::size_t>::max())
        failure_line << "none";
    else
        failure_line << retry.representative_cell;

    std::cout << header_line.str() << std::endl;
    std::cout << attempt_line.str() << std::endl;
    std::cout << failure_line.str() << std::endl;
    if(!retry.diagnostics.empty())
        std::cout << "  "
                  << RuntimeLabel("diagnostic", "\033[35m", color_enabled)
                  << " | " << retry.diagnostics << std::endl;
    std::cout << std::endl;
}

}

Simulation::Simulation(Tessellation3D &tess_, const std::vector<ComputationalCell3D> &cells_, EquationOfState &eos_, bool new_start) :
     tess(tess_), cells(cells_), extensives(cells_.size()), eos(eos_), Max_ID(0),
     wallclockTime(0), initializedFromRestart(!new_start)
#ifdef RICH_MPI
     , currentBox(tess_.GetBoxCoordinates())
#endif // RICH_MPI
{
    #ifdef RICH_MPI
        this->currentLoad = nullptr;
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    #else // RICH_MPI
        this->rank = 0;
        this->size = 1;
    #endif // RICH_MPI

    if(new_start)
    {
        this->initializeCellIDs();
    }
    else
    {
        this->recomputeMaxID();
    }

#ifdef RICH_MPI
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#endif

    size_t N = this->tess.GetPointNo();
    for(size_t i = 0; i < N; ++i)
    {
        PrimitiveToConserved(this->cells[i], this->tess.GetVolume(i), this->extensives[i]);
    }
}

void Simulation::initializeCellIDs(void)
{
    size_t N = this->cells.size();
    size_t nstart = 0;
#ifdef RICH_MPI
    std::vector<size_t> nrecv(static_cast<size_t>(this->size), 0);
    size_t nsend = N;
    MPI_Allgather(&nsend, 1, MPI_UNSIGNED_LONG_LONG, &nrecv[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    for (int i = 0; i < this->rank; ++i)
        nstart += nrecv[static_cast<size_t>(i)];
#endif
    for (size_t i = 0; i < N; ++i)
        this->cells[i].ID = nstart + i;
    this->Max_ID = nstart + N - 1;
#ifdef RICH_MPI
    for (int i = this->rank + 1; i < this->size; ++i)
        this->Max_ID += nrecv[static_cast<size_t>(i)];
#endif
}

void Simulation::recomputeMaxID(void)
{
    size_t N = this->cells.size();
    size_t maxid = 0;
    for (size_t i = 0; i < N; ++i)
        maxid = std::max(maxid, this->cells[i].ID);
#ifdef RICH_MPI
    MPI_Allreduce(&maxid, &this->Max_ID, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    this->Max_ID = maxid;
#endif
}

size_t &Simulation::GetMaxID(void)
{
    return this->Max_ID;
}

const size_t &Simulation::GetMaxID(void) const
{
    return this->Max_ID;
}

void Simulation::addPhysics(std::shared_ptr<PhysicsStep> physicsStep)
{
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual)
        requireIndividualTimeStepSupport(*physicsStep);
    this->physics.push_back(physicsStep);
}

void Simulation::EnableIndividualTimeSteps(IndividualTimeStepOptions options)
{
	if(this->individualScheduler && this->individualScheduler->initialized())
		throw std::logic_error("Individual timesteps are already active");
	for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
		requireIndividualTimeStepSupport(*physicsStep);
#ifdef RICH_MPI
	// Expose canonical owned-cell state immediately.  Callers may initialize or
	// customize bins before the first event; ghost copies must never enter the
	// stable-ID scheduler.
	this->cells.resize(this->tess.GetPointNo());
	this->extensives.resize(this->tess.GetPointNo());
#endif
	options.full_source_sweep_interval_minimum_steps =
		individualFullSourceSweepRuntimeInterval(
			options.full_source_sweep_interval_minimum_steps);
	this->individualScheduler = std::make_unique<IndividualTimeStepScheduler>(options);
    this->timeIntegrationMode = TimeIntegrationMode::Individual;
}

void Simulation::RequestSynchronizedIndividualEvent(void)
{
    // Every global step boundary is synchronized; nothing to request.
    if(this->timeIntegrationMode != TimeIntegrationMode::Individual)
        return;
    if(!this->individualScheduler)
        throw std::logic_error(
            "A synchronized event requires individual-timestep mode");
    if(this->individualEventInProgress)
        throw std::logic_error(
            "Cannot request a synchronized event during an individual event");
    this->individualSynchronizedEventRequested = true;
}

bool Simulation::IndividualStateSynchronized(void) const
{
    bool synchronized =
        this->timeIntegrationMode == TimeIntegrationMode::Individual &&
        !this->individualEventInProgress &&
        this->individualScheduler && this->individualScheduler->initialized() &&
        this->extensives.size() == this->cells.size() &&
        this->cells.size() == this->individualScheduler->states().size() &&
        this->tess.GetPointNo() == this->cells.size();
    if(synchronized)
    {
        std::uint64_t const current_tick =
            this->individualScheduler->currentTick();
        for(CellTimeState const& state : this->individualScheduler->states())
            if(state.begin_tick != current_tick ||
               state.last_primitive_tick != current_tick)
            {
                synchronized = false;
                break;
            }
    }
#ifdef RICH_MPI
    int synchronized_on_every_rank = synchronized ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &synchronized_on_every_rank, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    synchronized = synchronized_on_every_rank != 0;
#endif
    return synchronized;
}

bool Simulation::StateSynchronized(void) const
{
    if(this->timeIntegrationMode != TimeIntegrationMode::Individual)
        return !this->individualEventInProgress;
    return this->IndividualStateSynchronized();
}

namespace
{
    void RequireOnAllRanks(bool const valid, char const* message)
    {
        int ok = valid ? 1 : 0;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        if(ok == 0)
            throw std::logic_error(message);
    }
}

std::vector<Vector3D> Simulation::CommittedGeneratorPoints(void) const
{
    bool const individual =
        this->timeIntegrationMode == TimeIntegrationMode::Individual;
    std::size_t const count =
        individual ? this->cells.size() : this->tess.GetPointNo();
    std::vector<Vector3D> points;
    if(individual)
    {
        // As writeTessellation: the hydro step keeps the committed canonical
        // positions; the all-points array holds them when it does not.
        for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        {
            std::vector<Vector3D> candidate;
            if(physicsStep->getIndividualGeneratorPoints(candidate) &&
               candidate.size() >= count)
            {
                points.swap(candidate);
                break;
            }
        }
        if(points.size() < count)
            points = this->tess.getAllPoints();
    }
    else
        points = this->tess.getMeshPoints();
    RequireOnAllRanks(points.size() >= count,
        "Committed generator positions are missing on an MPI rank");
    points.resize(count);
    return points;
}

void Simulation::NotifyDomainChanged(void)
{
    AdaptiveModeState& a = this->adaptiveMode;
    ++a.domainEpoch;
    // The accumulated throughput mixes two workloads.  Measure again from
    // here, excluding the ramp again; a running probe keeps the wall it has
    // already spent against its budget, so the change does not lengthen it.
    a.stepsInMode = 0;
    a.wallMeasured = 0;
    a.simMeasured = 0;
    if(!a.probing)
        a.wallTotalInMode = 0;
    if(a.enabled && this->rank == 0)
        std::cout << std::setprecision(12)
                  << "RICH_MODE_DOMAIN_CHANGE cycle=" << this->tracker.getCycle()
                  << " time=" << this->tracker.getTime()
                  << " mode=" << (this->timeIntegrationMode ==
                      TimeIntegrationMode::Individual ? "individual" : "global")
                  << " probing=" << (a.probing ? 1 : 0)
                  << " epoch=" << a.domainEpoch << std::endl;
}

void Simulation::refreshCurrentAccelerationCaches(void)
{
    IndividualTimeStepScheduler& scheduler = *this->individualScheduler;
    std::vector<CellTimeState>& states = scheduler.states();
    std::size_t const count = this->cells.size();
    // The refresh evaluates owned cells in mesh order: it needs the full mesh
    // in the order of the owned arrays.
    bool aligned = this->tess.GetPointNo() == count && states.size() == count;
#ifdef RICH_MPI
    Tessellation3D::AllPointsMap const& build_map =
        this->tess.GetIndicesInAllPoints();
    for(std::size_t i = 0; aligned && i < count; ++i)
    {
        auto const entry = build_map.find(i);
        aligned = entry != build_map.end() && entry->second == i;
    }
    int aligned_everywhere = aligned ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &aligned_everywhere, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    aligned = aligned_everywhere != 0;
#endif
    std::vector<Vector3D> accelerations;
    if(aligned)
        for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
            if(physicsStep->refreshIndividualAccelerations(accelerations))
                break;
    // Present on some populated rank when a step refreshed (see the growth).
    int refreshed = accelerations.empty() ? 0 : 1;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &refreshed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if(refreshed != 0)
        RequireOnAllRanks(accelerations.size() == count,
            "AMR cache refresh returned the wrong number of accelerations");
    std::uint64_t const current_tick = scheduler.currentTick();
    for(std::size_t i = 0; i < states.size(); ++i)
    {
        if(states[i].begin_tick != current_tick)
            continue;
        if(refreshed != 0)
        {
            states[i].cached_acceleration = accelerations[i];
            states[i].gravity_half_kick_pending = true;
        }
        else
        {
            states[i].cached_acceleration = Vector3D();
            states[i].gravity_half_kick_pending = false;
        }
    }
}

Simulation::DomainGrowthReport
Simulation::GrowDomainAtSynchronizedIndividualState(
    Vector3D const& ll, Vector3D const& ur,
    std::vector<Vector3D> const& added_points,
    std::vector<ComputationalCell3D> const& added_cells)
{
    auto const start = std::chrono::high_resolution_clock::now();
    RequireOnAllRanks(
        this->timeIntegrationMode == TimeIntegrationMode::Individual &&
        this->individualScheduler && this->individualScheduler->initialized() &&
        !this->individualEventInProgress,
        "Domain growth requires individual mode between events");
    if(!this->IndividualStateSynchronized())
        throw std::logic_error(
            "Domain growth requires a synchronized individual state");
    RequireOnAllRanks(added_points.size() == added_cells.size(),
        "Domain growth received different numbers of points and cells");
#ifdef RICH_MPI
    // A registered buffer cannot be extended by the new cells, and the
    // legacy global resize does not move them either: refuse.
    RequireOnAllRanks(this->migrationBuffers.empty(),
        "Domain growth cannot extend registered migration buffers");
    std::shared_ptr<PhysicsStep> const balanceStep =
        this->findIndividualBalanceStep();
    RequireOnAllRanks(balanceStep != nullptr,
        "Domain growth needs a load-balancing physics step");
#endif
    IndividualTimeStepScheduler& scheduler = *this->individualScheduler;
    std::vector<CellTimeState>& states = scheduler.states();
    DomainGrowthReport report;

    // Before any hook releases them.
    std::vector<Vector3D> points = this->CommittedGeneratorPoints();
    std::size_t const count_before = this->cells.size();
    RequireOnAllRanks(states.size() == count_before &&
        this->extensives.size() == count_before,
        "Domain growth found misaligned individual state");

    unsigned long long max_existing_id = 0;
    unsigned long long min_added_id =
        std::numeric_limits<unsigned long long>::max();
    double before[2] = {0, 0};
    for(std::size_t i = 0; i < count_before; ++i)
    {
        max_existing_id = std::max(max_existing_id,
            static_cast<unsigned long long>(this->cells[i].ID));
        before[0] += this->extensives[i].mass;
        before[1] += this->extensives[i].energy;
    }
    for(ComputationalCell3D const& cell : added_cells)
        min_added_id = std::min(min_added_id,
            static_cast<unsigned long long>(cell.ID));
    unsigned long long counts[2] = {
        static_cast<unsigned long long>(count_before),
        static_cast<unsigned long long>(added_cells.size())};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_existing_id, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_added_id, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, before, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    report.cells_before = counts[0];
    report.added_cells = counts[1];
    report.mass_before = before[0];
    report.energy_before = before[1];
    // New cells are recognised by ID below, so every added ID must be fresh.
    RequireOnAllRanks(counts[1] == 0 || min_added_id > max_existing_id,
        "Domain growth needs added cell IDs above every existing ID");

    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        physicsStep->beforeIndividualRebalance();
#ifdef RICH_MPI
    balanceStep->beforeLB();
#endif

    // New cells open at the finest bin in use; their own limits, evaluated on
    // the rebuilt mesh below, can only shorten that.
    std::uint8_t const seed_bin = scheduler.minimumOccupiedBin();
    report.seed_bin = seed_bin;
    points.insert(points.end(), added_points.begin(), added_points.end());
    this->cells.insert(this->cells.end(), added_cells.begin(),
                       added_cells.end());
    this->extensives.resize(this->cells.size());
    for(ComputationalCell3D const& cell : added_cells)
        states.push_back(scheduler.synchronizedCellState(cell.ID, seed_bin));

    this->tess.SetBox(ll, ur);
    {
        MeshBuildTimer mesh_build_timer(report.mesh_build_timing);
        ++report.mesh_build_timing.full_builds;
#ifdef RICH_MPI
        // Rebalance and exchange, as the legacy resize: SetBox replaced the
        // decomposition, and the added points may lie on one rank.
        this->tess.BuildParallel(points);
#else
        this->tess.Build(points);
#endif
    }
    std::vector<Vector3D>().swap(points);
#ifdef RICH_MPI
    // One exchange for every per-cell array, in the order of the passed
    // points, as buildDataTransfer does.
    MPI_exchange_data(this->tess, this->extensives, false);
    MPI_exchange_data(this->tess, this->cells, false);
    MPI_exchange_data(this->tess, states, false);
#endif
    std::size_t const count_after = this->tess.GetPointNo();
    this->cells.resize(count_after);
    this->extensives.resize(count_after);
    // rebuildIndex overwrites state IDs; check the pairing first.  An
    // exchanging build indexes its owned points in post-exchange order, the
    // order the arrays above now have (the neighbour closure below maps mesh
    // points to states through GetIndicesInAllPoints).
    bool aligned = states.size() == count_after;
    for(std::size_t i = 0; aligned && i < count_after; ++i)
        aligned = states[i].cell_id == this->cells[i].ID;
#ifdef RICH_MPI
    Tessellation3D::AllPointsMap const& build_map =
        this->tess.GetIndicesInAllPoints();
    for(std::size_t i = 0; aligned && i < count_after; ++i)
    {
        auto const entry = build_map.find(i);
        aligned = entry != build_map.end() && entry->second == i;
    }
#endif
    RequireOnAllRanks(aligned,
        "Domain growth separated a scheduler state from its cell");
    scheduler.rebuildIndex(this->cells);
    unsigned long long total_after =
        static_cast<unsigned long long>(count_after);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &total_after, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    report.cells_after = total_after;
    RequireOnAllRanks(total_after == report.cells_before + report.added_cells,
        "Domain growth changed the number of existing cells");

    // New cells, and old cells whose volume changed: at a synchronized state
    // the committed volume is mass / density.
    std::vector<unsigned char> reseed(count_after, 0);
    unsigned long long reseeded = 0;
    for(std::size_t i = 0; i < count_after; ++i)
    {
        bool changed = static_cast<unsigned long long>(this->cells[i].ID) >
            max_existing_id;
        if(!changed)
        {
            double const committed_volume =
                this->extensives[i].mass / this->cells[i].density;
            double const volume = this->tess.GetVolume(i);
            changed = !(std::isfinite(committed_volume) &&
                        committed_volume > 0) ||
                std::abs(volume - committed_volume) > 1e-8 * committed_volume;
        }
        if(changed)
        {
            reseed[i] = 1;
            ++reseeded;
        }
    }
    // The legacy resize's semantics (user's choice, 2026-09-24): every
    // extensive from its primitive on the new mesh.
    double after[4] = {0, 0, 0, 0};
    for(std::size_t i = 0; i < count_after; ++i)
    {
        PrimitiveToConserved(this->cells[i], this->tess.GetVolume(i),
                             this->extensives[i]);
        after[0] += this->extensives[i].mass;
        after[1] += this->extensives[i].energy;
        if(static_cast<unsigned long long>(this->cells[i].ID) > max_existing_id)
        {
            after[2] += this->extensives[i].mass;
            after[3] += this->extensives[i].energy;
        }
    }
    // Fresh limits on the rebuilt mesh before anything advances: the next
    // event runs its physics before it refreshes limits at commit.  Every
    // cell, since shape, faces and neighbours can change without a volume
    // change: the individual event's rule (wave-speed CFL, source limits,
    // mesh-drift guard), with the velocities each generator moves with
    // through its next interval (committed at its last event; new cells at
    // rest).  A step that keeps the acceleration cache returns fresh
    // accelerations of the rebuilt state, which every next first half kick
    // then uses; without them the cache is invalidated.
    std::vector<Vector3D> point_velocities(count_after);
    for(std::size_t i = 0; i < count_after; ++i)
        point_velocities[i] = states[i].point_velocity;
    std::vector<double> limits(count_after,
                               std::numeric_limits<double>::infinity());
    std::vector<Vector3D> accelerations;
    bool have_limits = false;
    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
    {
        std::vector<double> step_limits;
        std::vector<Vector3D> step_accelerations;
        if(!physicsStep->synchronizedCellTimeStepLimits(point_velocities,
                                                        step_limits,
                                                        step_accelerations))
            continue;
        have_limits = true;
        RequireOnAllRanks(step_limits.size() >= count_after,
            "Domain growth received too few per-cell limits");
        for(std::size_t i = 0; i < count_after; ++i)
            limits[i] = std::min(limits[i], step_limits[i]);
        if(!step_accelerations.empty())
            accelerations.swap(step_accelerations);
    }
    RequireOnAllRanks(have_limits,
        "Domain growth needs a physics step with per-cell limits on the rebuilt mesh");
    double smallest_limit = std::numeric_limits<double>::infinity();
    for(double const limit : limits)
        smallest_limit = std::min(smallest_limit, limit);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &smallest_limit, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    report.smallest_limit = smallest_limit;
    RequireOnAllRanks(smallest_limit > 0 &&
        smallest_limit >= scheduler.timeQuantum(),
        "A cell limit after domain growth is below the individual time quantum");
    std::vector<std::uint8_t> maximum_bins(count_after);
    for(std::size_t i = 0; i < count_after; ++i)
    {
        std::uint8_t bin = scheduler.binForTimeStep(limits[i]);
        // New and volume-changed cells also start no coarser than the finest
        // bin in use.
        if(reseed[i] != 0)
            bin = std::min(bin, seed_bin);
        maximum_bins[i] = bin;
    }
    unsigned long long shortened = static_cast<unsigned long long>(
        scheduler.limitSynchronizedBins(maximum_bins));
    // A step that keeps the cache fills one acceleration per owned cell, so
    // it is present on some populated rank; a rank that owns no cells has
    // none either way.
    int refreshed = accelerations.empty() ? 0 : 1;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &refreshed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if(refreshed != 0)
        RequireOnAllRanks(accelerations.size() == count_after,
            "Domain growth refreshed accelerations on some ranks only");
    if(refreshed != 0)
    {
        for(std::size_t i = 0; i < count_after; ++i)
        {
            states[i].cached_acceleration = accelerations[i];
            states[i].gravity_half_kick_pending = true;
        }
        report.accelerations_refreshed = true;
    }
    else
        scheduler.invalidateCachedAccelerations();
    if(!scheduler.options().force_synchronized)
        scheduler.enforceNeighborBinClosure(this->tess, this->cells);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, after, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &reseeded, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &shortened, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    report.mass_after = after[0];
    report.energy_after = after[1];
    report.inserted_mass = after[2];
    report.inserted_energy = after[3];
    report.reseeded_cells = reseeded;
    report.shortened_cells = shortened;

#ifdef RICH_MPI
    // As rebalanceCommittedIndividualState: a new ownership epoch and the
    // decomposition SetBox created as the current load balance.
    this->lastRebalanceCycle = this->tracker.getCycle();
    ++this->individualOwnershipEpoch;
    std::string const load_name = balanceStep->getRequiredLB();
    this->currentLoad = this->tess.GetLoadBalancer();
    this->loads[load_name] = this->currentLoad;
    this->currentLB = load_name;
    this->individualActiveHilbertBoundaries.clear();
    // The new domain returns to positional ownership: segmented-ownership
    // references measured on the old domain do not apply.
    this->individualSegmentDecision = IndividualSegmentDecisionState();
    this->lastIndividualEventLedgerLocalSeconds = 0;
    balanceStep->afterLB();
#endif
    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        physicsStep->afterIndividualAMR();
    this->NotifyDomainChanged();

    report.seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &report.seconds, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    return report;
}

// ---------------------------------------------------------------------------
// Adaptive integration mode
//
// The simulation decides for itself whether individual timesteps pay.  Two
// measurements drive it.  The realised throughput of the current mode,
// simulated time per wall second, is accumulated over a dwell window; the
// other mode is then probed with a wall-time budget that is a fraction of
// that window, and the faster mode is kept with a margin.  While stepping
// globally the per-cell CFL distribution gives an upper bound on the gain of
// any individual scheme (global cell-updates over ideal individual
// cell-updates); below a minimum there is little to gain by construction and
// no probe is spent.  Dwell windows double after every decision that confirms
// the current mode, so probe overhead shrinks over time.  Switches happen at
// synchronized states only: out of individual mode after one all-active
// event, into it from any global step boundary with a fresh scheduler whose
// quantum follows the global step of that moment.  Every quantity below is a
// collectively reduced value, so all ranks take the same decision.
namespace
{
struct AdaptiveIntegrationRuntimeOptions
{
    bool configured = false;
    bool enabled = false;
    std::size_t dwell_min_steps = 64;
    std::size_t minimum_samples = 6;
    std::size_t ramp_individual_events = 12;
    std::size_t ramp_global_steps = 2;
    std::size_t gate_interval_steps = 16;
    std::size_t dwell_backoff_cap = 16;
    double probe_fraction = 0.1;
    double margin = 1.15;
    double gain_minimum = 1.5;
};

AdaptiveIntegrationRuntimeOptions const& adaptiveIntegrationRuntimeOptions()
{
    static AdaptiveIntegrationRuntimeOptions const options = []()
    {
        AdaptiveIntegrationRuntimeOptions result;
        bool locally_valid = true;
        char const* const configured =
            std::getenv("RICH_INDIVIDUAL_ADAPTIVE_MODE");
        result.configured = configured != nullptr && configured[0] != '\0';
        result.enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_ADAPTIVE_MODE", false, locally_valid);
        result.dwell_min_steps = parseEnvironmentSize(
            "RICH_ADAPTIVE_DWELL_MIN_STEPS", 64, locally_valid);
        result.minimum_samples = parseEnvironmentSize(
            "RICH_ADAPTIVE_MIN_SAMPLES", 6, locally_valid);
        result.ramp_individual_events = parseEnvironmentSize(
            "RICH_ADAPTIVE_RAMP_EVENTS", 12, locally_valid);
        result.ramp_global_steps = parseEnvironmentSize(
            "RICH_ADAPTIVE_RAMP_STEPS", 2, locally_valid);
        result.gate_interval_steps = parseEnvironmentSize(
            "RICH_ADAPTIVE_GATE_INTERVAL", 16, locally_valid);
        result.dwell_backoff_cap = parseEnvironmentSize(
            "RICH_ADAPTIVE_DWELL_BACKOFF_CAP", 16, locally_valid);
        result.probe_fraction = parseEnvironmentDouble(
            "RICH_ADAPTIVE_PROBE_FRACTION", 0.1, 0.0, locally_valid);
        result.margin = parseEnvironmentDouble(
            "RICH_ADAPTIVE_MARGIN", 1.15, 1.0, locally_valid);
        result.gain_minimum = parseEnvironmentDouble(
            "RICH_ADAPTIVE_GAIN_MIN", 1.5, 1.0, locally_valid);
        int mask = (result.configured ? 1 : 0) | (result.enabled ? 2 : 0) |
            (locally_valid ? 0 : 4);
        unsigned long long sizes[6] = {
            result.dwell_min_steps, result.minimum_samples,
            result.ramp_individual_events, result.ramp_global_steps,
            result.gate_interval_steps, result.dwell_backoff_cap};
        double values[3] = {result.probe_fraction, result.margin,
                            result.gain_minimum};
#ifdef RICH_MPI
        int minimum_mask = mask;
        int maximum_mask = mask;
        unsigned long long minimum_sizes[6];
        unsigned long long maximum_sizes[6];
        double minimum_values[3];
        double maximum_values[3];
        std::copy(std::begin(sizes), std::end(sizes), minimum_sizes);
        std::copy(std::begin(sizes), std::end(sizes), maximum_sizes);
        std::copy(std::begin(values), std::end(values), minimum_values);
        std::copy(std::begin(values), std::end(values), maximum_values);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minimum_sizes, 6, MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maximum_sizes, 6, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minimum_values, 3, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maximum_values, 3, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        bool const consistent = minimum_mask == maximum_mask &&
            std::equal(std::begin(minimum_sizes), std::end(minimum_sizes),
                       std::begin(maximum_sizes)) &&
            std::equal(std::begin(minimum_values), std::end(minimum_values),
                       std::begin(maximum_values));
        if((maximum_mask & 4) != 0 || !consistent)
            throw std::invalid_argument(
                "Invalid or inconsistent adaptive integration mode environment options");
#else
        if((mask & 4) != 0)
            throw std::invalid_argument(
                "Invalid adaptive integration mode environment options");
#endif
        return result;
    }();
    return options;
}

char const* IntegrationModeName(TimeIntegrationMode mode)
{
    return mode == TimeIntegrationMode::Individual ? "individual" : "global";
}
}

bool Simulation::AdaptiveIntegrationModeWillEnable(bool requested)
{
    AdaptiveIntegrationRuntimeOptions const& runtime =
        adaptiveIntegrationRuntimeOptions();
    return runtime.configured ? runtime.enabled : requested;
}

void Simulation::SetAdaptiveIntegrationMode(bool enabled,
                                            IndividualTimeStepOptions options)
{
    AdaptiveIntegrationRuntimeOptions const& runtime =
        adaptiveIntegrationRuntimeOptions();
    if(runtime.configured)
        enabled = runtime.enabled;
    if(enabled)
        for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
            requireIndividualTimeStepSupport(*physicsStep);
    options.full_source_sweep_interval_minimum_steps =
        individualFullSourceSweepRuntimeInterval(
            options.full_source_sweep_interval_minimum_steps);
    // Individual mode is entered with a fresh scheduler whose quantum is
    // derived from the global step of that moment.
    options.time_quantum = 0;
    this->adaptiveMode = AdaptiveModeState();
    this->adaptiveMode.enabled = enabled;
    this->adaptiveMode.individualOptions = options;
    if(enabled && this->rank == 0)
        std::cout << std::setprecision(6)
                  << "RICH_MODE_CONTROLLER enabled=1"
                  << " start_mode=" << IntegrationModeName(this->timeIntegrationMode)
                  << " dwell_min_steps=" << runtime.dwell_min_steps
                  << " minimum_samples=" << runtime.minimum_samples
                  << " ramp_events=" << runtime.ramp_individual_events
                  << " ramp_steps=" << runtime.ramp_global_steps
                  << " probe_fraction=" << runtime.probe_fraction
                  << " margin=" << runtime.margin
                  << " gain_min=" << runtime.gain_minimum
                  << " gate_interval=" << runtime.gate_interval_steps
                  << " dwell_backoff_cap=" << runtime.dwell_backoff_cap
                  << std::endl;
}

void Simulation::adaptiveResetWindow(void)
{
    this->adaptiveMode.stepsInMode = 0;
    this->adaptiveMode.wallTotalInMode = 0;
    this->adaptiveMode.wallMeasured = 0;
    this->adaptiveMode.simMeasured = 0;
}

void Simulation::adaptiveLogDecision(TimeIntegrationMode mode,
                                     char const* action, double tau) const
{
    if(this->rank != 0)
        return;
    AdaptiveModeState const& a = this->adaptiveMode;
    std::cout << std::setprecision(12)
              << "RICH_MODE_DECISION cycle=" << this->tracker.getCycle()
              << " time=" << this->tracker.getTime()
              << " mode=" << IntegrationModeName(mode)
              << " phase=" << (a.probing ? "probe" : "dwell")
              << " steps=" << a.stepsInMode
              << " wall_s=" << a.wallTotalInMode
              << " tau=" << tau
              << " tau_individual=" << a.tauIndividual
              << " tau_global=" << a.tauGlobal
              << " gain_bound=" << a.gainBound
              << " gain_bound_uniform_caps=" << a.gainBoundUniformCaps
              << " dwell_multiplier=" << a.dwellMultiplier
              << " action=" << action << std::endl;
}

void Simulation::adaptiveRequestSwitch(TimeIntegrationMode target,
                                       std::string const& reason)
{
    if(target == this->timeIntegrationMode)
        return;
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual)
    {
        // Leaving individual mode needs one all-active event so that every
        // cell is committed at the same tick; the switch follows it.
        this->adaptiveMode.switchToGlobalRequested = true;
        this->adaptiveMode.pendingReason = reason;
        this->RequestSynchronizedIndividualEvent();
        return;
    }
    this->adaptiveEnterIndividual(reason);
}

// Collective (every rank calls it at the same switch).  The switch itself does
// no physics, only ownership and ghost bookkeeping, so before/after sums must
// agree to round-off (plan S7 gate b).
void Simulation::adaptiveLogSwitchState(char const* phase, char const* direction,
                                        std::size_t owned, double next_dt,
                                        double scheduler_next_dt) const
{
    double sums[8] = {static_cast<double>(std::min(owned, this->extensives.size())),
                      0, 0, 0, 0, 0, 0, 0};
    for(std::size_t i = 0; i < this->extensives.size() && i < owned; ++i)
    {
        Conserved3D const& e = this->extensives[i];
        sums[1] += e.mass;
        sums[2] += e.momentum.x;
        sums[3] += e.momentum.y;
        sums[4] += e.momentum.z;
        sums[5] += std::abs(e.momentum.x) + std::abs(e.momentum.y) +
            std::abs(e.momentum.z);
        sums[6] += e.energy;
        sums[7] += e.Erad;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, sums, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(this->rank != 0)
        return;
    std::cout << std::setprecision(17)
              << "RICH_MODE_SWITCH_STATE cycle=" << this->tracker.getCycle()
              << " time=" << this->tracker.getTime()
              << " phase=" << phase << " direction=" << direction
              << " owned_cells=" << static_cast<unsigned long long>(sums[0])
              << " mass=" << sums[1] << " momentum_x=" << sums[2]
              << " momentum_y=" << sums[3] << " momentum_z=" << sums[4]
              << " momentum_abs=" << sums[5] << " energy=" << sums[6]
              << " erad=" << sums[7] << " next_dt=" << next_dt
              << " scheduler_next_dt=" << scheduler_next_dt << std::endl;
}

void Simulation::adaptiveEnterGlobal(std::string const& reason)
{
    if(!this->individualScheduler)
        throw std::logic_error(
            "Cannot leave individual mode without a scheduler");
    double next_dt = this->individualScheduler->nextEventTimeStep();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &next_dt, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    double const scheduler_next_dt = next_dt;
    if(!(std::isfinite(next_dt) && next_dt > 0))
        next_dt = this->lastStepAdvance;
    // The first global step may be as long as the finest occupied bin: every
    // cell's bin lies within its own limits, so that length is legal for all
    // cells (and the global CFL evaluation still caps it).  The next event of
    // the individual schedule can be far shorter (a mid-interval shortening
    // cascade put it at 5e-5 against bins of 1e-3 on the TDE); starting the
    // global probe there, with the step growing 1.25x per step, spent the
    // whole probe ramping and made global mode look slower than it is.
    {
        double const finest_bin_dt = std::ldexp(
            this->individualScheduler->timeQuantum(),
            static_cast<int>(this->individualScheduler->minimumOccupiedBin()));
        if(std::isfinite(finest_bin_dt) && finest_bin_dt > next_dt)
            next_dt = finest_bin_dt;
    }
#ifdef RICH_MPI
    // A segmented partition (several curve ranges per rank, built for
    // individual events) returns to one weighted range per rank before the
    // global steps, while the scheduler state it migrates still exists.  The
    // weighted rebalance cuts positional ranges (HilbertLoadBalancer::
    // rebalance).  Agreed across ranks; a failure aborts the switch.
    {
        std::shared_ptr<HilbertLoadBalancer<Vector3D>> const hilbert =
            std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
                this->tess.GetLoadBalancer());
        int const local_segmented =
            hilbert && !hilbert->positionalOwnership() ? 1 : 0;
        int segmented_extrema[2] = {local_segmented, -local_segmented};
        MPI_Allreduce(MPI_IN_PLACE, segmented_extrema, 2, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if(segmented_extrema[0] != -segmented_extrema[1])
            throw std::logic_error(
                "Segmented Hilbert ownership differs across MPI ranks");
        if(segmented_extrema[0] != 0)
        {
            std::shared_ptr<PhysicsStep> const balance_step =
                this->findIndividualBalanceStep();
            if(!balance_step)
                throw std::logic_error(
                    "Segmented Hilbert ownership has no balancing physics step");
            IndividualRebalanceResult const collapse =
                this->rebalanceToPositionalOwnership(balance_step);
            std::shared_ptr<HilbertLoadBalancer<Vector3D>> const collapsed =
                std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
                    this->tess.GetLoadBalancer());
            int positional = collapse.applied && collapsed &&
                collapsed->positionalOwnership() ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &positional, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            if(positional == 0)
                throw std::logic_error(
                    "Segmented Hilbert ownership did not collapse before global mode");
            this->individualSegmentMigrationSeconds += collapse.maximumSeconds;
            if(this->rank == 0)
                std::cout << "RICH_SEGMENT_COLLAPSE cycle=" << this->tracker.getCycle()
                          << " migrated_cells=" << collapse.migratedCells
                          << " seconds=" << collapse.maximumSeconds << std::endl;
        }
        // The next individual period measures afresh: its positional
        // reference, ledger and previous-event sample belong to it.
        this->individualSegmentDecision = IndividualSegmentDecisionState();
        this->lastIndividualEventLedgerLocalSeconds = 0;
    }
#endif
    // Individual mode keeps owned-only canonical extensives.
    this->adaptiveLogSwitchState("before", "individual_to_global",
        this->extensives.size(), next_dt, scheduler_next_dt);
    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
        physicsStep->beforeIndividualRebalance();
    this->individualScheduler.reset();
    this->timeIntegrationMode = TimeIntegrationMode::Global;
    this->individualSynchronizedEventRequested = false;
    this->adaptiveMode.switchToGlobalRequested = false;
#ifdef RICH_MPI
    this->individualActiveHilbertBoundaries.clear();
    // The global path keeps ghost primitives behind the owned cells; the
    // committed individual state is owned-only.
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#endif
    this->tsc->SetTimeStep(next_dt);
    ++this->adaptiveMode.switches;
    this->adaptiveLogSwitchState("after", "individual_to_global",
        this->tess.GetPointNo(), next_dt, scheduler_next_dt);
    if(this->rank == 0)
        std::cout << std::setprecision(12)
                  << "RICH_MODE_SWITCH cycle=" << this->tracker.getCycle()
                  << " time=" << this->tracker.getTime()
                  << " from=individual to=global reason=\"" << reason << "\""
                  << " tau_individual=" << this->adaptiveMode.tauIndividual
                  << " tau_global=" << this->adaptiveMode.tauGlobal
                  << " gain_bound=" << this->adaptiveMode.gainBound
                  << " first_global_dt=" << next_dt
                  << " switches=" << this->adaptiveMode.switches << std::endl;
    this->adaptiveResetWindow();
}

void Simulation::adaptiveEnterIndividual(std::string const& reason)
{
    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
        requireIndividualTimeStepSupport(*physicsStep);
    // Global mode owns the first GetPointNo() cells.
    this->adaptiveLogSwitchState("before", "global_to_individual",
        this->tess.GetPointNo(), this->tsc->GetTimeStep(),
        std::numeric_limits<double>::quiet_NaN());
    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
    {
        physicsStep->beforeIndividualRebalance();
        physicsStep->afterIndividualAMR();
    }
#ifdef RICH_MPI
    this->cells.resize(this->tess.GetPointNo());
    this->extensives.resize(this->tess.GetPointNo());
    this->individualActiveHilbertBoundaries.clear();
    this->individualSegmentDecision = IndividualSegmentDecisionState();
    this->lastIndividualEventLedgerLocalSeconds = 0;
#endif
    double dt = this->tsc->GetTimeStep();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    if(!(std::isfinite(dt) && dt > 0))
        dt = this->lastStepAdvance;
    this->tsc->SetTimeStep(dt);
    this->individualScheduler = std::make_unique<IndividualTimeStepScheduler>(
        this->adaptiveMode.individualOptions);
    this->timeIntegrationMode = TimeIntegrationMode::Individual;
    this->individualSynchronizedEventRequested = false;
    ++this->adaptiveMode.switches;
    this->adaptiveLogSwitchState("after", "global_to_individual",
        this->extensives.size(), dt, std::numeric_limits<double>::quiet_NaN());
    if(this->rank == 0)
        std::cout << std::setprecision(12)
                  << "RICH_MODE_SWITCH cycle=" << this->tracker.getCycle()
                  << " time=" << this->tracker.getTime()
                  << " from=global to=individual reason=\"" << reason << "\""
                  << " tau_individual=" << this->adaptiveMode.tauIndividual
                  << " tau_global=" << this->adaptiveMode.tauGlobal
                  << " gain_bound=" << this->adaptiveMode.gainBound
                  << " initial_dt=" << dt
                  << " switches=" << this->adaptiveMode.switches << std::endl;
    this->adaptiveResetWindow();
}

double Simulation::adaptiveGainBound(void)
{
    // Every step with per-cell limits contributes cell by cell; the others act
    // as a uniform cap, as in a global step.  A step counts as per-cell only
    // when every rank has its limits, so all ranks bound the same quantity.
    // The legacy bound, the form before radiation had per-cell limits, takes
    // only limits evaluated on the current state (hydro) per cell and every
    // cached per-cell step (radiation) as its uniform cap; it is logged in
    // RICH_MODE_GAIN beside the bound.
    // The hydro limits need the face velocities of the mesh they are
    // evaluated on, known only for the mesh the hydro step leaves.  When that
    // mesh was rebuilt since on any rank (the post-step callback's AMR pass, a
    // rebalance after the hydro step), the bound is skipped rather than formed
    // with hydro as a cap, and the next global step retries it.
    std::size_t const count = this->physics.size();
    std::vector<std::vector<double>> own(count);
    std::vector<int> per_cell(count, 0);
    int stale = 0;
    for(std::size_t index = 0; index < count; ++index)
    {
        per_cell[index] =
            this->physics[index]->collectCellTimeStepLimits(own[index]) ? 1 : 0;
        if(this->physics[index]->cellTimeStepLimitsStale())
            stale = 1;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &stale, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    this->adaptiveMode.gainRetry = stale != 0;
    if(stale != 0)
    {
        std::size_t const streak = ++this->adaptiveMode.gainSkipStreak;
        // First skip of a streak, then at powers of two.
        if(this->rank == 0 && (streak & (streak - 1)) == 0)
            std::cout << std::setprecision(12)
                      << "RICH_MODE_GAIN_SKIPPED cycle=" << this->tracker.getCycle()
                      << " time=" << this->tracker.getTime()
                      << " reason=stale_cell_limits consecutive=" << streak
                      << " retry=next_global_step" << std::endl;
        this->adaptiveMode.gainBoundUniformCaps =
            std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::quiet_NaN();
    }
    this->adaptiveMode.gainSkipStreak = 0;
    std::size_t provider = count;
    // Twice: agree on which steps have limits, then on which of them cover
    // the same cells one to one.
    for(int pass = 0; pass < 2; ++pass)
    {
        if(pass == 1)
            for(std::size_t index = 0; index < count; ++index)
                if(per_cell[index] != 0 &&
                   own[index].size() != own[provider].size())
                    per_cell[index] = 0;
#ifdef RICH_MPI
        if(count > 0)
            MPI_Allreduce(MPI_IN_PLACE, per_cell.data(),
                          static_cast<int>(count), MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
#endif
        provider = count;
        for(std::size_t index = 0; index < count && provider == count; ++index)
            if(per_cell[index] != 0)
                provider = index;
        if(provider == count)
            return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t const cell_count = own[provider].size();
    bool have_legacy = false;
    for(std::size_t index = 0; index < count; ++index)
        if(per_cell[index] != 0 &&
           !this->physics[index]->cellTimeStepLimitsCached())
            have_legacy = true;
    std::vector<double> limits(cell_count, std::numeric_limits<double>::infinity());
    std::vector<double> legacy(cell_count, std::numeric_limits<double>::infinity());
    // Per step: its smallest per-cell limit or its uniform cap (MIN), and the
    // cells whose limit it sets (SUM).
    std::vector<double> minima(count, std::numeric_limits<double>::infinity());
    std::vector<double> binding(count, 0);
    double fallbacks = 0;
    // Each step's global limit: the cap of a step without per-cell limits,
    // and of a cached per-cell step in the legacy bound.
    std::vector<double> caps(count, std::numeric_limits<double>::infinity());
    for(std::size_t index = 0; index < count; ++index)
    {
        double const cap = this->physics[index]->suggestTimeStep();
        if(std::isfinite(cap) && cap > 0)
            caps[index] = cap;
        bool const cells_here = per_cell[index] != 0;
        bool const cells_legacy = cells_here &&
            !this->physics[index]->cellTimeStepLimitsCached();
        if(cells_here)
        {
            fallbacks += static_cast<double>(
                this->physics[index]->cellTimeStepLimitFallbacks());
            for(double const limit : own[index])
                if(limit > 0)
                    minima[index] = std::min(minima[index], limit);
            double const own_minimum =
                this->physics[index]->cellTimeStepLimitMinimum();
            if(own_minimum > 0)
                minima[index] = std::min(minima[index], own_minimum);
        }
        else
            minima[index] = caps[index];
        for(std::size_t cell = 0; cell < cell_count; ++cell)
        {
            limits[cell] = std::min(limits[cell],
                cells_here ? own[index][cell] : caps[index]);
            legacy[cell] = std::min(legacy[cell],
                cells_legacy ? own[index][cell] : caps[index]);
        }
    }
    for(std::size_t cell = 0; cell < cell_count; ++cell)
    {
        std::size_t winner = count;
        double smallest = std::numeric_limits<double>::infinity();
        for(std::size_t index = 0; index < count; ++index)
        {
            double const limit = per_cell[index] != 0 ?
                own[index][cell] : caps[index];
            if(limit > 0 && limit < smallest)
            {
                smallest = limit;
                winner = index;
            }
        }
        if(winner < count)
            binding[winner] += 1;
    }
    // Ideal global steps: the smallest limit of each set (MIN).  The bound's
    // also takes every per-cell step's own minimum, so a cell left without a
    // limit here (migrated) cannot raise the reference step: coverage gaps
    // only move cells to coarser bins.
    std::vector<double> smallest(2 + count);
    smallest[0] = smallest[1] = std::numeric_limits<double>::infinity();
    for(std::size_t cell = 0; cell < cell_count; ++cell)
    {
        if(std::isfinite(limits[cell]) && limits[cell] > 0)
            smallest[0] = std::min(smallest[0], limits[cell]);
        if(std::isfinite(legacy[cell]) && legacy[cell] > 0)
            smallest[1] = std::min(smallest[1], legacy[cell]);
    }
    for(std::size_t index = 0; index < count; ++index)
        if(per_cell[index] != 0 && std::isfinite(minima[index]))
            smallest[0] = std::min(smallest[0], minima[index]);
    std::copy(minima.begin(), minima.end(), smallest.begin() + 2);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, smallest.data(),
                  static_cast<int>(smallest.size()), MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    double const dt_global = smallest[0];
    double const dt_legacy = smallest[1];
    if(!(std::isfinite(dt_global) && dt_global > 0))
        return std::numeric_limits<double>::quiet_NaN();
    // The grid an individual phase would use: m times the last global step's
    // CFL/source suggestion under an anchor margin (as initialization), else
    // the smallest limit.  Bins may fall below initial_bin (radiation-limited
    // cells); update counts are normalized against the combined limit.
    double const anchor_reference = this->adaptiveMode.anchorReferenceStep;
    double const dt_grid = IndividualBinAnchorMargin() > 0 && anchor_reference > 0 ?
        IndividualBinAnchorMargin() * anchor_reference : dt_global;
    double const anchor_scale = dt_grid / dt_global;
    IndividualTimeStepOptions const& options =
        this->adaptiveMode.individualOptions;
    int const bin_span = options.maximum_bin > options.initial_bin ?
        static_cast<int>(options.maximum_bin - options.initial_bin) : 0;
    // Ideal individual cell-updates per global step: each cell steps at the
    // power-of-two bin its limit allows, the finest bin being the global step.
    // Layout: new bound {sum 2^-bin, cells}, legacy bound {same}, fallbacks,
    // cells per bin of the new bound, cells bound per step.
    std::size_t const bins = static_cast<std::size_t>(bin_span) + 1;
    std::vector<double> totals(5 + bins + count, 0);
    // The top bin is decided before the ratio is formed, so the ratio stays
    // below 2^bin_span and its bin converts to int safely.
    // Below the grid's finest bin only on an anchored grid (limit < dt_min).
    auto bin_of = [bin_span](double limit, double dt_min)
    {
        if(!std::isfinite(limit) || std::ldexp(limit, -bin_span) >= dt_min)
            return bin_span;
        double const bin = std::floor(std::log2(limit / dt_min));
        return bin > -60 ? static_cast<int>(bin) : -60;
    };
    // Under the bin-spread cap (IndividualTimeStepScheduler::maximumBinSpread)
    // no bin exceeds the finest bin over all ranks by more than K.
    int finest_bin = bin_span;
    for(std::size_t cell = 0; cell < cell_count; ++cell)
        if(limits[cell] > 0)
            finest_bin = std::min(finest_bin, bin_of(limits[cell], dt_grid));
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &finest_bin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    int const spread = IndividualTimeStepScheduler::maximumBinSpread();
    for(std::size_t cell = 0; cell < cell_count; ++cell)
    {
        if(limits[cell] > 0)
        {
            int const bin = spread >= 0 ? std::min(bin_of(limits[cell], dt_grid), finest_bin + spread) :
                bin_of(limits[cell], dt_grid);
            totals[0] += std::ldexp(1.0, -bin);
            totals[1] += 1;
            totals[5 + static_cast<std::size_t>(std::max(bin, 0))] += 1;
        }
        if(legacy[cell] > 0 && std::isfinite(dt_legacy) && dt_legacy > 0)
        {
            totals[2] += std::ldexp(1.0, -bin_of(legacy[cell], dt_legacy));
            totals[3] += 1;
        }
    }
    totals[4] = fallbacks;
    std::copy(binding.begin(), binding.end(), totals.begin() + 5 + bins);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, totals.data(), static_cast<int>(totals.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    // Under an anchor margin the grid's bins last 2^bin * dt_grid (dt_grid = m
    // times the last global step's hydro/source suggestion): the update count
    // on that grid, normalized against the combined global limit dt_global.
    double const bound = totals[0] > 0 && totals[1] > 0 ?
        anchor_scale * totals[1] / totals[0] : std::numeric_limits<double>::quiet_NaN();
    this->adaptiveMode.gainBoundUniformCaps =
        have_legacy && totals[2] > 0 && totals[3] > 0 ?
        totals[3] / totals[2] : std::numeric_limits<double>::quiet_NaN();
    if(this->rank == 0)
    {
        std::ostringstream line;
        line << std::setprecision(6)
             << "RICH_MODE_GAIN cycle=" << this->tracker.getCycle()
             << " time=" << std::setprecision(12) << this->tracker.getTime()
             << std::setprecision(6)
             << " gain_bound=" << bound
             << " gain_bound_uniform_caps="
             << this->adaptiveMode.gainBoundUniformCaps
             << " dt_cell_min=" << dt_global << " limits=";
        for(std::size_t index = 0; index < count; ++index)
            line << (index ? "," : "") << this->physics[index]->getName()
                 << ':' << (per_cell[index] != 0 ? "cell" : "cap") << ':'
                 << smallest[2 + index];
        line << " bound_cells=";
        for(std::size_t index = 0; index < count; ++index)
            line << (index ? "," : "") << this->physics[index]->getName()
                 << ':' << static_cast<unsigned long long>(
                        totals[5 + bins + index]);
        line << " bins=";
        bool first = true;
        for(std::size_t bin = 0; bin < bins; ++bin)
            if(totals[5 + bin] > 0)
            {
                line << (first ? "" : ",") << bin << ':'
                     << static_cast<unsigned long long>(totals[5 + bin]);
                first = false;
            }
        line << " fallback_cells="
             << static_cast<unsigned long long>(totals[4]);
        std::cout << line.str() << std::endl;
    }
    return bound;
}

void Simulation::adaptiveAfterStep(TimeIntegrationMode mode)
{
    AdaptiveModeState& a = this->adaptiveMode;
    if(!a.enabled)
        return;
    AdaptiveIntegrationRuntimeOptions const& o =
        adaptiveIntegrationRuntimeOptions();
    bool const individual = mode == TimeIntegrationMode::Individual;
    std::size_t const ramp = individual ?
        o.ramp_individual_events : o.ramp_global_steps;
    ++a.stepsInMode;
    a.wallTotalInMode += this->lastStepSecondsMax;
    if(a.stepsInMode > ramp)
    {
        a.wallMeasured += this->lastStepSecondsMax;
        a.simMeasured += this->lastStepAdvance;
    }
    if(!individual && (a.stepsInMode == 1 ||
                       a.stepsInMode % o.gate_interval_steps == 0 ||
                       a.gainRetry))
        a.gainBound = this->adaptiveGainBound();

    if(individual && a.switchToGlobalRequested)
    {
        if(this->IndividualStateSynchronized())
            this->adaptiveEnterGlobal(a.pendingReason);
        return;
    }
    if(a.stepsInMode < ramp + o.minimum_samples || !(a.wallMeasured > 0))
        return;
    // A driver target pending on the global path postpones every decision to
    // the first global step after it (SetAdaptiveDecisionsDeferred).
    if(!individual && this->adaptiveDecisionsDeferred)
        return;
    double const tau = a.simMeasured / a.wallMeasured;
    double& tau_here = individual ? a.tauIndividual : a.tauGlobal;
    std::uint64_t& tau_here_epoch =
        individual ? a.tauIndividualEpoch : a.tauGlobalEpoch;
    double const tau_other = individual ? a.tauGlobal : a.tauIndividual;
    std::uint64_t const tau_other_epoch =
        individual ? a.tauGlobalEpoch : a.tauIndividualEpoch;
    TimeIntegrationMode const other = individual ?
        TimeIntegrationMode::Global : TimeIntegrationMode::Individual;

    if(a.probing)
    {
        if(a.wallTotalInMode < a.probeWallBudget)
            return;
        // An individual probe decides only once its measured window spans the
        // interval of the coarsest occupied bin, i.e. every cell has stepped at
        // least once in it.  A shorter window measures a fraction of the cycle
        // (all-active events, or a run of closely spaced events) and can be
        // off by an order of magnitude either way (TDE job 10208589, t=21.13: a
        // 20-event probe, 7 of them measured, all in a burst of events 0.004 of
        // the global step apart, gave 0.04x global against 0.78x for the
        // previous probe).  Waiting stops at four times the wall budget, so a
        // probe of a mode that is truly slow still ends.  The scheduler state is
        // replicated, so is this branch (one MPI_MAX per event while waiting).
        if(individual && a.wallTotalInMode < 4 * a.probeWallBudget &&
           this->individualScheduler && this->individualScheduler->initialized())
        {
            int coarsest_bin = -1;
            for(CellTimeState const& state : this->individualScheduler->states())
                coarsest_bin = std::max(coarsest_bin, static_cast<int>(state.time_bin));
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &coarsest_bin, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
#endif
            double const coarsest_interval = coarsest_bin < 0 ? 0 :
                this->individualScheduler->timeQuantum() * static_cast<double>(
                    this->individualScheduler->binTicks(
                        static_cast<std::uint8_t>(coarsest_bin)));
            if(a.simMeasured < coarsest_interval)
            {
                // Logged on the first event that extends: the previous event
                // was not yet eligible to decide (before the ramp and sample
                // gate, or before the wall budget).
                bool const previous_eligible =
                    a.stepsInMode > ramp + o.minimum_samples &&
                    a.wallTotalInMode - this->lastStepSecondsMax >= a.probeWallBudget;
                if(this->rank == 0 && !previous_eligible)
                    std::cout << std::setprecision(12)
                              << "RICH_MODE_PROBE_EXTENDED cycle=" << this->tracker.getCycle()
                              << " time=" << this->tracker.getTime()
                              << " measured_sim=" << a.simMeasured
                              << " coarsest_bin=" << coarsest_bin
                              << " coarsest_interval=" << coarsest_interval
                              << " wall_s=" << a.wallTotalInMode
                              << " wall_budget_s=" << a.probeWallBudget
                              << " wall_cap_s=" << 4 * a.probeWallBudget << std::endl;
                return;
            }
        }
        tau_here = tau;
        tau_here_epoch = a.domainEpoch;
        a.probing = false;
        ++a.decisions;
        // The domain changed after the baseline was measured (box growth
        // can change the gravity cost ~10x): the comparison would be against
        // a different workload.  Return to the baseline mode, which then
        // re-measures before the next probe.
        if(tau_other > 0 && tau_other_epoch != a.domainEpoch)
        {
            this->adaptiveLogDecision(mode,
                (std::string("revert_to_") + IntegrationModeName(other) +
                 "_stale_baseline").c_str(), tau);
            this->adaptiveRequestSwitch(other,
                std::string("baseline of ") + IntegrationModeName(other) +
                " predates a domain change");
            return;
        }
        if(!(tau_other > 0) || tau >= tau_other * o.margin)
        {
            a.dwellMultiplier = 1;
            this->adaptiveLogDecision(mode,
                (std::string("adopt_") + IntegrationModeName(mode)).c_str(),
                tau);
            this->adaptiveResetWindow();
            return;
        }
        a.dwellMultiplier = std::min(2 * a.dwellMultiplier,
                                     o.dwell_backoff_cap);
        this->adaptiveLogDecision(mode,
            (std::string("revert_to_") + IntegrationModeName(other)).c_str(),
            tau);
        this->adaptiveRequestSwitch(other,
            std::string("probe of ") + IntegrationModeName(mode) +
            " was slower");
        return;
    }

    if(a.stepsInMode < ramp + o.dwell_min_steps * a.dwellMultiplier)
        return;
    tau_here = tau;
    tau_here_epoch = a.domainEpoch;
    ++a.decisions;
    if(!individual && std::isfinite(a.gainBound) &&
       a.gainBound < o.gain_minimum)
    {
        a.dwellMultiplier = std::min(2 * a.dwellMultiplier,
                                     o.dwell_backoff_cap);
        this->adaptiveLogDecision(mode, "stay_global_little_to_gain", tau);
        this->adaptiveResetWindow();
        return;
    }
    a.probing = true;
    a.probeWallBudget = o.probe_fraction * a.wallTotalInMode;
    this->adaptiveLogDecision(mode,
        (std::string("probe_") + IntegrationModeName(other)).c_str(), tau);
    this->adaptiveRequestSwitch(other,
        std::string("dwell complete, probing ") + IntegrationModeName(other));
}

double Simulation::GetTime(void) const
{
    return this->tracker.getTime();
}

size_t Simulation::GetCycle(void) const
{
    return this->tracker.getCycle();
}

void Simulation::SetCycle(size_t cycle)
{
    this->tracker.cycle = cycle;
    // ReadSimulation restores the cycle on an already constructed Simulation.
    // Remember that provenance so an opt-in pre-event rebalance can
    // distinguish restored committed state from a fresh start.
    this->initializedFromRestart = true;
}

void Simulation::SetTime(double t)
{
    this->tracker.time = t;
    if(this->individualScheduler)
        this->individualScheduler->resetTimeOrigin(t);
}

void Simulation::SetTimeStep(double dt)
{
    this->tsc->SetTimeStep(dt);
}

double Simulation::GetTimeStep(void) const
{
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual &&
       this->individualScheduler && this->individualScheduler->initialized())
        return this->individualScheduler->nextEventTimeStep();
    return this->tsc->GetTimeStep();
}

#ifdef RICH_MPI
    void Simulation::buildDataTransfer(void)
    {
        MPI_exchange_data(this->tess, this->extensives, false);
        MPI_exchange_data(this->tess, this->cells, false);
        const bool individual =
            this->timeIntegrationMode == TimeIntegrationMode::Individual &&
            this->individualScheduler && this->individualScheduler->initialized();
        if(individual)
            MPI_exchange_data(this->tess, this->individualScheduler->states(), false);
        else
        {
            ComputationalCell3D cdummy;
            MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
        }
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transfer();
        }
        if(individual)
        {
            // Individual state is canonical and owned-only between events.
            this->cells.resize(this->tess.GetPointNo());
            this->extensives.resize(this->tess.GetPointNo());
            this->individualScheduler->rebuildIndex(this->cells);
        }
    }

    void Simulation::buildDataTransfer(const ExchangeChain &chain)
    {
        if (chain.GetNorg() == 0)
            return;
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transferChain(chain);
        }
    }
#endif // RICH_MPI

double Simulation::GetWallclockTime(void) const
{
    return this->wallclockTime;
}

void Simulation::SetWallclockTime(double t)
{
    this->wallclockTime = t;
}

void Simulation::ReportRuntimeAMREvent(
    std::string const& mode, std::size_t const cycle, double const time,
    std::uint64_t const local_cells_before,
    std::uint64_t const local_added_cells,
    std::uint64_t const local_removed_cells,
    std::uint64_t const local_cells_after) const
{
    ValidateRuntimeLogConfiguration();
    std::array<std::uint64_t, 4> counts{{
        local_cells_before, local_added_cells, local_removed_cells,
        local_cells_after}};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, counts.data(),
                  static_cast<int>(counts.size()), MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if(counts[0] + counts[1] < counts[2] ||
       counts[3] != counts[0] + counts[1] - counts[2])
        throw std::logic_error("AMR runtime cell accounting is inconsistent");
    if(counts[1] == 0 && counts[2] == 0)
        return;

    RuntimeAMROutput output;
    output.mode = mode;
    output.cycle = cycle;
    output.time = time;
    output.cells_before = counts[0];
    output.added_cells = counts[1];
    output.removed_cells = counts[2];
    output.cells_after = counts[3];
    WriteRuntimeAMR(this->rank, RuntimeColorEnabled(), output);
}

void Simulation::step(void)
{
    ValidateRuntimeLogConfiguration();
    bool const detailed_runtime_log = RuntimeLogDetailed();
    bool const runtime_color_enabled = RuntimeColorEnabled();
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual)
    {
        this->stepIndividual();
        this->adaptiveAfterStep(TimeIntegrationMode::Individual);
        return;
    }

    MEMORY_DEBUG_PRINT("Simulation::step START cycle=" + std::to_string(this->tracker.getCycle()));
    this->lastPhysicsTimes.clear();
    this->lastLocalPhysicsTimes.clear();
    this->lastLocalPhysicsMpiTimes.clear();
    auto stepWallStart = std::chrono::high_resolution_clock::now();
    double const step_mpi_start = mpi_wait_profiler::Seconds();
    std::size_t const step_cycle = this->tracker.getCycle();
    double const step_start_time = this->tracker.getTime();
    std::uint64_t const local_step_active_cells =
        static_cast<std::uint64_t>(this->tess.GetPointNo());
    SourceStepTiming local_source_timing;
    MeshBuildTiming local_mesh_build_timing;
    RuntimeStepOutput retry_output;
    retry_output.mode = "global";
    retry_output.cycle = step_cycle;
    retry_output.active_cells = RuntimeSum(local_step_active_cells);
    retry_output.active_bins = "global";
    double next_time_step = std::numeric_limits<double>::max();
    // double dt = std::numeric_limits<double>::max();
    #ifdef RICH_MPI
        if(this->rank == 0)
    #endif // RICH_MPI
    if(detailed_runtime_log)
    {
        std::cout << "\nCycle " << this->tracker.getCycle() << " at time " << this->tracker.getTime() << std::endl;
    }

    for(std::shared_ptr<PhysicsStep> physics : this->physics)
    {
        std::string name = physics->getName();
        std::size_t retry_attempt = 0;
        int const runtime_rank = this->rank;
        physics->setStepRetryReporter(
            [retry_output, name, retry_attempt, runtime_rank,
             runtime_color_enabled](
                StepRetryRecord const& retry) mutable
            {
                ReportRuntimeRetry(runtime_rank, runtime_color_enabled,
                                   retry_output, name,
                                   ++retry_attempt, retry);
            });
        if(detailed_runtime_log && this->rank == 0)
            std::cout << "Running physics: " << name << std::endl;

        bool didRebalance = false;
        #ifdef RICH_MPI
            // if(this->tracker.getCycle() == this->lastRebalanceCycle + 2)
            // {
            //     physics->dumpCost(this->tracker.getCycle());
            // }

            std::string LB = physics->getRequiredLB();
            bool firstTime = false;

            if(this->currentLB != LB)
            {
                if(detailed_runtime_log && this->rank == 0) std::cout << "Changing load balance to " << LB << " (from " << this->currentLB << ")" << std::endl;
                auto it = this->loads.find(LB);
                if(it != this->loads.cend())
                {
                    if(detailed_runtime_log && this->rank == 0) std::cout << "Load balance restored" << std::endl;
                    this->setCurrentLoadBalance(LB,
                                                local_mesh_build_timing);
                }
                else
                {
                    if(detailed_runtime_log && this->rank == 0) std::cout << "Load balance generated for first time" << std::endl;
                    // std::vector<double> weights = physics->getLoadBalanceWeights();
                    // std::vector<Vector3D> points = this->tess.getMeshPoints();
                    // points.resize(this->tess.GetPointNo());
                    // this->tess.BuildParallel(points, weights, true);
                    firstTime = true;
                    // this->buildDataTransfer();
                }
            }

            bool forceRebalance = this->forceRebalanceSteps > 0 && this->tracker.getCycle() < this->forceRebalanceSteps;
            double rebalanceTime = 0;

            if(physics->allowRebalance() || forceRebalance)
            {
                if(detailed_runtime_log && this->rank == 0) std::cout << "allowRebalance=true, computing weights..." << std::endl;
                std::vector<double> weights = physics->getLoadBalanceWeights();
                if(detailed_runtime_log && this->rank == 0) std::cout << "Weights computed (" << weights.size() << "), checking ShouldRebalance..." << std::endl;
                bool shouldRebalance = this->tess.ShouldRebalance(weights);
                if(detailed_runtime_log && this->rank == 0)
                {
                    std::cout << "Should Rebalance: " << shouldRebalance << std::endl;
                }
                if(shouldRebalance)
                {
                    if(detailed_runtime_log && this->rank == 0) std::cout << "Doing rebalance on LB " << LB << std::endl;
                    auto rebalanceStart = std::chrono::high_resolution_clock::now();

                    didRebalance = true;
                    this->lastRebalanceCycle = this->tracker.getCycle();
                    physics->beforeLB();
                    {
                        MeshBuildTimer mesh_build_timer(
                            local_mesh_build_timing);
                        this->tess.Rebalance(weights);
                    }
                    if(detailed_runtime_log && this->rank == 0)
                    {
                        std::cout << "Did rebalanced" << std::endl;
                        // auto lb = this->tess.GetLoadBalancer();
                        // if (lb) lb->printInfo();
                    }                
                    this->buildDataTransfer();
                    physics->afterLB();

                    MPI_Barrier(MPI_COMM_WORLD);
                    rebalanceTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - rebalanceStart).count();
                    if(detailed_runtime_log && this->rank == 0) std::cout << "Rebalance time: " << rebalanceTime << "s" << std::endl;
                }
                else
                {
                    if(detailed_runtime_log && this->rank == 0) std::cout << LB << " is already rebalanced" << std::endl;
                }
            }

            std::shared_ptr<LoadBalancer<Vector3D>> load = this->tess.GetLoadBalancer();
            this->loads[LB] = load;
            this->currentLoad = load;
            this->currentLB = LB;
        #endif // RICH_MPI

        double dt = this->tsc->GetTimeStep();
        if(detailed_runtime_log && this->rank == 0)
            std::cout << "Running " << name << " with dt " << dt << std::endl;
        if(detailed_runtime_log)
            std::cout.flush();
        double dt_before = dt;

        MEMORY_DEBUG_PRINT("Before " + name);
        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI
        auto start = std::chrono::high_resolution_clock::now();
        double const physics_mpi_start = mpi_wait_profiler::Seconds();

        if(not didRebalance and this->tracker.getCycle() > 100)
        {
            vtune_start();
        }
        physics->step(dt);
        vtune_stop();

        double dt_actual = this->tsc->GetTimeStep();
        if(detailed_runtime_log && this->rank == 0 && dt_actual != dt_before)
            std::cout << "Hydro dt actually used: " << dt_actual << " (requested: " << dt_before << ")" << std::endl;

        double localTime = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastLocalPhysicsMpiTimes[name] =
            mpi_wait_profiler::Seconds() - physics_mpi_start;

        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI

        MEMORY_DEBUG_PRINT("After " + name);
        auto end = std::chrono::high_resolution_clock::now();
        double physicsTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        #ifdef RICH_MPI
        double totalPhysicsTime = physicsTime + rebalanceTime;
        if(detailed_runtime_log && this->rank == 0) std::cout << "Physics " << name << " time: " << totalPhysicsTime << " (step=" << physicsTime << "s, rebalance=" << rebalanceTime << "s)" << std::endl;
        this->lastPhysicsTimes[name] = totalPhysicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #else
        if(detailed_runtime_log && this->rank == 0) std::cout << "Physics " << name << " time: " << physicsTime << std::endl;
        this->lastPhysicsTimes[name] = physicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #endif

        SourceStepTiming const source_timing = physics->getSourceStepTiming();
        local_source_timing.first_seconds += source_timing.first_seconds;
        local_source_timing.second_seconds += source_timing.second_seconds;
        local_source_timing.calls += source_timing.calls;
        MeshBuildTiming const mesh_build_timing = physics->getMeshBuildTiming();
        local_mesh_build_timing.seconds += mesh_build_timing.seconds;
        local_mesh_build_timing.builds += mesh_build_timing.builds;
        double dt_suggest = physics->suggestTimeStep();
        next_time_step = std::min(next_time_step, dt_suggest);
        // if(this->rank == 0) std::cout << "Suggested " << next_time_step << ", dt_suggest " << dt_suggest << std::endl;
        
        #ifdef RICH_MPI
            this->buildDataTransfer(physics->GetExchangeChain());
            
            if(firstTime)
            {
                physics->beforeLB();
                std::vector<double> weights = physics->getLoadBalanceWeights();
                {
                    MeshBuildTimer mesh_build_timer(
                        local_mesh_build_timing);
                    this->tess.Rebalance(weights);
                }
                if(detailed_runtime_log && this->rank == 0)
                {
                    std::cout << "Rebalanced first time - load balance:" << std::endl;
                    auto lb = this->tess.GetLoadBalancer();
                    if (lb) lb->printInfo();
                }            
                this->buildDataTransfer();
                physics->afterLB();
            }
        #endif // RICH_MPI

        #ifdef RICH_MPI
            std::pair<Vector3D, Vector3D> newBox = this->tess.GetBoxCoordinates();
            if(newBox != this->currentBox)
            {
                this->currentBox = newBox;
                for(auto &entry : this->loads)
                {
                    entry.second->changeBox(this->currentBox);
                }
            }
        #endif // RICH_MPI

        if(detailed_runtime_log && this->rank == 0)
        {
            std::cout << name << " suggested " << dt_suggest << " for dt " << std::endl;
            std::cout << std::endl;
        }
    }
    
    double dt_used = this->tsc->GetTimeStep();
    if(detailed_runtime_log && this->rank == 0)
        std::cout << "Advancing time by dt=" << dt_used << ", next suggested dt=" << next_time_step << std::endl;
    this->tracker.updateTime(dt_used);
    this->tsc->SetTimeStep(next_time_step);
    // Anchor reference for a later individual phase: this step's CFL/source
    // suggestion, taken here while the mesh and the step's limits agree (the
    // post-step callback below may run AMR).  Collective; 0 when unavailable.
    if(IndividualBinAnchorMargin() > 0)
    {
        double suggested = this->tsc->SuggestTimeStep();
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &suggested, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
        this->adaptiveMode.anchorReferenceStep = std::isfinite(suggested) && suggested > 0 ? suggested : 0;
    }
    // The run's global post-step work follows the completed cycle, as in a
    // global driver's main loop, and is part of the timed step so that the
    // controller measures it as it measures the individual callbacks.  Before
    // the controller acts: a switch at the end of this step must not skip it.
    this->tracker.updateCycle();
    if(this->globalPostStep)
        this->globalPostStep();

    double const local_step_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - stepWallStart).count();
    this->wallclockTime += local_step_seconds;
#ifdef RICH_MPI
    this->reportPhaseBusyTimes("global", step_cycle, local_step_seconds,
        mpi_wait_profiler::Seconds() - step_mpi_start);
#endif

    RuntimeStepOutput output;
    output.mode = "global";
    output.cycle = step_cycle;
    output.start_time = step_start_time;
    output.end_time = this->tracker.getTime();
    output.event_dt = dt_used;
    output.applied_dt_min = dt_used;
    output.applied_dt_max = dt_used;
    output.next_event_dt = next_time_step;
    output.active_cells = retry_output.active_cells;
    output.total_cells = retry_output.active_cells;
    output.active_bins = retry_output.active_bins;
    output.source_calls = RuntimeMaximum(static_cast<std::uint64_t>(
        local_source_timing.calls));
    output.mesh_builds = RuntimeMaximum(static_cast<std::uint64_t>(
        local_mesh_build_timing.builds));
    output.mesh_seconds = RuntimeMaximum(local_mesh_build_timing.seconds);
    output.hydro_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "hydro"));
    output.gravity_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "gravity"));
    output.radiation_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "radiation"));
    output.amr_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "amr") + LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "topology-release"));
    output.source_seconds = RuntimeMaximum(
        local_source_timing.totalSeconds());
    output.source_first_seconds = RuntimeMaximum(
        local_source_timing.first_seconds);
    output.source_second_seconds = RuntimeMaximum(
        local_source_timing.second_seconds);
    output.step_seconds = RuntimeMaximum(local_step_seconds);
    WriteRuntimeStep(this->rank, detailed_runtime_log,
                     runtime_color_enabled, output);
    this->lastStepAdvance = dt_used;
    this->lastStepSecondsMax = output.step_seconds;
    this->adaptiveAfterStep(TimeIntegrationMode::Global);
}

#ifdef RICH_MPI
Simulation::IndividualRebalanceResult Simulation::rebalanceToPositionalOwnership(
    std::shared_ptr<PhysicsStep> const& balanceStep)
{
    std::shared_ptr<HilbertLoadBalancer<Vector3D>> const current =
        std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
            this->tess.GetLoadBalancer());
    std::vector<Vector3D> points = this->tess.getAllPoints();
    int local_valid = current && points.size() >= this->cells.size() ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &local_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_valid == 0)
        throw std::logic_error(
            "Positional Hilbert ownership needs a Hilbert balancer and canonical generators");
    points.resize(this->cells.size());
    double const planning_start = MPI_Wtime();
    std::shared_ptr<HilbertLoadBalancer<Vector3D>> proposal = current->clone();
    proposal->rebalance(points, std::vector<double>());
    double planning_seconds = MPI_Wtime() - planning_start;
    MPI_Allreduce(MPI_IN_PLACE, &planning_seconds, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    IndividualRebalanceResult result = this->rebalanceCommittedIndividualState(
        balanceStep, true, 1.0, std::vector<double>(this->cells.size(), 1.0),
        proposal);
    // The cut is part of the transition cost callers charge.
    result.maximumSeconds += planning_seconds;
    return result;
}

std::shared_ptr<PhysicsStep> Simulation::findIndividualBalanceStep(void) const
{
    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        if(physicsStep->allowRebalance())
            return physicsStep;
    return std::shared_ptr<PhysicsStep>();
}

Simulation::IndividualRebalanceResult
Simulation::rebalanceCommittedIndividualState(
    std::shared_ptr<PhysicsStep> const& balanceStep,
    bool const forceRebalance, double const threshold,
    std::vector<double> explicitWeights,
    std::shared_ptr<LoadBalancer<Vector3D>> exactLoadBalance)
{
    if(!balanceStep || !balanceStep->allowRebalance())
        throw std::logic_error(
            "Individual load balancing has no supporting physics step");
    if(!this->individualScheduler ||
       !this->individualScheduler->initialized())
        throw std::logic_error(
            "Individual load balancing requires initialized scheduler state");

    IndividualRebalanceResult result;
    result.currentRssBeforeKiB = currentResidentSetKiB();
    const auto balance_start = std::chrono::high_resolution_clock::now();
    this->lastIndividualBalanceCheckCycle = this->tracker.getCycle();

    int local_state_valid =
        this->extensives.size() == this->cells.size() &&
        this->individualScheduler->states().size() == this->cells.size() ?
        1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &local_state_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_state_valid == 0)
        throw std::logic_error(
            "Individual load balancing requires aligned committed state");

    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    result.totalOwnedCells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &result.totalOwnedCells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::vector<std::size_t> previous_owned_ids;
    previous_owned_ids.reserve(this->cells.size());
    for(ComputationalCell3D const& cell : this->cells)
        previous_owned_ids.push_back(cell.ID);
    std::sort(previous_owned_ids.begin(), previous_owned_ids.end());
    bool const use_exact_load_balance = exactLoadBalance != nullptr;
    // The build below is collective, so every rank must take the same branch:
    // a rank whose partial target covered all its owned cells has as many
    // mesh points as cells while the others do not.
    bool canonical_mesh =
        this->tess.GetPointNo() == this->cells.size();
    Tessellation3D::AllPointsMap const& build_map =
        this->tess.GetIndicesInAllPoints();
    for(std::size_t i = 0; canonical_mesh && i < this->cells.size(); ++i)
    {
        auto const entry = build_map.find(i);
        canonical_mesh = entry != build_map.end() && entry->second == i;
    }
    int canonical_everywhere = canonical_mesh ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &canonical_everywhere, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(!use_exact_load_balance || canonical_everywhere == 0)
    {
        std::vector<Vector3D> points = this->tess.getAllPoints();
        int local_points_valid = points.size() >= this->cells.size() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_points_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(local_points_valid == 0)
            throw std::logic_error(
                "Individual load balancing is missing canonical generator positions");
        points.resize(this->cells.size());
        {
            MeshBuildTimer mesh_build_timer(
                this->individualMeshBuildTiming);
            ++this->individualMeshBuildTiming.full_builds;
            this->tess.BuildParallel(points, true /* no rebalance */,
                                     true /* no exchange */);
        }
        int local_mesh_valid =
            this->tess.GetPointNo() == this->cells.size() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_mesh_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(local_mesh_valid == 0)
            throw std::logic_error(
                "Individual load balancing could not restore the canonical mesh");
    }

    balanceStep->beforeLB();
    std::vector<double> weights = use_exact_load_balance ?
        std::move(explicitWeights) : balanceStep->getLoadBalanceWeights();
    int local_weights_valid = weights.size() == this->cells.size() ?
        1 : 0;
    double local_weight = 0;
    if(local_weights_valid != 0)
        for(double const weight : weights)
        {
            if(!std::isfinite(weight) || weight < 0 ||
               (!use_exact_load_balance && weight == 0))
            {
                local_weights_valid = 0;
                break;
            }
            local_weight += weight;
        }
    MPI_Allreduce(MPI_IN_PLACE, &local_weights_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_weights_valid == 0)
    {
        balanceStep->afterLB();
        throw std::logic_error(
            "Individual load-balance weights must align with owned cells "
            "and be finite and nonnegative");
    }

    double total_weight = local_weight;
    double maximum_weight = local_weight;
    MPI_Allreduce(MPI_IN_PLACE, &total_weight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_weight, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    if(!(total_weight > 0))
    {
        balanceStep->afterLB();
        throw std::logic_error(
            "Individual load balancing requires positive global work");
    }
    double const mean_weight = total_weight / static_cast<double>(this->size);
    result.weightSkew = mean_weight > 0 ? maximum_weight / mean_weight : 1.0;
    int local_should_rebalance =
        (forceRebalance ||
         (result.weightSkew > threshold &&
          this->tess.ShouldRebalance(weights))) ? 1 : 0;
    int minimum_should_rebalance = local_should_rebalance;
    int maximum_should_rebalance = local_should_rebalance;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_should_rebalance, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_should_rebalance, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(minimum_should_rebalance != maximum_should_rebalance)
    {
        balanceStep->afterLB();
        throw std::logic_error(
            "Individual load-balance decision differs across MPI ranks");
    }
    result.applied = maximum_should_rebalance != 0;
    bool owned_count_preserved = true;

    if(result.applied)
    {
        // Release every package's old ownership-sized scratch before the
        // tessellation and registered state buffers migrate.  Calling only the
        // selected balanceStep's beforeLB() is insufficient: radiation is
        // normally not the package that supplies the hydro balance weights.
        for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
            physicsStep->beforeIndividualRebalance();
        {
            MeshBuildTimer mesh_build_timer(this->individualMeshBuildTiming);
            if(use_exact_load_balance)
                this->tess.SetLoadBalancer(std::move(exactLoadBalance));
            else
                this->tess.Rebalance(weights);
        }
        std::vector<double>().swap(weights);
        // This migrates primitives, extensives, scheduler state, and every
        // registered migration buffer by stable cell ID, then rebuilds the
        // scheduler index against the new ownership.
        this->buildDataTransfer();
        for(ComputationalCell3D const& cell : this->cells)
            if(!std::binary_search(previous_owned_ids.begin(),
                                   previous_owned_ids.end(), cell.ID))
                ++result.migratedCells;
        MPI_Allreduce(MPI_IN_PLACE, &result.migratedCells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        unsigned long long total_owned_after =
            static_cast<unsigned long long>(this->cells.size());
        MPI_Allreduce(MPI_IN_PLACE, &total_owned_after, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        owned_count_preserved = total_owned_after == result.totalOwnedCells;
        if(owned_count_preserved)
        {
            this->lastRebalanceCycle = this->tracker.getCycle();
            ++this->individualOwnershipEpoch;
            const std::string load_name = balanceStep->getRequiredLB();
            this->currentLoad = this->tess.GetLoadBalancer();
            this->loads[load_name] = this->currentLoad;
            this->currentLB = load_name;
        }
    }
	std::vector<double>().swap(weights);
	std::vector<std::size_t>().swap(previous_owned_ids);

    // The selected physics step owns topology-dependent mesh caches.
    balanceStep->afterLB();
    if(!owned_count_preserved)
        throw std::logic_error(
            "Individual load balancing changed global owned-cell count");
    MPI_Barrier(MPI_COMM_WORLD);
    result.localSeconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - balance_start).count();
    result.maximumSeconds = result.localSeconds;
    MPI_Allreduce(MPI_IN_PLACE, &result.maximumSeconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    if(result.applied)
        this->lastIndividualRebalanceSeconds = result.maximumSeconds;
    result.currentRssAfterKiB = currentResidentSetKiB();
    return result;
}

bool Simulation::individualActiveHilbertBalanceEnabled(void) const
{
    IndividualActiveHilbertRuntimeOptions const& options =
        individualActiveHilbertRuntimeOptions();
    if(!options.enabled)
        return false;

    bool const has_balance_step =
        static_cast<bool>(this->findIndividualBalanceStep());
    bool const has_hilbert_load_balance = static_cast<bool>(
        std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
            this->tess.GetLoadBalancer()));
    unsigned int const local_support =
        (has_balance_step ? 1u : 0u) |
        (has_hilbert_load_balance ? 2u : 0u);
    unsigned int support_mask = 1u << local_support;
    MPI_Allreduce(MPI_IN_PLACE, &support_mask, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((support_mask & (support_mask - 1u)) != 0u)
        throw std::logic_error(
            "Active Hilbert balance compatibility differs across MPI ranks");

    if(has_balance_step && has_hilbert_load_balance)
        return true;
    if(!options.explicitly_configured)
        return false;
    if(!has_balance_step)
        throw std::logic_error(
            "Active Hilbert balancing has no supporting physics step");
    throw std::logic_error(
        "Active Hilbert balancing requires a Hilbert load balancer");
}

void Simulation::maybeRebalanceBeforeFirstIndividualEvent(
    bool const activeHilbertBalance)
{
    if(this->preFirstIndividualRebalanceChecked)
        return;
    this->preFirstIndividualRebalanceChecked = true;

    if(activeHilbertBalance)
        return;

    IndividualRebalanceRuntimeOptions const& options =
        individualRebalanceRuntimeOptions();
    auto const trace_decision = [&](char const* const reason,
                                    bool const supported,
                                    double const owned_cell_skew,
                                    bool const requested)
    {
        if(options.trace && this->rank == 0)
            std::clog << std::setprecision(17)
                      << "INDIVIDUAL_PRE_FIRST_EVENT_REBALANCE_DECISION"
                      << " cycle=" << this->tracker.getCycle()
                      << " restart="
                      << (this->initializedFromRestart ? 1 : 0)
                      << " enabled="
                      << (options.before_first_event ? 1 : 0)
                      << " auto_rebalance=" << (options.enabled ? 1 : 0)
                      << " supported=" << (supported ? 1 : 0)
                      << " owned_max_mean=" << owned_cell_skew
                      << " threshold=" << options.threshold
                      << " requested=" << (requested ? 1 : 0)
                      << " reason=" << reason
                      << " ownership_epoch="
                      << this->individualOwnershipEpoch
                      << std::endl;
    };

    if(!options.before_first_event)
    {
        trace_decision("disabled", false, 1.0, false);
        return;
    }

    int const local_state = (this->initializedFromRestart ? 1 : 0) |
        (this->individualScheduler &&
         this->individualScheduler->initialized() ? 2 : 0);
    unsigned int state_mask =
        1u << static_cast<unsigned int>(local_state);
    MPI_Allreduce(MPI_IN_PLACE, &state_mask, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((state_mask & (state_mask - 1u)) != 0u)
        throw std::logic_error(
            "Pre-first-event rebalance state differs across MPI ranks");
    if(!this->initializedFromRestart)
    {
        trace_decision("fresh_start", false, 1.0, false);
        return;
    }
    if(!this->individualScheduler ||
       !this->individualScheduler->initialized())
    {
        trace_decision("scheduler_uninitialized", false, 1.0, false);
        return;
    }
    if(this->individualEventInProgress)
        throw std::logic_error(
            "Pre-first-event rebalance cannot run during an event");
    if(!options.enabled)
    {
        trace_decision("auto_rebalance_disabled", false, 1.0, false);
        return;
    }

    std::shared_ptr<PhysicsStep> const balance_step =
        this->findIndividualBalanceStep();
    unsigned int support_mask =
        1u << static_cast<unsigned int>(balance_step ? 1 : 0);
    MPI_Allreduce(MPI_IN_PLACE, &support_mask, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((support_mask & (support_mask - 1u)) != 0u)
        throw std::logic_error(
            "Pre-first-event rebalance support differs across MPI ranks");
    if(!balance_step)
    {
        trace_decision("unsupported", false, 1.0, false);
        return;
    }

    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    unsigned long long total_owned_cells = local_owned_cells;
    unsigned long long maximum_owned_cells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &total_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    double const mean_owned_cells = static_cast<double>(total_owned_cells) /
        static_cast<double>(this->size);
    double const owned_cell_skew = mean_owned_cells > 0 ?
        static_cast<double>(maximum_owned_cells) / mean_owned_cells : 1.0;
    bool const requested = owned_cell_skew > options.threshold;
    trace_decision(requested ? "requested" : "balanced", true,
                   owned_cell_skew, requested);
    if(!requested)
        return;

    IndividualRebalanceResult const result =
        this->rebalanceCommittedIndividualState(
            balance_step, false, options.threshold);
    this->lastPhysicsTimes["individual-pre-first-event-load-balance"] =
        result.maximumSeconds;
    this->lastLocalPhysicsTimes["individual-pre-first-event-load-balance"] =
        result.localSeconds;
    if(this->rank == 0)
        std::clog << std::setprecision(17)
                  << "INDIVIDUAL_PRE_FIRST_EVENT_REBALANCE_RESULT"
                  << " cycle=" << this->tracker.getCycle()
                  << " applied=" << (result.applied ? 1 : 0)
                  << " seconds_max=" << result.maximumSeconds
                  << " weight_max_mean=" << result.weightSkew
                  << " migrated_cells=" << result.migratedCells
                  << " total_owned_cells=" << result.totalOwnedCells
                  << " ownership_epoch=" << this->individualOwnershipEpoch
                  << " current_rss_before_kib="
                  << result.currentRssBeforeKiB
                  << " current_rss_after_kib="
                  << result.currentRssAfterKiB
                  << " peak_rss_kib=" << peakResidentSetKiB()
                  << std::endl;
}

// RICH_MPI_WAIT_PROFILE=1: for each timed phase of this step (the physics steps, AMR, the
// active-Hilbert decision, the load balance, the unattributed rest, and the
// whole step), the rank distribution of busy time (wall minus time inside MPI
// calls), plus a HEURISTIC of what the phase would cost with its busy work
// spread evenly: heuristic_balanced = mean busy + the smallest per-rank MPI
// time, heuristic_imbalance = 1 - heuristic_balanced / wall max.  Not a bound:
// it misses imbalance when the slowest rank changes between collectives
// inside a phase (two ranks alternating 10 s of work between barriers show
// 0), and busy time that no ownership change moves counts as balanced work.
// Enabled by RICH_MPI_WAIT_PROFILE=1 alone (agreed across ranks on the first
// call, which every rank makes on its first step).  One rank-0 record
// <MODE>_PHASE_BUSY per phase.  Collective.
void Simulation::reportPhaseBusyTimes(std::string const& mode,
    std::size_t const cycle, double const step_seconds,
    double const step_mpi_seconds)
{
    static int profile_agreed = -1;
    if(profile_agreed < 0)
    {
        int minimum = mpi_wait_profiler::Enabled() ? 1 : 0;
        int maximum = minimum;
        MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(minimum != maximum)
            throw std::invalid_argument(
                "RICH_MPI_WAIT_PROFILE differs across MPI ranks");
        profile_agreed = maximum;
    }
    if(profile_agreed == 0)
        return;

    // A fixed layout on every rank: a phase that did not run reports zeros.
    std::vector<std::string> phases;
    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        phases.push_back(physicsStep->getName());
    if(mode == "individual")
        for(char const* const phase : {"individual-amr",
                                       "individual-active-hilbert-balance",
                                       "individual-load-balance"})
            phases.push_back(phase);
    std::size_t const timed = phases.size();
    phases.push_back("unattributed");
    phases.push_back("step");
    std::vector<double> local(2 * phases.size(), 0);
    double attributed_wall = 0;
    double attributed_mpi = 0;
    for(std::size_t k = 0; k < timed; ++k)
    {
        auto const wall = this->lastLocalPhysicsTimes.find(phases[k]);
        auto const mpi = this->lastLocalPhysicsMpiTimes.find(phases[k]);
        if(wall == this->lastLocalPhysicsTimes.end() ||
           mpi == this->lastLocalPhysicsMpiTimes.end())
            continue;
        local[2 * k] = wall->second;
        local[2 * k + 1] = mpi->second;
        attributed_wall += wall->second;
        attributed_mpi += mpi->second;
    }
    local[2 * timed] = step_seconds - attributed_wall;
    local[2 * timed + 1] = step_mpi_seconds - attributed_mpi;
    local[2 * timed + 2] = step_seconds;
    local[2 * timed + 3] = step_mpi_seconds;

    std::vector<double> gathered(this->rank == 0 ?
        local.size() * static_cast<std::size_t>(this->size) : 0);
    MPI_Gather(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
               this->rank == 0 ? gathered.data() : nullptr,
               static_cast<int>(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(this->rank != 0)
        return;
    std::string label = mode;
    for(char& c : label)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    for(std::size_t k = 0; k < phases.size(); ++k)
    {
        double wall_max = 0, busy_max = 0, busy_sum = 0;
        double busy_min = std::numeric_limits<double>::infinity();
        double mpi_min = std::numeric_limits<double>::infinity(), mpi_max = 0;
        int busy_max_rank = 0;
        for(int r = 0; r < this->size; ++r)
        {
            double const wall = gathered[static_cast<std::size_t>(r) * local.size() + 2 * k];
            double const mpi = gathered[static_cast<std::size_t>(r) * local.size() + 2 * k + 1];
            double const busy = wall - mpi;
            wall_max = std::max(wall_max, wall);
            if(busy > busy_max || r == 0)
            {
                busy_max = busy;
                busy_max_rank = r;
            }
            busy_min = std::min(busy_min, busy);
            busy_sum += busy;
            mpi_min = std::min(mpi_min, mpi);
            mpi_max = std::max(mpi_max, mpi);
        }
        double const busy_mean = busy_sum / this->size;
        double const balanced = busy_mean + mpi_min;
        std::cout << std::setprecision(6) << label << "_PHASE_BUSY cycle=" << cycle
                  << " phase=" << phases[k] << " wall_max=" << wall_max
                  << " busy_max=" << busy_max << " busy_max_rank=" << busy_max_rank
                  << " busy_mean=" << busy_mean << " busy_min=" << busy_min
                  << " mpi_min=" << mpi_min << " mpi_max=" << mpi_max
                  << " heuristic_balanced=" << balanced
                  << " heuristic_imbalance=" << (wall_max > 0 ?
                      std::max(0.0, 1.0 - balanced / wall_max) : 0.0)
                  << std::endl;
    }
}

void Simulation::maybeBalanceIndividualEventByActiveBins(
    IndividualStepContext& context, std::uint64_t const eventTick)
{
    IndividualActiveHilbertRuntimeOptions const& options =
        individualActiveHilbertRuntimeOptions();

    auto const decision_start = std::chrono::high_resolution_clock::now();
    double const decision_mpi_start = mpi_wait_profiler::Seconds();
    std::shared_ptr<PhysicsStep> const balance_step =
        this->findIndividualBalanceStep();
    std::shared_ptr<HilbertLoadBalancer<Vector3D>> current_load_balance =
        std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
            this->tess.GetLoadBalancer());
    if(!balance_step || !current_load_balance)
        throw std::logic_error(
            "Active Hilbert balance compatibility changed during an event");

    unsigned int local_input_errors = 0;
    if(context.active_mask.size() != this->cells.size())
        local_input_errors |= 1u << 0;
    if(this->individualScheduler->states().size() != this->cells.size())
        local_input_errors |= 1u << 1;
    for(std::size_t const active : context.active_indices)
    {
        if(active >= this->cells.size())
        {
            local_input_errors |= 1u << 2;
            continue;
        }
        if(!context.isActive(active))
            local_input_errors |= 1u << 3;
        if(this->individualScheduler->states()[active].time_bin > 62)
            local_input_errors |= 1u << 4;
    }
    unsigned int global_input_errors = local_input_errors;
    MPI_Allreduce(MPI_IN_PLACE, &global_input_errors, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if(global_input_errors != 0)
    {
        int first_bad_rank = local_input_errors == 0 ? this->size : this->rank;
        MPI_Allreduce(MPI_IN_PLACE, &first_bad_rank, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        throw std::logic_error(
            "Active Hilbert balancing requires aligned Hilbert scheduler state"
            " (error mask " + std::to_string(global_input_errors) +
            ", first rank " + std::to_string(first_bad_rank) + ")");
    }

    std::uint64_t active_bin_mask = IndividualActiveTimeBinMask(
        context, this->individualScheduler->states());
    MPI_Allreduce(MPI_IN_PLACE, &active_bin_mask, 1, MPI_UINT64_T,
                  MPI_BOR, MPI_COMM_WORLD);
    if(active_bin_mask == 0)
        throw std::logic_error(
            "Active Hilbert balancing found an event without active cells");

    IndividualHilbertPartitionLoad const current_load =
        measureIndividualCurrentLoad(this->cells.size(),
                                     context.active_indices.size(),
                                     this->size);
    if(current_load.active_cells == 0 || current_load.owned_cells == 0)
        throw std::logic_error(
            "Active Hilbert balancing found empty global scheduler state");

    auto cached = this->individualActiveHilbertBoundaries.find(
        active_bin_mask);
    int local_cache_hit = cached !=
        this->individualActiveHilbertBoundaries.end() ? 1 : 0;
    int minimum_cache_hit = local_cache_hit;
    int maximum_cache_hit = local_cache_hit;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_cache_hit, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_cache_hit, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if(minimum_cache_hit != maximum_cache_hit)
        throw std::logic_error(
            "Active Hilbert boundary cache differs across MPI ranks");
    bool const cache_hit = maximum_cache_hit != 0;

    std::string action;
    IndividualHilbertPartitionLoad proposed_load = current_load;
    std::shared_ptr<HilbertLoadBalancer<Vector3D>> proposed_load_balance;
    std::vector<double> active_weights;
    IndividualActiveCutBound cut_bound;
    double cut_bound_max_mean = 0;
    // Segmented ownership decided to return to a weighted positional cut.
    bool revert_to_positional = false;
    // A segmented safety repair migrates even to an unchanged assignment.
    bool force_segment_migration = false;
    // What proposed_load measures: "current" (balanced), "cache", "rebuild",
    // or "none" when the rebuild was skipped (no proposal was measured).
    char const* proposal_source = "current";

    bool const current_active_balanced =
        current_load.activeBalanced(options.active_threshold);
    bool const current_owned_safe = current_load.ownedMaxMean() <=
        options.maximum_owned_cell_skew;
    // The boundary cache holds positional (one range per rank) cuts only; a
    // segmented partition is never stored or compared as bare boundaries.
    bool const current_positional = current_load_balance->positionalOwnership();
    // Segmented ownership decides (and accounts) on every event.
    if(options.segments_per_rank <= 1 && current_active_balanced &&
       current_owned_safe)
    {
        if(current_positional)
            this->individualActiveHilbertBoundaries[active_bin_mask] =
                current_load_balance->getBoundaries();
        action = cache_hit ? "refresh-balanced" : "store-balanced";
    }
    else
    {
        std::vector<Vector3D> points = this->tess.getAllPoints();
        int local_points_valid = points.size() >= this->cells.size() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_points_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(local_points_valid == 0)
            throw std::logic_error(
                "Active Hilbert balancing is missing owned generator positions");
        points.resize(this->cells.size());
        active_weights.assign(this->cells.size(), 0);
        for(std::size_t const active : context.active_indices)
            active_weights[active] = 1;

        // The current cut's segments at the current positions bound every
        // active-only rebuild (individualActiveCutOwnedLowerBound) and measure a
        // cached cut that equals the current one.
        std::vector<unsigned long long> const current_segments =
            countIndividualHilbertSegments(*current_load_balance, points,
                                           context.active_mask, this->size);
        IndividualHilbertPartitionLoad const current_position_load =
            individualHilbertLoadFromSegments(current_segments, this->size);
        // The passive-run bound needs the current cut to be one curve range
        // per rank, in rank order.
        if(current_positional)
            cut_bound = individualActiveCutOwnedLowerBound(current_segments,
                                                           this->size);
        cut_bound_max_mean = current_position_load.owned_cells > 0 ?
            static_cast<double>(cut_bound.cells) /
                (static_cast<double>(current_position_load.owned_cells) /
                 this->size) : 0;
        bool const rebuild_exceeds_owned_skew =
            cut_bound_max_mean > options.maximum_owned_cell_skew;

        bool rebuild_boundaries = !cache_hit;
        if(options.segments_per_rank > 1)
        {
            // Segmented ownership (plan stages A2/A3) replaces the active-only
            // single-range cut, which provably cannot spread clustered active
            // cells (the passive-run bound), and its cache.  Every quantity
            // below is reduced, so every rank takes the same branch.
            //
            // Adoption is a measured probe, as in the adaptive mode
            // controller: a partition is adopted once a positional reference
            // exists and the current partition concentrates a time bin; it is
            // then judged by its event-mesh time against positional events of
            // the same active-fraction class, charged with its planning and
            // migration.  A loss reverts to positional, with a cooldown that
            // doubles on every revert.
            rebuild_boundaries = false;
            IndividualRebalanceRuntimeOptions const& balance_options =
                individualRebalanceRuntimeOptions();
            IndividualSegmentDecisionState& decision =
                this->individualSegmentDecision;
            ++decision.event_counter;
            // The previous event's ledger seconds (max over ranks); named
            // "mesh" for the debt baseline it also feeds.
            double last_event_mesh = this->lastIndividualEventLedgerLocalSeconds;
            MPI_Allreduce(MPI_IN_PLACE, &last_event_mesh, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);
            int const measured_class = decision.pending_class;
            IndividualBinOccupancy const occupancy =
                measureIndividualBinOccupancy(this->individualScheduler->states());
            double const current_bin_skew =
                occupancy.skew(this->size, options.active_threshold);
            std::uint64_t const cooldown = static_cast<std::uint64_t>(
                std::max<std::size_t>(balance_options.cooldown_events, 1));
            // Events above half active are full builds whatever the
            // partition (the closure threshold): they stay out of the ledger.
            int const ledger_classes = IndividualSegmentDecisionState::classes - 1;
            std::uint64_t positional_samples = 0;
            for(int c = 0; c < ledger_classes; ++c)
                positional_samples += decision.positional_count[static_cast<std::size_t>(c)];
            bool const measured = measured_class >= 0 &&
                measured_class < ledger_classes && last_event_mesh > 0;
            // Reference for an event class: its own positional mean, with at
            // least two samples.  Classes differ in kind (below 0.1 % active
            // the positional builds are partial, above it mostly full), so no
            // other class stands in: an event of a class without a reference
            // is not credited, and a probe without enough credited events is
            // kept rather than reverted (the concentration that adopted it is
            // the evidence).
            int reference_source = -1;
            auto positional_reference = [&](int c)
            {
                std::size_t const k = static_cast<std::size_t>(c);
                if(decision.positional_count[k] >= 2)
                {
                    reference_source = c;
                    return decision.positional_sum[k] /
                        static_cast<double>(decision.positional_count[k]);
                }
                reference_source = -1;
                return -1.0;
            };
            bool replan = false;
            char const* replan_reason = "";
            if(current_positional)
            {
                if(measured)
                {
                    decision.positional_sum[static_cast<std::size_t>(measured_class)] +=
                        last_event_mesh;
                    ++decision.positional_count[static_cast<std::size_t>(measured_class)];
                    ++positional_samples;
                }
                replan = decision.event_counter >= decision.cooldown_until_event &&
                    positional_samples >= cooldown &&
                    current_bin_skew > options.active_threshold;
                replan_reason = "positional-bin-skew";
            }
            else
            {
                ++decision.probe_events;
                ++decision.events_since_migration;
                if(measured)
                {
                    ++decision.measured_events;
                    decision.measured_seconds += last_event_mesh;
                    decision.class_seconds[static_cast<std::size_t>(measured_class)] +=
                        last_event_mesh;
                    decision.window_measured_seconds += last_event_mesh;
                    double const reference = positional_reference(measured_class);
                    if(reference >= 0)
                    {
                        double const saving = reference - last_event_mesh;
                        decision.net_benefit += saving;
                        decision.window_saving += saving;
                        ++decision.credited_events;
                        ++decision.window_credited;
                        decision.credited_seconds += last_event_mesh;
                        decision.window_credited_seconds += last_event_mesh;
                    }
                    // The first event after a migration pays for it: it is
                    // in the ledger above but not in the baseline.
                    if(decision.events_since_migration >= 2)
                    {
                        if(decision.baseline_pending)
                        {
                            decision.baseline_mesh = last_event_mesh;
                            decision.baseline_pending = false;
                        }
                        else
                            decision.debt += std::max(0.0,
                                last_event_mesh - decision.baseline_mesh);
                    }
                }
                ++decision.window_events;
                bool revert = false;
                if(decision.window_events >= cooldown)
                {
                    // A window counts as losing only when at least half of
                    // its own events were credited.
                    bool const window_representative =
                        2 * decision.window_credited >= decision.window_events &&
                        2 * decision.window_credited_seconds >=
                            decision.window_measured_seconds;
                    decision.negative_windows =
                        decision.window_saving < 0 && window_representative ?
                        decision.negative_windows + 1 : 0;
                    decision.window_saving = 0;
                    decision.window_events = 0;
                    decision.window_credited = 0;
                    decision.window_measured_seconds = 0;
                    decision.window_credited_seconds = 0;
                    // Revert (a) after two consecutive losing windows, (b) when
                    // three windows have not repaid the period's costs.
                    // Only on representative evidence: enough credited
                    // events, covering at least half of the measured ones.
                    // Every class holding a quarter or more of the probe's
                    // measured seconds must have its own reference.
                    bool dominant_classes_referenced = true;
                    for(int c = 0; c < ledger_classes; ++c)
                        if(4 * decision.class_seconds[static_cast<std::size_t>(c)] >=
                               decision.measured_seconds &&
                           decision.class_seconds[static_cast<std::size_t>(c)] > 0 &&
                           decision.positional_count[static_cast<std::size_t>(c)] < 2)
                            dominant_classes_referenced = false;
                    revert = decision.credited_events >= cooldown &&
                        dominant_classes_referenced &&
                        2 * decision.credited_events >= decision.measured_events &&
                        2 * decision.credited_seconds >= decision.measured_seconds &&
                        (decision.negative_windows >= 2 ||
                         (decision.probe_events >= 3 * cooldown &&
                          decision.net_benefit < 0));
                }
                if(revert)
                {
                    revert_to_positional = true;
                    proposal_source = "positional";
                    action = "revert-segmented";
                    decision.cooldown_until_event = decision.event_counter +
                        (3 * cooldown << std::min(decision.reverts, 10));
                    ++decision.reverts;
                }
                else if(!current_owned_safe || !current_load.ownsEveryRank())
                {
                    // Safety: the owned-cell skew drifted past its limit.  An
                    // accepted repair migrates even if its assignment equals
                    // the current one (cells moved out of their segments).
                    replan = true;
                    replan_reason = "segmented-owned-skew";
                    force_segment_migration = true;
                }
                else if(options.segment_bins && !decision.baseline_pending &&
                        decision.event_counter >= decision.cooldown_until_event &&
                        decision.debt > balance_options.amortization_factor *
                            decision.last_migration_seconds &&
                        current_bin_skew > options.active_threshold)
                {
                    replan = true;
                    replan_reason = "segmented-debt";
                }
                else
                {
                    proposal_source = "none";
                    action = "keep-segmented";
                }
            }
            IndividualSegmentPlanStats plan_stats;
            if(replan)
            {
                double const planning_start = MPI_Wtime();
                ++this->individualActiveHilbertBuiltRebuilds;
                proposed_load_balance = options.segment_bins ?
                    buildIndividualBinBalancedSegments(*current_load_balance,
                        points, this->individualScheduler->states(), occupancy,
                        options.segments_per_rank, this->size,
                        options.active_threshold, plan_stats) :
                    current_load_balance->clone();
                if(!options.segment_bins)
                {
                    std::vector<double> piece_weights;
                    if(options.segment_work > 0)
                    {
                        // Work weights from the replicated bin occupancy.
                        int coarsest = 0;
                        long double total_work = 0;
                        unsigned long long total_cells = 0;
                        for(int bin = 0; bin < 64; ++bin)
                            if(occupancy.total[static_cast<std::size_t>(bin)] > 0)
                                coarsest = bin;
                        for(int bin = 0; bin < 64; ++bin)
                        {
                            total_cells += occupancy.total[static_cast<std::size_t>(bin)];
                            total_work += std::ldexp(static_cast<long double>(
                                occupancy.total[static_cast<std::size_t>(bin)]),
                                coarsest - bin);
                        }
                        double const scale = total_work > 0 ?
                            static_cast<double>(static_cast<long double>(total_cells) /
                                total_work) : 0;
                        auto const& states = this->individualScheduler->states();
                        piece_weights.resize(points.size());
                        for(std::size_t i = 0; i < points.size(); ++i)
                            piece_weights[i] = 1 + options.segment_work * scale *
                                std::ldexp(1.0, coarsest - static_cast<int>(
                                    std::min<std::size_t>(states[i].time_bin, 63)));
                    }
                    proposed_load_balance->rebalanceInterleaved(points,
                        piece_weights, options.segments_per_rank);
                }
                proposed_load = measureIndividualHilbertPartition(
                    *proposed_load_balance, points, context.active_mask,
                    this->size);
                proposal_source = "segmented";
                // Bin plans must also meet the per-bin cap (slack 2 on the
                // active threshold, plan section 1.2) and improve the skew.
                bool const bins_ok = !options.segment_bins ||
                    (plan_stats.planned_bin_skew <= 2 * options.active_threshold &&
                     plan_stats.planned_bin_skew < current_bin_skew);
                if(!proposed_load.ownsEveryRank())
                {
                    proposed_load_balance.reset();
                    action = "reject-segmented-empty-rank";
                }
                else if(proposed_load.ownedMaxMean() >
                        options.maximum_owned_cell_skew)
                {
                    proposed_load_balance.reset();
                    action = "reject-segmented-owned-skew";
                }
                else if(!bins_ok)
                {
                    proposed_load_balance.reset();
                    action = "reject-segmented-bins";
                }
                else
                {
                    action = "segmented";
                    // An accepted re-plan within segmented ownership migrates
                    // even when it reproduces the current map: cells drifted
                    // across segment boundaries are what it repairs.
                    if(!current_positional)
                        force_segment_migration = true;
                }
                double planning_seconds = MPI_Wtime() - planning_start;
                MPI_Allreduce(MPI_IN_PLACE, &planning_seconds, 1, MPI_DOUBLE,
                              MPI_MAX, MPI_COMM_WORLD);
                this->individualSegmentMigrationSeconds += planning_seconds;
                if(proposed_load_balance)
                    decision.pending_planning_seconds = planning_seconds;
                else
                {
                    // A rejected plan's cost is lost; wait before the next.
                    if(!current_positional)
                        decision.net_benefit -= planning_seconds;
                    decision.cooldown_until_event = decision.event_counter + cooldown;
                    // A failed repair of an unsafe segmented partition returns
                    // to positional ownership (validated by the revert path).
                    if(!current_positional &&
                       (!current_owned_safe || !current_load.ownsEveryRank()))
                    {
                        revert_to_positional = true;
                        proposal_source = "positional";
                        action = "revert-segmented-unsafe";
                    }
                }
            }
            if(options.trace && this->rank == 0)
                std::clog << std::setprecision(6)
                          << "RICH_ACTIVE_HILBERT_SEGMENT_DECISION cycle="
                          << this->tracker.getCycle() << " event=" << decision.event_counter
                          << " action=" << (action.empty() ? "none" : action.c_str())
                          << " reason=" << (replan ? replan_reason : "")
                          << " positional=" << (current_positional ? 1 : 0)
                          << " bin_skew=" << current_bin_skew
                          << " planned_bin_skew=" << plan_stats.planned_bin_skew
                          << " planned_owned_skew=" << plan_stats.planned_owned_skew
                          << " constrained_bins=" << plan_stats.constrained_bins
                          << " pieces=" << plan_stats.pieces
                          << " segments=" << plan_stats.segments
                          << " last_event_mesh=" << last_event_mesh
                          << " last_event_class=" << measured_class
                          << " reference_class=" << reference_source
                          << " credited_events=" << decision.credited_events
                          << " measured_events=" << decision.measured_events
                          << " positional_samples=" << positional_samples
                          << " baseline=" << decision.baseline_mesh
                          << " debt=" << decision.debt
                          << " last_migration=" << decision.last_migration_seconds
                          << " net_benefit=" << decision.net_benefit
                          << " negative_windows=" << decision.negative_windows
                          << " reverts=" << decision.reverts
                          << " segment_cost_total=" << this->individualSegmentMigrationSeconds
                          << " active_spread=" << current_load.activeMaxMean() << std::endl;
            decision.pending_class = IndividualSegmentEventClass(
                current_load.active_cells, current_load.owned_cells);
        }
        else if(cache_hit)
        {
            int local_cached_is_current = current_positional &&
                cached->second == current_load_balance->getBoundaries() ? 1 : 0;
            int minimum_cached_is_current = local_cached_is_current;
            int maximum_cached_is_current = local_cached_is_current;
            MPI_Allreduce(MPI_IN_PLACE, &minimum_cached_is_current, 1, MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &maximum_cached_is_current, 1, MPI_INT,
                          MPI_MAX, MPI_COMM_WORLD);
            if(minimum_cached_is_current != maximum_cached_is_current)
                throw std::logic_error(
                    "Active Hilbert cached cut comparison differs across MPI ranks");
            proposed_load_balance = current_load_balance->clone();
            proposed_load_balance->setSegments(cached->second, {});
            proposed_load = maximum_cached_is_current != 0 ?
                current_position_load : measureIndividualHilbertPartition(
                    *proposed_load_balance, points, context.active_mask,
                    this->size);
            proposal_source = "cache";
            if(!proposed_load.ownsEveryRank())
            {
                proposed_load_balance.reset();
                rebuild_boundaries = true;
            }
            else if(proposed_load.ownedMaxMean() >
               options.maximum_owned_cell_skew)
            {
                proposed_load_balance.reset();
                rebuild_boundaries = true;
            }
            else if(proposed_load.activeBalanced(options.active_threshold))
                action = "reuse-cache";
            else
            {
                proposed_load_balance.reset();
                rebuild_boundaries = true;
            }
        }

        if(rebuild_boundaries && rebuild_exceeds_owned_skew &&
           !options.bound_check && this->individualActiveHilbertBoundTrusted)
        {
            // The rebuild would be rejected for its owned-cell skew (or an
            // empty rank); skip its root-gathered sort of every generator.
            proposed_load_balance.reset();
            proposal_source = "none";
            action = "skip-rebuild-owned-skew-bound";
            ++this->individualActiveHilbertSkippedRebuilds;
        }
        else if(rebuild_boundaries)
        {
            ++this->individualActiveHilbertBuiltRebuilds;
            proposed_load_balance = current_load_balance->clone();
            proposed_load_balance->rebalance(points, active_weights);
            proposed_load = measureIndividualHilbertPartition(
                *proposed_load_balance, points, context.active_mask,
                this->size);
            proposal_source = "rebuild";
            // Both sides are global, so every rank takes the same branch.
            if(proposed_load.maximum_owned_cells < cut_bound.cells)
            {
                std::string const violation =
                    "Active Hilbert rebuild owns fewer cells on its largest rank ("
                    + std::to_string(proposed_load.maximum_owned_cells) +
                    ") than the passive-run bound (" +
                    std::to_string(cut_bound.cells) + ")";
                if(options.bound_check)
                    throw std::logic_error(violation);
                // Production: stop trusting the bound; the measured acceptance
                // checks below still decide this and every later rebuild.
                this->individualActiveHilbertBoundTrusted = false;
                if(this->rank == 0)
                    std::cout << "INDIVIDUAL_ACTIVE_HILBERT_BOUND_VIOLATION cycle="
                              << this->tracker.getCycle() << " event_tick="
                              << eventTick << " active_bin_mask="
                              << active_bin_mask << " bound_cells="
                              << cut_bound.cells << " bound_first_segment="
                              << cut_bound.first_segment << " bound_segments="
                              << cut_bound.segments << " proposed_max_owned="
                              << proposed_load.maximum_owned_cells
                              << " action=skips_disabled (" << violation << ")"
                              << std::endl;
            }
            if(!proposed_load.ownsEveryRank())
            {
                proposed_load_balance.reset();
                action = "reject-rebuild-empty-rank";
            }
            else if(proposed_load.ownedMaxMean() >
               options.maximum_owned_cell_skew)
            {
                proposed_load_balance.reset();
                action = "reject-rebuild-owned-skew";
            }
            else if(!proposed_load.activeBalanced(options.active_threshold))
            {
                proposed_load_balance.reset();
                action = "reject-rebuild-active-skew";
            }
            else
            {
                this->individualActiveHilbertBoundaries[active_bin_mask] =
                    proposed_load_balance->getBoundaries();
                action = "rebuild";
            }
        }
    }

    IndividualRebalanceResult migration;
    bool migrated = false;
    if(revert_to_positional)
    {
        migration = this->rebalanceToPositionalOwnership(balance_step);
        migrated = migration.applied;
        std::shared_ptr<HilbertLoadBalancer<Vector3D>> const reverted =
            std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
                this->tess.GetLoadBalancer());
        int positional = migrated && reverted &&
            reverted->positionalOwnership() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &positional, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(positional == 0)
            throw std::logic_error(
                "Segmented Hilbert ownership did not revert to positional");
        this->individualSegmentMigrationSeconds += migration.maximumSeconds;
        // A fresh positional reference period; the revert cooldown and the
        // revert count stay.
        IndividualSegmentDecisionState& decision = this->individualSegmentDecision;
        IndividualSegmentDecisionState fresh;
        fresh.last_migration_seconds = migration.maximumSeconds;
        fresh.cooldown_until_event = decision.cooldown_until_event;
        fresh.event_counter = decision.event_counter;
        fresh.reverts = decision.reverts;
        fresh.pending_class = decision.pending_class;
        decision = fresh;
        bool const all_active_event =
            current_load.active_cells == current_load.owned_cells;
        context = this->individualScheduler->prepareEvent(
            this->cells, eventTick, all_active_event);
    }
    else if(proposed_load_balance)
    {
        int local_partition_change =
            !proposed_load_balance->sameAssignment(*current_load_balance) ? 1 : 0;
        int minimum_partition_change = local_partition_change;
        int maximum_partition_change = local_partition_change;
        MPI_Allreduce(MPI_IN_PLACE, &minimum_partition_change, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_partition_change, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);
        if(minimum_partition_change != maximum_partition_change)
            throw std::logic_error(
                "Active Hilbert partition decision differs across MPI ranks");
        if(maximum_partition_change != 0 || force_segment_migration)
        {
            bool const all_active_event =
                proposed_load.active_cells == proposed_load.owned_cells;
            migration = this->rebalanceCommittedIndividualState(
                balance_step, true, options.active_threshold,
                std::move(active_weights), proposed_load_balance);
            migrated = migration.applied;
            std::string const source = proposal_source;
            if(migrated && source == "segmented")
            {
                // A migration into or within segmented ownership: charge it
                // and its planning once, restart the per-period state.
                this->individualSegmentMigrationSeconds += migration.maximumSeconds;
                IndividualSegmentDecisionState& decision =
                    this->individualSegmentDecision;
                if(current_positional)
                {
                    // A new probe: its age, windows and ledger start here.
                    decision.net_benefit = 0;
                    decision.probe_events = 0;
                    decision.measured_events = 0;
                    decision.credited_events = 0;
                    decision.negative_windows = 0;
                    decision.window_events = 0;
                    decision.window_saving = 0;
                    decision.window_credited = 0;
                    decision.window_measured_seconds = 0;
                    decision.window_credited_seconds = 0;
                    decision.measured_seconds = 0;
                    decision.credited_seconds = 0;
                    decision.class_seconds.fill(0);
                }
                // An internal re-plan keeps the probe's age, windows and
                // ledger (so re-plans cannot postpone a revert); only the
                // debt baseline restarts.
                decision.net_benefit -= migration.maximumSeconds +
                    decision.pending_planning_seconds;
                decision.pending_planning_seconds = 0;
                decision.last_migration_seconds = migration.maximumSeconds;
                decision.events_since_migration = 0;
                decision.baseline_mesh = 0;
                decision.baseline_pending = true;
                decision.debt = 0;
            }
            context = this->individualScheduler->prepareEvent(
                this->cells, eventTick, all_active_event);
        }
        else
            action += "-already-current";
    }

    double local_decision_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - decision_start).count();
    // MPI time over the same interval as the wall time (before the timing
    // reduction below), so busy = wall - MPI covers identical code.
    double const decision_mpi_seconds =
        mpi_wait_profiler::Seconds() - decision_mpi_start;
    double maximum_decision_seconds = local_decision_seconds;
    MPI_Allreduce(MPI_IN_PLACE, &maximum_decision_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    this->lastPhysicsTimes["individual-active-hilbert-balance"] =
        maximum_decision_seconds;
    this->lastLocalPhysicsTimes["individual-active-hilbert-balance"] =
        local_decision_seconds;
    this->lastLocalPhysicsMpiTimes["individual-active-hilbert-balance"] =
        decision_mpi_seconds;

    bool const proposal_measured = std::string(proposal_source) != "none";
    if(options.trace && this->rank == 0)
        std::clog << std::setprecision(17)
                  << "INDIVIDUAL_ACTIVE_HILBERT_DECISION"
                  << " cycle=" << this->tracker.getCycle()
                  << " event_tick=" << eventTick
                  << " active_bin_mask=" << active_bin_mask
                  << " cache_hit=" << (cache_hit ? 1 : 0)
                  << " action=" << action
                  << " active_cells=" << current_load.active_cells
                  << " current_active_max_mean="
                  << current_load.activeMaxMean()
                  << " proposal_source=" << proposal_source
                  << " proposed_active_max_mean=" << (proposal_measured ?
                      std::to_string(proposed_load.activeMaxMean()) : "nan")
                  << " proposed_owned_max_mean=" << (proposal_measured ?
                      std::to_string(proposed_load.ownedMaxMean()) : "nan")
                  << " proposed_owned_min=" << (proposal_measured ?
                      std::to_string(proposed_load.minimum_owned_cells) : "nan")
                  << " active_threshold=" << options.active_threshold
                  << " maximum_owned_cell_skew="
                  << options.maximum_owned_cell_skew
                  << " cached_partitions="
                  << this->individualActiveHilbertBoundaries.size()
                  << " cut_bound_max_mean=" << cut_bound_max_mean
                  << " cut_bound_segments=" << cut_bound.segments
                  << " cut_bound_first_segment=" << cut_bound.first_segment
                  << " bound_check=" << (options.bound_check ? 1 : 0)
                  << " bound_trusted="
                  << (this->individualActiveHilbertBoundTrusted ? 1 : 0)
                  << " skipped_rebuilds_total="
                  << this->individualActiveHilbertSkippedRebuilds
                  << " built_rebuilds_total="
                  << this->individualActiveHilbertBuiltRebuilds
                  << " migrated=" << (migrated ? 1 : 0)
                  << " migrated_cells=" << migration.migratedCells
                  << " migration_seconds_max=" << migration.maximumSeconds
                  << " decision_seconds_max=" << maximum_decision_seconds
                  << " ownership_epoch=" << this->individualOwnershipEpoch
                  << std::endl;
}
#endif

void Simulation::stepIndividual(void)
{
    bool const detailed_runtime_log = RuntimeLogDetailed();
    bool const runtime_color_enabled = RuntimeColorEnabled();
    if(!this->individualScheduler)
        throw std::logic_error("Individual timestep mode has no scheduler");
    if(!this->individualScheduler->initialized())
    {
        for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
            if(!physicsStep->supportsIndividualTimeSteps())
                throw std::invalid_argument("Physics step '" + physicsStep->getName() +
                                            "' does not support individual timesteps");
#ifdef RICH_MPI
        // Canonical scheduler state is owned-cell only.  Ghost primitives are
        // reconstructed collectively by each physics step from the current
        // tessellation exchange map.
        this->cells.resize(this->tess.GetPointNo());
        this->extensives.resize(this->tess.GetPointNo());
        double initial_dt_min = this->tsc->GetTimeStep();
        double initial_dt_max = initial_dt_min;
        MPI_Allreduce(MPI_IN_PLACE, &initial_dt_min, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &initial_dt_max, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        if(initial_dt_min != initial_dt_max)
            throw std::logic_error(
                "MPI individual timestep initialization requires one shared base timestep");
        this->tsc->SetTimeStep(initial_dt_min);
#endif
        // The anchor is the grid the gain estimate used: m times the last
        // global step's hydro/source suggestion (MPI minimum, taken before its
        // post-step AMR).  Without one (a fresh individual start) the grid stays
        // on the initial step.  The reference is collective, so every rank
        // takes the same branch.
        // An explicitly configured quantum fixes the grid itself, so it keeps it.
        double const anchor_margin = IndividualBinAnchorMargin();
        double const anchor_reference = this->adaptiveMode.anchorReferenceStep;
        double const anchor_dt = anchor_margin > 0 && std::isfinite(anchor_reference) &&
            anchor_reference > 0 && !(this->individualScheduler->options().time_quantum > 0) ?
            anchor_margin * anchor_reference : 0;
        this->individualScheduler->initialize(this->cells,
                                              this->tracker.getTime(),
                                              this->tsc->GetTimeStep(),
                                              anchor_dt);
        for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
            physicsStep->onIndividualSchedulerStart(this->tsc->GetTimeStep());
        if(this->rank == 0 && anchor_margin > 0)
            std::cout << std::setprecision(12)
                      << "RICH_INDIVIDUAL_BIN_ANCHOR time=" << this->tracker.getTime()
                      << " global_dt=" << this->tsc->GetTimeStep()
                      << " anchor_margin=" << anchor_margin
                      << " anchor_reference=" << anchor_reference
                      << " anchor_dt=" << (anchor_dt > 0 ? anchor_dt : this->tsc->GetTimeStep())
                      << " first_interval=" << this->individualScheduler->nextEventTimeStep() << std::endl;
        this->individualScheduler->enforceNeighborBinClosure(
            this->tess, this->cells);
    }

    this->lastPhysicsTimes.clear();
    this->lastLocalPhysicsTimes.clear();
    this->lastLocalPhysicsMpiTimes.clear();
    const auto stepWallStart = std::chrono::high_resolution_clock::now();
    double const step_mpi_start = mpi_wait_profiler::Seconds();
    // Exclusive per-rank timers for the scheduler-side sections outside the
    // physics steps (decision 2026-09-22 D2, plan r2 step 3).  Rank maxima
    // are taken once, at output; no control flow depends on them.
    // prepare_s covers the prepareEvent calls made here; the one inside
    // active-Hilbert balancing stays in that phase.
    double individual_sync_seconds = 0;
    double individual_prepare_seconds = 0;
    double individual_closure_seconds = 0;
    double individual_suggest_seconds = 0;
    double individual_wake_seconds = 0;
    double individual_sweep_seconds = 0;
    double individual_commit_seconds = 0;
    auto const seconds_since =
        [](std::chrono::high_resolution_clock::time_point const& start)
    {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
    };
    std::size_t const step_cycle = this->tracker.getCycle();
    SourceStepTiming local_source_timing;
    MeshBuildTiming local_mesh_build_timing;
#ifndef RICH_MPI
    requireSerialIndividualActiveHilbertCompatibility();
#endif
#ifdef RICH_MPI
    this->individualMeshBuildTiming = MeshBuildTiming();
    bool const active_hilbert_balance =
        this->individualActiveHilbertBalanceEnabled();
    // Restart state is already committed.  Any ownership migration must
    // finish before prepareEvent creates an in-flight event context.
    this->maybeRebalanceBeforeFirstIndividualEvent(active_hilbert_balance);
#endif

    this->individualEventInProgress = true;
    struct EventProgressReset
    {
        bool& flag;
        ~EventProgressReset() { flag = false; }
    } event_progress_reset{this->individualEventInProgress};

    std::uint64_t event_tick = this->individualScheduler->nextEventTick();
    bool force_all_active_latched =
        this->individualScheduler->forceAllActiveLatched();
    bool force_all_active_once = this->individualSynchronizedEventRequested;
#ifdef RICH_MPI
    std::uint64_t const local_current_tick =
        this->individualScheduler->currentTick();
    std::uint64_t event_collective[7] = {
        event_tick, force_all_active_latched ? 1u : 0u,
        force_all_active_latched ? 0u : 1u, local_current_tick,
        std::numeric_limits<std::uint64_t>::max() - local_current_tick,
        force_all_active_once ? 1u : 0u,
        force_all_active_once ? 0u : 1u};
    {
        auto const sync_start = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(MPI_IN_PLACE, event_collective, 7, MPI_UINT64_T,
                      MPI_MIN, MPI_COMM_WORLD);
        individual_sync_seconds += seconds_since(sync_start);
    }
    event_tick = event_collective[0];
    if(event_collective[1] == 0 && event_collective[2] == 0)
        throw std::logic_error(
            "Inconsistent force-all-active latch state across MPI ranks");
    std::uint64_t const maximum_current_tick =
        std::numeric_limits<std::uint64_t>::max() - event_collective[4];
    if(event_collective[3] != maximum_current_tick)
        throw std::logic_error(
            "Inconsistent individual scheduler clock across MPI ranks");
    if(event_collective[5] == 0 && event_collective[6] == 0)
        throw std::logic_error(
            "Inconsistent synchronized-event request across MPI ranks");
    force_all_active_latched = event_collective[1] != 0;
    force_all_active_once = event_collective[5] != 0;
#endif
    if(event_tick == std::numeric_limits<std::uint64_t>::max())
        throw std::logic_error("Individual timestep scheduler has no cells on any rank");
    auto const prepare_start = std::chrono::high_resolution_clock::now();
    IndividualStepContext context =
        this->individualScheduler->prepareEvent(this->cells, event_tick);
    individual_prepare_seconds += seconds_since(prepare_start);
#ifndef RICH_MPI
    if((force_all_active_latched || force_all_active_once) &&
       context.active_indices.size() != this->cells.size())
    {
        auto const prepare_again = std::chrono::high_resolution_clock::now();
        context = this->individualScheduler->prepareEvent(
            this->cells, event_tick, true);
        individual_prepare_seconds += seconds_since(prepare_again);
    }
#endif
#ifdef RICH_MPI
    IndividualForceAllActiveRuntimeOptions const& force_options =
        individualForceAllActiveRuntimeOptions();
    auto const closure_start = std::chrono::high_resolution_clock::now();
    double closure_sync_seconds = 0;
    double closure_prepare_seconds = 0;
    double closure_release_seconds = 0;
    if(force_options.enabled || force_all_active_latched ||
       force_all_active_once)
    {
        std::uint64_t closure_counts[2] = {
            static_cast<std::uint64_t>(this->cells.size()),
            static_cast<std::uint64_t>(context.active_indices.size())};
        std::uint64_t maximum_active_interval_ticks = 0;
        std::vector<CellTimeState> const& states =
            this->individualScheduler->states();
        for(std::size_t const active : context.active_indices)
        {
            CellTimeState const& state = states.at(active);
            if(state.begin_tick >= event_tick)
                throw std::logic_error(
                    "Active individual cell has a non-advancing event interval");
            maximum_active_interval_ticks = std::max(
                maximum_active_interval_ticks,
                event_tick - state.begin_tick);
        }
        {
            auto const sync_start = std::chrono::high_resolution_clock::now();
            MPI_Allreduce(MPI_IN_PLACE, closure_counts, 2, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &maximum_active_interval_ticks, 1,
                          MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
            closure_sync_seconds += seconds_since(sync_start);
        }
        std::uint64_t const force_threshold_ticks =
            force_options.enabled ?
            (std::uint64_t(1) << force_options.minimum_bin) : 0;
        bool const threshold_reached = force_options.enabled &&
            maximum_active_interval_ticks >= force_threshold_ticks;
        bool const latch_now = force_all_active_latched ||
            (force_options.latch_enabled && threshold_reached);
        if(latch_now && !force_all_active_latched)
        {
            this->individualScheduler->setForceAllActiveLatched(true);
            auto const release_start =
                std::chrono::high_resolution_clock::now();
            for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
                physicsStep->onIndividualForceAllActiveLatch();
            double release_seconds = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() -
                release_start).count();
            closure_release_seconds += release_seconds;
            {
                auto const sync_start = std::chrono::high_resolution_clock::now();
                MPI_Allreduce(MPI_IN_PLACE, &release_seconds, 1, MPI_DOUBLE,
                              MPI_MAX, MPI_COMM_WORLD);
                closure_sync_seconds += seconds_since(sync_start);
            }
            if(this->rank == 0)
                std::clog << std::setprecision(17)
                          << "INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH_SET"
                          << " cycle=" << this->tracker.getCycle()
                          << " previous_event_tick="
                          << this->individualScheduler->currentTick()
                          << " event_tick=" << event_tick
                          << " maximum_active_interval_ticks="
                          << maximum_active_interval_ticks
                          << " maximum_active_interval_dt="
                          << (this->individualScheduler->timeQuantum() *
                              static_cast<double>(
                                  maximum_active_interval_ticks))
                          << " minimum_bin=" << force_options.minimum_bin
                          << " threshold_ticks=" << force_threshold_ticks
                          << " global_active=" << closure_counts[1]
                          << " global_owned_cells=" << closure_counts[0]
                          << " topology_release_seconds_max="
                          << release_seconds
                          << std::endl;
        }
        bool const force_all_active = closure_counts[1] < closure_counts[0] &&
            (latch_now || threshold_reached || force_all_active_once);
        if(force_all_active)
        {
            std::uint64_t const global_promoted_cells =
                closure_counts[0] - closure_counts[1];
            {
                auto const prepare_again = std::chrono::high_resolution_clock::now();
                context = this->individualScheduler->prepareEvent(
                    this->cells, event_tick, true);
                closure_prepare_seconds += seconds_since(prepare_again);
            }
            if(this->rank == 0)
                std::clog << std::setprecision(17)
                          << "INDIVIDUAL_FORCE_ALL_ACTIVE"
                          << " reason=" << (force_all_active_once ?
                              "requested" : (latch_now ? "latched" : "threshold"))
                          << " latched=" << (latch_now ? 1 : 0)
                          << " cycle=" << this->tracker.getCycle()
                          << " previous_event_tick="
                          << this->individualScheduler->currentTick()
                          << " event_tick=" << event_tick
                          << " event_interval_ticks="
                          << (event_tick -
                              this->individualScheduler->currentTick())
                          << " maximum_active_interval_ticks="
                          << maximum_active_interval_ticks
                          << " maximum_active_interval_dt="
                          << (this->individualScheduler->timeQuantum() *
                              static_cast<double>(
                                  maximum_active_interval_ticks))
                          << " minimum_bin=" << force_options.minimum_bin
                          << " threshold_enabled="
                          << (force_options.enabled ? 1 : 0)
                          << " threshold_ticks=" << force_threshold_ticks
                          << " global_active_before=" << closure_counts[1]
                          << " global_owned_cells=" << closure_counts[0]
                          << " global_promoted_cells="
                          << global_promoted_cells << std::endl;
        }
    }
    individual_sync_seconds += closure_sync_seconds;
    individual_prepare_seconds += closure_prepare_seconds;
    if(closure_release_seconds > 0)
        this->lastLocalPhysicsTimes["individual-topology-release"] +=
            closure_release_seconds;
    // Local closure work only: the block minus its collectives, its
    // prepareEvent and the latch-release loop (reported separately as
    // topology_release_seconds_max).
    individual_closure_seconds += std::max(0.0, seconds_since(closure_start) -
        closure_sync_seconds - closure_prepare_seconds -
        closure_release_seconds);
#endif
#ifdef RICH_MPI
    if(active_hilbert_balance)
        this->maybeBalanceIndividualEventByActiveBins(context, event_tick);
    IndividualRebalanceRuntimeOptions const& balance_options =
        individualRebalanceRuntimeOptions();
#endif
    std::vector<double> timeStepLimits(
        this->individualScheduler->states().size(),
                                      std::numeric_limits<double>::infinity());
    std::vector<double> signalWakeDeadlines(
        this->individualScheduler->states().size(),
        std::numeric_limits<double>::infinity());
    std::vector<double> changeWakeRatios(
        this->individualScheduler->states().size(), 0.0);

    std::uint64_t const local_active_cells =
        static_cast<std::uint64_t>(context.active_indices.size());
    std::array<std::uint64_t, 2> global_cell_counts{{
        local_active_cells,
        static_cast<std::uint64_t>(this->cells.size())}};
#ifdef RICH_MPI
    {
        auto const sync_start = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(MPI_IN_PLACE, global_cell_counts.data(),
                      static_cast<int>(global_cell_counts.size()), MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD);
        individual_sync_seconds += seconds_since(sync_start);
    }
#endif
    std::uint64_t const global_active_cells = global_cell_counts[0];
    std::uint64_t const global_total_cells = global_cell_counts[1];
    std::array<std::uint64_t, 63> active_bin_counts{};
    double local_applied_dt_min = std::numeric_limits<double>::infinity();
    double local_applied_dt_max = 0;
    std::vector<CellTimeState> const& scheduler_states =
        this->individualScheduler->states();
    for(std::size_t const active_index : context.active_indices)
    {
        if(active_index >= scheduler_states.size() ||
           scheduler_states[active_index].time_bin >= active_bin_counts.size())
            throw std::logic_error("Individual active cell has an invalid time bin");
        ++active_bin_counts[scheduler_states[active_index].time_bin];
        double const active_dt = context.cellTimeStep(active_index);
        local_applied_dt_min = std::min(local_applied_dt_min, active_dt);
        local_applied_dt_max = std::max(local_applied_dt_max, active_dt);
    }
#ifdef RICH_MPI
    {
        auto const sync_start = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(MPI_IN_PLACE, active_bin_counts.data(),
                      static_cast<int>(active_bin_counts.size()), MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD);
        individual_sync_seconds += seconds_since(sync_start);
    }
#endif
    double const applied_dt_min = RuntimeMinimum(local_applied_dt_min);
    double const applied_dt_max = RuntimeMaximum(local_applied_dt_max);
    std::ostringstream active_bins_stream;
    active_bins_stream << '['
                       << std::setprecision(runtime_simulation_digits);
    bool first_active_bin = true;
    for(std::size_t bin = 0; bin < active_bin_counts.size(); ++bin)
        if(active_bin_counts[bin] > 0)
        {
            if(!first_active_bin)
                active_bins_stream << "; ";
            active_bins_stream << "bin=" << bin
                << ",count=" << active_bin_counts[bin]
                << ",dt="
                << std::ldexp(this->individualScheduler->timeQuantum(),
                              static_cast<int>(bin));
            first_active_bin = false;
        }
    active_bins_stream << ']';
    std::string const active_bins = active_bins_stream.str();
    RuntimeStepOutput retry_output;
    retry_output.mode = "individual";
    retry_output.cycle = step_cycle;
    retry_output.active_cells = global_active_cells;
    retry_output.active_bins = active_bins;

    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
    {
        const std::string name = physicsStep->getName();
        std::size_t retry_attempt = 0;
        int const runtime_rank = this->rank;
        physicsStep->setStepRetryReporter(
            [retry_output, name, retry_attempt, runtime_rank,
             runtime_color_enabled](
                StepRetryRecord const& retry) mutable
            {
                ReportRuntimeRetry(runtime_rank, runtime_color_enabled,
                                   retry_output, name,
                                   ++retry_attempt, retry);
            });
        MEMORY_DEBUG_PRINT("Before individual " + name);
        const auto start = std::chrono::high_resolution_clock::now();
        double const physics_mpi_start = mpi_wait_profiler::Seconds();
        physicsStep->stepIndividual(context);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastPhysicsTimes[name] = elapsed;
        this->lastLocalPhysicsTimes[name] = elapsed;
        this->lastLocalPhysicsMpiTimes[name] =
            mpi_wait_profiler::Seconds() - physics_mpi_start;
        for(auto const& counter :
            physicsStep->getIndividualPerformanceCounters())
            this->lastLocalPhysicsTimes[counter.first] = counter.second;
        SourceStepTiming const source_timing =
            physicsStep->getSourceStepTiming();
        local_source_timing.first_seconds += source_timing.first_seconds;
        local_source_timing.second_seconds += source_timing.second_seconds;
        local_source_timing.calls += source_timing.calls;
        MeshBuildTiming const mesh_build_timing =
            physicsStep->getMeshBuildTiming();
        local_mesh_build_timing.seconds += mesh_build_timing.seconds;
        local_mesh_build_timing.builds += mesh_build_timing.builds;
        local_mesh_build_timing.full_builds += mesh_build_timing.full_builds;
        MEMORY_DEBUG_PRINT("After individual " + name);
        if(detailed_runtime_log && this->rank == 0)
            std::cout << "Individual physics " << name << " time: "
                      << elapsed << std::endl;
    }

    if(this->individualPostPhysics)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        this->individualPostPhysics(context);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastPhysicsTimes["individual-post-physics"] = elapsed;
        this->lastLocalPhysicsTimes["individual-post-physics"] = elapsed;
    }

    // Cadence diagnostic (RICH_INDIVIDUAL_CADENCE_TRACE=1, rank-consistent):
    // which physics step set each cell's limit, and after the commit why the
    // active cells assigned the finest bin got it.
    // Gates collectives (the change-wake statistics), so it must agree.
    static bool const cadence_trace = []()
    {
        bool valid = true;
        bool const value = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_CADENCE_TRACE", false, valid);
        int code[2] = {valid ? (value ? 1 : 0) : -1, 0};
        code[1] = -code[0];
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, code, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(code[0] < 0 || code[0] != -code[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_CADENCE_TRACE must be one boolean on every rank");
        return value;
    }();
    std::vector<unsigned char> cadence_limit_step;
    std::vector<double> cadence_limits_before;
    if(cadence_trace)
        cadence_limit_step.assign(timeStepLimits.size(), 0);
    std::size_t cadence_step_index = 0;
    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        if(cadence_trace)
            cadence_limits_before = timeStepLimits;
        physicsStep->suggestIndividualTimeSteps(context, timeStepLimits);
        if(cadence_trace)
            for(std::size_t i = 0; i < timeStepLimits.size(); ++i)
                if(timeStepLimits[i] < cadence_limits_before[i])
                    cadence_limit_step[i] =
                        static_cast<unsigned char>(cadence_step_index + 1);
        ++cadence_step_index;
        physicsStep->suggestIndividualWakeDeadlines(
            context, signalWakeDeadlines);
        physicsStep->suggestIndividualChangeWakes(context, changeWakeRatios);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        // Reported as its own section (suggest_s) so hydro_s/radiation_s
        // stay exclusive.
        individual_suggest_seconds += elapsed;
    }
    this->lastPhysicsTimes["individual-suggest"] = individual_suggest_seconds;
    this->lastLocalPhysicsTimes["individual-suggest"] = individual_suggest_seconds;

    auto const wake_start = std::chrono::high_resolution_clock::now();
    IndividualFullSourceSweepReport const full_source_sweep =
        limitIndividualTreeWakeTimeSteps(
            context, this->tess, this->cells, this->eos, this->physics,
            scheduler_states, timeStepLimits,
            signalWakeDeadlines,
            this->individualScheduler->lastFullSourceSweepTick(),
            this->individualScheduler->options().
                full_source_sweep_interval_minimum_steps);

    {
        double const wake_total = seconds_since(wake_start);
        individual_sweep_seconds = full_source_sweep.performed ?
            full_source_sweep.seconds_local : 0;
        individual_wake_seconds = std::max(0.0, wake_total - individual_sweep_seconds);
    }
#ifdef RICH_MPI
    {
        static std::uint64_t routing_rebuilds = 0;
        static std::uint64_t routing_reuses = 0;
        routing_rebuilds += full_source_sweep.routing_rebuilds;
        routing_reuses += full_source_sweep.routing_reuses;
        if(this->rank == 0 && individualActiveHilbertRuntimeOptions().trace &&
           full_source_sweep.routing_rebuilds + full_source_sweep.routing_reuses > 0 &&
           (routing_rebuilds + routing_reuses) % 20 == 0)
            std::cout << "INDIVIDUAL_WAKE_ROUTING cycle=" << this->tracker.getCycle()
                      << " rebuilds=" << routing_rebuilds
                      << " reuses=" << routing_reuses
                      << " tree_seconds_local=" << full_source_sweep.tree_seconds
                      << std::endl;
    }
#endif
    // Sub-phases of wake_s + sweep_s (S5); nested, so not subtracted from other_s.
    this->lastLocalPhysicsTimes["individual-wake-prepare"] =
        full_source_sweep.prepare_seconds;
    this->lastLocalPhysicsTimes["individual-wake-sources"] =
        full_source_sweep.sources_seconds;
    this->lastLocalPhysicsTimes["individual-wake-tree"] =
        full_source_sweep.tree_seconds;
    this->lastLocalPhysicsTimes["individual-wake-local-route"] =
        full_source_sweep.local_route_seconds;
    this->lastLocalPhysicsTimes["individual-wake-local-eval"] =
        full_source_sweep.local_eval_seconds;
    this->lastLocalPhysicsTimes["individual-wake-route"] =
        full_source_sweep.route_seconds;
    this->lastLocalPhysicsTimes["individual-wake-exchange"] =
        full_source_sweep.exchange_seconds;
    this->lastLocalPhysicsTimes["individual-wake-remote"] =
        full_source_sweep.remote_seconds;
    std::vector<std::uint8_t> cadence_old_bins;
    if(cadence_trace)
    {
        std::vector<CellTimeState> const& before_commit =
            this->individualScheduler->states();
        cadence_old_bins.reserve(context.active_indices.size());
        for(std::size_t const global : context.active_indices)
            cadence_old_bins.push_back(global < before_commit.size() ?
                before_commit[global].time_bin : 0);
    }
    auto const commit_start = std::chrono::high_resolution_clock::now();
    this->individualScheduler->commitEvent(context, this->tess,
                                           this->cells, timeStepLimits,
                                           signalWakeDeadlines,
                                           changeWakeRatios);
    individual_commit_seconds += seconds_since(commit_start);
    if(cadence_trace)
    {
        // Categories: 0 hydro CFL/source, 1 mesh drift, 2 mass loss,
        // 3 thermal loss, 4 another physics step (radiation), 5 one-bin
        // growth cap, 6 scheduler below the physics bin (neighbour closure,
        // wake, synchronisation), 7 unattributed.
        constexpr int categories = 8;
        std::vector<unsigned char> hydro_reasons;
        std::size_t hydro_step = this->physics.size();
        for(std::size_t k = 0; k < this->physics.size(); ++k)
            if(this->physics[k]->contributesIndividualHydrodynamicSignal() &&
               this->physics[k]->getIndividualLimitReasons(hydro_reasons))
            {
                hydro_step = k;
                break;
            }
        std::vector<CellTimeState> const& after_commit =
            this->individualScheduler->states();
        int const maximum_bin = static_cast<int>(
            this->individualScheduler->options().maximum_bin);
        std::vector<int> category(context.active_indices.size(), 7);
        std::vector<int> physics_bin(context.active_indices.size(), maximum_bin);
        int finest = std::numeric_limits<int>::max();
        for(std::size_t k = 0; k < context.active_indices.size(); ++k)
        {
            std::size_t const global = context.active_indices[k];
            if(global >= after_commit.size() || global >= timeStepLimits.size())
                continue;
            int const new_bin = after_commit[global].time_bin;
            finest = std::min(finest, new_bin);
            double const limit = timeStepLimits[global];
            // The scheduler's own quantization (saturating, trap-safe); a limit
            // it would reject leaves the cell counted as unconstrained here.
            if(limit > 0 && (std::isfinite(limit) || std::isinf(limit)))
                physics_bin[k] = this->individualScheduler->binForTimeStep(limit);
            if(new_bin < physics_bin[k])
                category[k] = new_bin == static_cast<int>(cadence_old_bins[k]) + 1 ?
                    5 : 6;
            else
            {
                unsigned char const step = global < cadence_limit_step.size() ?
                    cadence_limit_step[global] : 0;
                if(step == 0)
                    category[k] = 7;
                else if(static_cast<std::size_t>(step - 1) == hydro_step &&
                        global < hydro_reasons.size() &&
                        hydro_reasons[global] >= 1 && hydro_reasons[global] <= 4)
                    category[k] = hydro_reasons[global] - 1;
                else if(static_cast<std::size_t>(step - 1) == hydro_step)
                    category[k] = 7;
                else
                    category[k] = 4;
            }
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &finest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        // Slot 2*categories: active cells whose committed interval ends before
        // its bin's full length (signal wake or alignment), which moves the
        // next event without changing any bin.
        unsigned long long counts[2 * categories + 1] = {};
        struct { double value; int rank; } example{
            std::numeric_limits<double>::infinity(), this->rank};
        std::size_t example_k = context.active_indices.size();
        for(std::size_t k = 0; k < context.active_indices.size(); ++k)
        {
            std::size_t const global = context.active_indices[k];
            if(global >= after_commit.size())
                continue;
            ++counts[categories + category[k]];
            CellTimeState const& committed = after_commit[global];
            if(committed.end_tick > committed.begin_tick &&
               committed.end_tick - committed.begin_tick <
                   this->individualScheduler->binTicks(committed.time_bin))
                ++counts[2 * categories];
            if(committed.time_bin != finest)
                continue;
            ++counts[category[k]];
            // A present candidate always beats an absent one in the MINLOC
            // below, even when its limit is unconstrained (infinite).
            double const limit = std::min(timeStepLimits[global],
                std::numeric_limits<double>::max());
            if(limit < example.value || example_k == context.active_indices.size())
            {
                example.value = limit;
                example_k = k;
            }
        }
        double example_record[7] = {-1, -1, -1, -1, -1, -1, -1};
        if(example_k < context.active_indices.size())
        {
            std::size_t const global = context.active_indices[example_k];
            ComputationalCell3D const& cell = this->cells[global];
            example_record[0] = static_cast<double>(cell.ID);
            example_record[1] = cadence_old_bins[example_k];
            example_record[2] = after_commit[global].time_bin;
            example_record[3] = physics_bin[example_k];
            example_record[4] = category[example_k];
            example_record[5] = cell.density;
            example_record[6] = fastabs(cell.velocity);
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, counts, 2 * categories + 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &example, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                      MPI_COMM_WORLD);
        unsigned long long example_id =
            static_cast<unsigned long long>(example_record[0] < 0 ? 0 : example_record[0]);
        if(example_k < context.active_indices.size())
            example_id = static_cast<unsigned long long>(
                this->cells[context.active_indices[example_k]].ID);
        MPI_Bcast(&example_id, 1, MPI_UNSIGNED_LONG_LONG, example.rank, MPI_COMM_WORLD);
        MPI_Bcast(example_record, 7, MPI_DOUBLE, example.rank, MPI_COMM_WORLD);
#else
        unsigned long long const example_id =
            static_cast<unsigned long long>(example_record[0] < 0 ? 0 : example_record[0]);
#endif
        if(this->rank == 0)
        {
            static char const* const names[categories] = {"cfl", "drift", "mass",
                "thermal", "radiation", "growth_cap", "scheduler", "unattributed"};
            std::ostringstream line;
            // Next-bin attribution for this event's active cells; the
            // shortened count covers wake/alignment-shortened intervals.
            line << std::setprecision(6) << "INDIVIDUAL_CADENCE event_tick="
                 << context.event_tick << " finest_new_bin=" << finest;
            unsigned long long finest_cells = 0;
            for(int c = 0; c < categories; ++c)
                finest_cells += counts[c];
            line << " finest_cells=" << finest_cells;
            for(int c = 0; c < categories; ++c)
                line << " finest_" << names[c] << "=" << counts[c];
            for(int c = 0; c < categories; ++c)
                line << " all_" << names[c] << "=" << counts[categories + c];
            line << " all_shortened_interval=" << counts[2 * categories];
            line << " example_id=" << example_id
                 << " example_old_bin=" << example_record[1]
                 << " example_new_bin=" << example_record[2]
                 << " example_physics_bin=" << example_record[3]
                 << " example_category=" << (example_record[4] >= 0 ?
                        names[static_cast<int>(example_record[4])] : "none")
                 << " example_limit=" << example.value
                 << " example_density=" << example_record[5]
                 << " example_speed=" << example_record[6];
            std::cout << line.str() << std::endl;
        }
    }
    if(full_source_sweep.performed)
    {
        this->individualScheduler->recordFullSourceSweep(
            context.event_tick);
        if(this->rank == 0)
        {
            double const minimum_dt = std::ldexp(
                context.time_quantum,
                static_cast<int>(
                    full_source_sweep.minimum_occupied_bin));
            std::ostringstream message;
            message << std::setprecision(17)
                    << "RICH_FULL_SOURCE_SWEEP"
                    << " event_tick=" << context.event_tick
                    << " time=" << context.event_time
                    << " min_bin="
                    << static_cast<unsigned>(
                        full_source_sweep.minimum_occupied_bin)
                    << " min_dt=" << minimum_dt
                    << " interval_min_steps="
                    << this->individualScheduler->options().
                        full_source_sweep_interval_minimum_steps
                    << " elapsed_min_steps="
                    << full_source_sweep.elapsed_minimum_steps
                    << " sources="
                    << full_source_sweep.global_source_count
                    << " passive_targets="
                    << full_source_sweep.global_target_count
                    << " mpi_rounds="
                    << full_source_sweep.mpi_exchange_rounds
                    << " mpi_source_records="
                    << full_source_sweep.mpi_source_records
                    << " mpi_peak_payload_bytes="
                    << full_source_sweep.
                        maximum_exchange_buffer_bytes
                    << " seconds_max="
                    << full_source_sweep.seconds_max;
            std::cout << message.str() << std::endl;
        }
    }
    this->individualSynchronizedEventRequested = false;
    this->tracker.time = context.event_time;
    bool individual_amr_applied = false;
    std::uint64_t const local_amr_cells_before =
        static_cast<std::uint64_t>(this->cells.size());
    std::uint64_t local_amr_added_cells = 0;
    std::uint64_t local_amr_removed_cells = 0;
    if(this->individualAMR)
    {
        const auto amr_start = std::chrono::high_resolution_clock::now();
        double const amr_mpi_start = mpi_wait_profiler::Seconds();
        // AMR rebuilds its mesh from these event-time generators; the
        // context's gravity source points are cell centroids.  Collective.
        context.generator_points = this->CommittedGeneratorPoints();
        IndividualAMRChangeSet const changes = this->individualAMR(context);
        std::vector<Vector3D>().swap(context.generator_points);
        local_mesh_build_timing.seconds += changes.mesh_build_timing.seconds;
        local_mesh_build_timing.builds += changes.mesh_build_timing.builds;
        local_mesh_build_timing.full_builds +=
            changes.mesh_build_timing.full_builds;
        local_amr_added_cells = static_cast<std::uint64_t>(
            changes.child_parent_ids.size());
        local_amr_removed_cells = static_cast<std::uint64_t>(
            changes.removed_cell_ids.size());
        this->individualScheduler->applyAMRChangeSet(this->cells, changes,
                                                      &this->tess);
        individual_amr_applied = !changes.empty();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - amr_start).count();
        this->lastPhysicsTimes["individual-amr"] = elapsed;
        this->lastLocalPhysicsTimes["individual-amr"] = elapsed;
        this->lastLocalPhysicsMpiTimes["individual-amr"] =
            mpi_wait_profiler::Seconds() - amr_mpi_start;
        if(detailed_runtime_log && this->rank == 0)
            std::cout << "Individual AMR time: " << elapsed << std::endl;
    }
#ifdef RICH_MPI
    // AMR change sets are rank-local, but the rebalance decision below enters
    // collectives.  Make the "AMR happened" predicate collective first so a
    // rank with no local changes cannot skip a rebalance requested by another
    // rank and deadlock the job.
    int any_individual_amr_applied = individual_amr_applied ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &any_individual_amr_applied, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    individual_amr_applied = any_individual_amr_applied != 0;

#endif

    if(individual_amr_applied)
    {
        // A cached acceleration is the acceleration at the start of its
        // cell's current interval.  For a cell in mid-interval that is a past
        // state, which this AMR pass cannot change.  For a cell whose interval
        // opens now (active in this event, and every AMR child) it is the
        // current state, which AMR just reshaped: children, parents, merge
        // recipients and refinement donors on any rank, the latter not named
        // by the change set.  Refresh those caches on the rebuilt mesh (one
        // solve; it also absorbs this event's sink removal), else drop them.
        auto const cache_refresh_start =
            std::chrono::high_resolution_clock::now();
        this->refreshCurrentAccelerationCaches();
        double cache_refresh_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() -
            cache_refresh_start).count();
        this->lastLocalPhysicsTimes["individual-amr-cache-refresh"] =
            cache_refresh_seconds;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &cache_refresh_seconds, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
#endif
        this->lastPhysicsTimes["individual-amr-cache-refresh"] =
            cache_refresh_seconds;
        auto const topology_release_start =
            std::chrono::high_resolution_clock::now();
        for(const std::shared_ptr<PhysicsStep>& physicsStep : this->physics)
            physicsStep->afterIndividualAMR();
        double const local_topology_release_seconds =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() -
                topology_release_start).count();
        double maximum_topology_release_seconds =
            local_topology_release_seconds;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &maximum_topology_release_seconds, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        this->lastLocalPhysicsTimes["individual-topology-release"] +=
            local_topology_release_seconds;
        this->lastPhysicsTimes["individual-topology-release"] =
            maximum_topology_release_seconds;
    }

#ifdef RICH_MPI
    bool const forced_balance = this->forceRebalanceSteps > 0 &&
        this->tracker.getCycle() < this->forceRebalanceSteps &&
        this->lastRebalanceCycle != this->tracker.getCycle();
    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    unsigned long long total_owned_cells = local_owned_cells;
    unsigned long long maximum_owned_cells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &total_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    double const mean_owned_cells = static_cast<double>(total_owned_cells) /
        static_cast<double>(this->size);
    double const owned_cell_skew = mean_owned_cells > 0 ?
        static_cast<double>(maximum_owned_cells) / mean_owned_cells : 1.0;

    double local_event_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - stepWallStart).count();
    double maximum_event_seconds = local_event_seconds;
    double total_event_seconds = local_event_seconds;
    MPI_Allreduce(MPI_IN_PLACE, &maximum_event_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_event_seconds, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    double const mean_event_seconds = total_event_seconds /
        static_cast<double>(this->size);
    double const predicted_horizon_savings =
        static_cast<double>(balance_options.cooldown_events) *
        std::max(0.0, maximum_event_seconds - mean_event_seconds);
    bool const amortized = this->lastIndividualRebalanceSeconds <= 0 ||
        predicted_horizon_savings > balance_options.amortization_factor *
            this->lastIndividualRebalanceSeconds;
    bool const cooldown_complete =
        this->lastIndividualBalanceCheckCycle ==
            std::numeric_limits<size_t>::max() ||
        (this->tracker.getCycle() >= this->lastIndividualBalanceCheckCycle &&
         this->tracker.getCycle() - this->lastIndividualBalanceCheckCycle >=
             balance_options.cooldown_events);
    bool const immediate_after_amr = individual_amr_applied &&
        owned_cell_skew > balance_options.immediate_amr_threshold;
    bool const automatic_balance = !active_hilbert_balance &&
        balance_options.enabled &&
        owned_cell_skew > balance_options.threshold &&
        (immediate_after_amr || (cooldown_complete && amortized));
    bool const request_balance = !active_hilbert_balance &&
        (forced_balance || automatic_balance);

    if(balance_options.trace && this->rank == 0)
        std::clog << std::setprecision(17)
                  << "INDIVIDUAL_LOAD_BALANCE_DECISION"
                  << " cycle=" << this->tracker.getCycle()
                  << " enabled=" << (balance_options.enabled ? 1 : 0)
                  << " active_hilbert_cache="
                  << (active_hilbert_balance ? 1 : 0)
                  << " forced=" << (forced_balance ? 1 : 0)
                  << " amr=" << (individual_amr_applied ? 1 : 0)
                  << " owned_max_mean=" << owned_cell_skew
                  << " threshold=" << balance_options.threshold
                  << " cooldown_complete=" << (cooldown_complete ? 1 : 0)
                  << " predicted_savings_seconds="
                  << predicted_horizon_savings
                  << " previous_migration_seconds="
                  << this->lastIndividualRebalanceSeconds
                  << " requested=" << (request_balance ? 1 : 0)
                  << " ownership_epoch=" << this->individualOwnershipEpoch
                  << std::endl;

    if(request_balance)
    {
        std::shared_ptr<PhysicsStep> const balance_step =
            this->findIndividualBalanceStep();
        if(!balance_step)
            throw std::logic_error(
                "Individual load balancing has no supporting physics step");
        // Wall and MPI time over the same interval: the whole call, including
        // the reductions it makes after its own internal timer stops.
        auto const balance_start = std::chrono::high_resolution_clock::now();
        double const balance_mpi_start = mpi_wait_profiler::Seconds();
        IndividualRebalanceResult const result =
            this->rebalanceCommittedIndividualState(
                balance_step, forced_balance, balance_options.threshold);
        double const balance_mpi_seconds =
            mpi_wait_profiler::Seconds() - balance_mpi_start;
        double const balance_local_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - balance_start).count();
        this->lastPhysicsTimes["individual-load-balance"] =
            result.maximumSeconds;
        this->lastLocalPhysicsTimes["individual-load-balance"] =
            balance_local_seconds;
        this->lastLocalPhysicsMpiTimes["individual-load-balance"] =
            balance_mpi_seconds;
        if(detailed_runtime_log && this->rank == 0)
            std::cout << "Individual load balance time: "
                      << result.maximumSeconds
                      << " s, applied=" << (result.applied ? 1 : 0)
                      << ", weight max/mean=" << result.weightSkew
                      << ", migrated cells=" << result.migratedCells
                      << ", ownership epoch="
                      << this->individualOwnershipEpoch << std::endl;
    }

#endif
    // Every endpoint of this event is set (commit, AMR, load balance), so the
    // conserved-change wakes can join the next scheduled event.
    std::uint64_t change_wake_next_tick = 0;
    {
        auto const finalize_start = std::chrono::high_resolution_clock::now();
        IndividualTimeStepScheduler::ChangeWakeFinalization const wakes =
            this->individualScheduler->finalizeChangeWakes(cadence_trace);
        change_wake_next_tick = wakes.next_tick_after;
        individual_commit_seconds += seconds_since(finalize_start);
        if(cadence_trace && this->rank == 0)
        {
            std::uint64_t const spacing = wakes.next_tick_after - event_tick;
            unsigned on_grid_bin = 0;
            while(on_grid_bin < 63 && wakes.next_tick_after % (std::uint64_t(2) << on_grid_bin) == 0)
                ++on_grid_bin;
            std::cout << "INDIVIDUAL_EVENT_DEADLINE event_tick=" << event_tick
                      << " next_event_tick=" << wakes.next_tick_after << " spacing_ticks=" << spacing
                      << " on_grid_bin=" << on_grid_bin << " finest_bin="
                      << static_cast<unsigned>(this->individualScheduler->minimumOccupiedBin())
                      << " change_wakes=" << wakes.applied << " change_wakes_shortened=" << wakes.shortened
                      << " merge_wakes=" << wakes.merge_wakes
                      << " lowered=" << wakes.lowered[0] << "," << wakes.lowered[1] << "," << wakes.lowered[2]
                      << "," << wakes.lowered[3]
                      << " lowered_overdue=" << wakes.lowered_overdue[0] << "," << wakes.lowered_overdue[1]
                      << "," << wakes.lowered_overdue[2] << "," << wakes.lowered_overdue[3]
                      << " lowered_worst_ratio=" << wakes.lowered_worst_ratio[0]
                      << "," << wakes.lowered_worst_ratio[1] << "," << wakes.lowered_worst_ratio[2]
                      << "," << wakes.lowered_worst_ratio[3]
                      << " largest_ratio=" << wakes.largest_ratio << std::endl;
        }
    }
    std::uint64_t next_tick = this->individualScheduler->nextEventTick();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &next_tick, 1, MPI_UINT64_T,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    // Both values are collective, so every rank throws together.  The
    // reference rule (spacing_ticks) may create events, so it is not checked.
    if(IndividualTimeStepScheduler::changeWakesJoinNextEvent() && next_tick != change_wake_next_tick)
        throw std::logic_error("Conserved-change wakes changed the next individual event");
    const double next_dt = next_tick == std::numeric_limits<std::uint64_t>::max()
        ? std::numeric_limits<double>::infinity()
        : this->individualScheduler->timeQuantum() *
          static_cast<double>(next_tick - this->individualScheduler->currentTick());
    this->tsc->SetTimeStep(next_dt);
    double const local_step_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - stepWallStart).count();
    this->wallclockTime += local_step_seconds;
    // Per-rank residual: step wall minus every exclusive section above and
    // every top-level physics/scheduler phase (sub-phase counters such as
    // radiation-driver are nested inside their phase and not subtracted).
    double local_other_seconds = local_step_seconds -
        (individual_sync_seconds + individual_prepare_seconds +
         individual_closure_seconds + individual_suggest_seconds +
         individual_wake_seconds + individual_sweep_seconds +
         individual_commit_seconds);
    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
        local_other_seconds -= LocalRuntimePhase(
            this->lastLocalPhysicsTimes, physicsStep->getName());
    for(char const* const phase : {"individual-post-physics",
                                    "individual-amr",
                                    "individual-amr-cache-refresh",
                                    "individual-topology-release",
                                    "individual-active-hilbert-balance",
                                    "individual-pre-first-event-load-balance",
                                    "individual-load-balance"})
        local_other_seconds -= LocalRuntimePhase(
            this->lastLocalPhysicsTimes, phase);
    this->lastLocalPhysicsTimes["individual-sync"] = individual_sync_seconds;
    this->lastLocalPhysicsTimes["individual-prepare"] = individual_prepare_seconds;
    this->lastLocalPhysicsTimes["individual-closure"] = individual_closure_seconds;
    this->lastLocalPhysicsTimes["individual-wake"] = individual_wake_seconds;
    this->lastLocalPhysicsTimes["individual-sweep"] = individual_sweep_seconds;
    this->lastLocalPhysicsTimes["individual-commit"] = individual_commit_seconds;
    this->lastLocalPhysicsTimes["individual-other"] = local_other_seconds;

    // The segmented-ownership ledger compares whole events, less the parts a
    // partition does not shape: the AMR pass and the balance decision (whose
    // planning and migration are charged separately).
#ifdef RICH_MPI
    this->lastIndividualEventLedgerLocalSeconds = std::max(0.0, local_step_seconds -
        LocalRuntimePhase(this->lastLocalPhysicsTimes, "individual-amr") -
        LocalRuntimePhase(this->lastLocalPhysicsTimes,
            "individual-active-hilbert-balance"));
    this->reportPhaseBusyTimes("individual", step_cycle, local_step_seconds,
        mpi_wait_profiler::Seconds() - step_mpi_start);
    if(balance_options.trace || detailed_runtime_log) {
        for(auto const& phase : this->lastLocalPhysicsTimes)
            reportIndividualRankDistribution(
                this->rank, this->size, step_cycle, phase.first,
                phase.second, "seconds");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "event-wall",
            local_step_seconds, "seconds");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "individual-wake-signal-sources",
            static_cast<double>(full_source_sweep.local_signal_sources), "count");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "individual-wake-outgoing-records",
            static_cast<double>(full_source_sweep.local_outgoing_records), "count");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "individual-wake-incoming-records",
            static_cast<double>(full_source_sweep.local_incoming_records), "count");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "individual-wake-saturated-sources",
            static_cast<double>(full_source_sweep.local_saturated_sources), "count");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "current-rss",
            currentResidentSetKiB(), "KiB");
        reportIndividualRankDistribution(
            this->rank, this->size, step_cycle, "peak-rss",
            peakResidentSetKiB(), "KiB");
    }
#endif

    RuntimeStepOutput output;
    output.mode = "individual";
    output.cycle = step_cycle;
    output.start_time = context.previous_event_time;
    output.end_time = context.event_time;
    output.event_dt = context.time_quantum * static_cast<double>(
        context.event_tick - context.previous_event_tick);
    output.applied_dt_min = applied_dt_min;
    output.applied_dt_max = applied_dt_max;
    output.next_event_dt = next_dt;
    output.active_cells = global_active_cells;
    output.total_cells = global_total_cells;
    output.active_bins = active_bins;
    output.source_calls = RuntimeMaximum(static_cast<std::uint64_t>(
        local_source_timing.calls));
#ifdef RICH_MPI
    local_mesh_build_timing.seconds +=
        this->individualMeshBuildTiming.seconds;
    local_mesh_build_timing.builds +=
        this->individualMeshBuildTiming.builds;
    local_mesh_build_timing.full_builds +=
        this->individualMeshBuildTiming.full_builds;
#endif
    output.mesh_builds = RuntimeMaximum(static_cast<std::uint64_t>(
        local_mesh_build_timing.builds));
    output.mesh_seconds = RuntimeMaximum(local_mesh_build_timing.seconds);
    output.hydro_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "hydro"));
    output.gravity_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "gravity"));
    output.radiation_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "radiation"));
    output.amr_seconds = RuntimeMaximum(LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "amr") + LocalRuntimePhase(
        this->lastLocalPhysicsTimes, "topology-release"));
    output.source_seconds = RuntimeMaximum(
        local_source_timing.totalSeconds());
    output.source_first_seconds = RuntimeMaximum(
        local_source_timing.first_seconds);
    output.source_second_seconds = RuntimeMaximum(
        local_source_timing.second_seconds);
    output.step_seconds = RuntimeMaximum(local_step_seconds);
    output.sync_seconds = RuntimeMaximum(individual_sync_seconds);
    output.prepare_seconds = RuntimeMaximum(individual_prepare_seconds);
    output.closure_seconds = RuntimeMaximum(individual_closure_seconds);
    output.suggest_seconds = RuntimeMaximum(individual_suggest_seconds);
    output.wake_seconds = RuntimeMaximum(individual_wake_seconds);
    output.sweep_seconds = RuntimeMaximum(individual_sweep_seconds);
    output.commit_seconds = RuntimeMaximum(individual_commit_seconds);
    output.other_seconds = RuntimeMaximum(local_other_seconds);
    output.full_builds = RuntimeMaximum(static_cast<std::uint64_t>(
        local_mesh_build_timing.full_builds));
    WriteRuntimeStep(this->rank, detailed_runtime_log,
                     runtime_color_enabled, output);
    if(individual_amr_applied)
        this->ReportRuntimeAMREvent(
            "individual", step_cycle, context.event_time,
            local_amr_cells_before, local_amr_added_cells,
            local_amr_removed_cells,
            static_cast<std::uint64_t>(this->cells.size()));
    this->tracker.updateCycle();
    this->lastStepAdvance = output.event_dt;
    this->lastStepSecondsMax = output.step_seconds;
}

#ifdef RICH_MPI
void Simulation::storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer<Vector3D>> lb)
{
    this->loads[name] = lb;
}

void Simulation::PresetLoadBalance(const std::string &name)
{
    this->currentLoad = this->tess.GetLoadBalancer();
    this->loads[name] = this->currentLoad;
    this->currentLB = name;
}

void Simulation::setCurrentLoadBalance(const std::string &name)
{
    MeshBuildTiming ignored_mesh_build_timing;
    this->setCurrentLoadBalance(name, ignored_mesh_build_timing);
}

void Simulation::setCurrentLoadBalance(
    const std::string &name, MeshBuildTiming& mesh_build_timing)
{
    if(name.empty())
    {
        return;
    }
    auto it = this->loads.find(name);
    if(it == this->loads.end())
    {
        UniversalError eo("setCurrentLoadBalance: unknown load balance");
        eo.addEntry("Name", name);
        throw eo;
    }

    size_t Ntotal = this->tess.GetPointNo();
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    #endif // RICH_MPI

    if(Ntotal == 0)
    {
        this->tess.PresetLoadBalancer(it->second);
    }
    else
    {
        {
            MeshBuildTimer mesh_build_timer(mesh_build_timing);
            this->tess.SetLoadBalancer(it->second);
        }
        this->buildDataTransfer();
    }

    std::shared_ptr<LoadBalancer<Vector3D>> load = this->tess.GetLoadBalancer();
    this->loads[name] = load;
    this->currentLoad = load;
    this->currentLB = name;

    if(RuntimeLogDetailed() && this->rank == 0)
    {
        std::cout << "Changed load balance" << std::endl;
        //this->currentLoad->printInfo();
    }
}

std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>> Simulation::GetLoads(void) const
{
    return std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>>(this->loads.begin(), this->loads.end());
}
#endif // RICH_MPI
