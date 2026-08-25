#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/time_step_function3D.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
const double pi = 3.141592653589793238462643383279502884;
const double performance_initial_density = 1;
const double performance_initial_pressure = 1e-8;
const std::uint8_t performance_maximum_bin = 4;
const double primitive_probe_updated_tracer = 0.75;

std::string EnvironmentString(char const* const name,
                              std::string const& fallback = std::string())
{
    char const* const value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : value;
}

std::size_t EnvironmentSize(char const* const name, std::size_t fallback)
{
    std::string const value = EnvironmentString(name);
    if(value.empty())
        return fallback;
    char* end = nullptr;
    unsigned long long const parsed = std::strtoull(value.c_str(), &end, 10);
    if(end == value.c_str() || *end != '\0' || parsed == 0)
        throw std::invalid_argument(std::string("Invalid positive integer in ") + name);
    return static_cast<std::size_t>(parsed);
}

std::vector<Vector3D> MakePoints(std::size_t side, bool perturb_points)
{
    std::vector<Vector3D> points;
    points.reserve(side * side * side);
    double const inverse_side = 1.0 / static_cast<double>(side);
    for(std::size_t i = 0; i < side; ++i)
        for(std::size_t j = 0; j < side; ++j)
            for(std::size_t k = 0; k < side; ++k)
            {
                double const seed = static_cast<double>(
                    1 + i + side * (j + side * k));
                double const dx = perturb_points ?
                    0.06 * std::sin(1.61803398875 * seed) : 0;
                double const dy = perturb_points ?
                    0.06 * std::sin(2.41421356237 * seed + 0.7) : 0;
                double const dz = perturb_points ?
                    0.06 * std::sin(3.14159265359 * seed + 1.3) : 0;
                points.push_back(Vector3D(
                    (static_cast<double>(i) + 0.5 + dx) * inverse_side,
                    (static_cast<double>(j) + 0.5 + dy) * inverse_side,
                    (static_cast<double>(k) + 0.5 + dz) * inverse_side));
            }
    return points;
}

std::size_t GridCoordinate(double coordinate, std::size_t side)
{
    double const scaled = coordinate * static_cast<double>(side);
    std::size_t result = scaled > 0 ? static_cast<std::size_t>(scaled) : 0;
    return std::min(result, side - 1);
}

std::size_t DoubledCentralDistance(std::size_t coordinate, std::size_t side)
{
    std::size_t const doubled_cell_center = 2 * coordinate + 1;
    return doubled_cell_center > side ? doubled_cell_center - side :
                                        side - doubled_cell_center;
}

std::uint8_t PerformanceBin(Vector3D const& point, std::size_t side)
{
    std::size_t const dx =
        DoubledCentralDistance(GridCoordinate(point.x, side), side);
    std::size_t const dy =
        DoubledCentralDistance(GridCoordinate(point.y, side), side);
    std::size_t const dz =
        DoubledCentralDistance(GridCoordinate(point.z, side), side);
    std::size_t const distance = std::max(dx, std::max(dy, dz));
    if(distance <= 1)
        return 0;
    if(distance <= 7)
        return 2;
    return performance_maximum_bin;
}

class BenchmarkTimeStep final : public TimeStepFunction3D
{
public:
    BenchmarkTimeStep(
        double global_time_step,
        std::unordered_map<std::size_t, double> individual_time_steps):
        global_time_step_(global_time_step),
        individual_time_steps_(std::move(individual_time_steps))
    {}

    double operator()(Tessellation3D const&,
                      std::vector<ComputationalCell3D> const&,
                      EquationOfState const&,
                      std::vector<Vector3D> const&,
                      double) override
    {
        return global_time_step_;
    }

    void SetTimeStep(double time_step) override
    {
        global_time_step_ = time_step;
    }

    double GetTimeStep(void) const override
    {
        return global_time_step_;
    }

    double SuggestTimeStep(void) const override
    {
        return global_time_step_;
    }

    void SuggestIndividualTimeSteps(
        Tessellation3D const&,
        std::vector<ComputationalCell3D> const& cells,
        EquationOfState const&,
        std::vector<Vector3D> const&,
        double,
        IndividualStepContext const& context,
        std::vector<double>& time_step_limits) const override
    {
        for(std::size_t const index : context.active_indices)
        {
            std::unordered_map<std::size_t, double>::const_iterator const found =
                individual_time_steps_.find(cells.at(index).ID);
            if(found == individual_time_steps_.end())
                throw std::logic_error("Missing benchmark timestep for an active cell");
            time_step_limits.at(index) =
                std::min(time_step_limits.at(index), found->second);
        }
    }

private:
    double global_time_step_;
    std::unordered_map<std::size_t, double> individual_time_steps_;
};

class CountingFmmAcceleration final : public Acceleration3D
{
public:
    CountingFmmAcceleration(FmmGravityOptions options, double gravity_constant):
        acceleration_(options, gravity_constant)
    {}

    void operator()(Tessellation3D const& tess,
                    std::vector<ComputationalCell3D> const& cells,
                    std::vector<Conserved3D> const& fluxes,
                    double time,
                    std::vector<Vector3D>& acceleration) const override
    {
        ++full_calls_;
        source_count_ += tess.GetPointNo();
        std::chrono::steady_clock::time_point const start =
            std::chrono::steady_clock::now();
        acceleration_(tess, cells, fluxes, time, acceleration);
        seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }

    bool SupportsIndividualTargetEvaluation(void) const override
    {
        return true;
    }

    void EvaluateIndividualTargets(
        std::pair<Vector3D, Vector3D> const& bounds,
        std::vector<Vector3D> const& source_points,
        std::vector<double> const& source_masses,
        std::vector<std::uint64_t> const& source_ids,
        std::vector<Vector3D> const& target_points,
        std::vector<ComputationalCell3D> const& target_cells,
        double time,
        std::vector<Vector3D>& acceleration) const override
    {
        ++target_calls_;
        source_count_ += source_points.size();
        target_count_ += target_points.size();
        std::chrono::steady_clock::time_point const start =
            std::chrono::steady_clock::now();
        acceleration_.EvaluateIndividualTargets(
            bounds, source_points, source_masses, source_ids, target_points,
            target_cells, time, acceleration);
        seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }

    std::uint64_t fullCalls(void) const {return full_calls_;}
    std::uint64_t targetCalls(void) const {return target_calls_;}
    std::uint64_t sourceCount(void) const {return source_count_;}
    std::uint64_t targetCount(void) const {return target_count_;}
    double seconds(void) const {return seconds_;}

private:
    FastMultipoleAcceleration3D acceleration_;
    mutable std::uint64_t full_calls_ = 0;
    mutable std::uint64_t target_calls_ = 0;
    mutable std::uint64_t source_count_ = 0;
    mutable std::uint64_t target_count_ = 0;
    mutable double seconds_ = 0;
};

class PrimitiveProbeTracerUpdate final :
    public ConditionExtensiveUpdater3D::Action3D
{
public:
    void operator()(std::vector<Conserved3D> const& /*fluxes*/,
                    Tessellation3D const& /*tess*/,
                    double /*time_step*/,
                    std::vector<ComputationalCell3D> const& /*cells*/,
                    std::vector<Conserved3D>& extensives,
                    std::size_t index,
                    double /*time*/) const override
    {
        extensives.at(index).tracers[1] =
            primitive_probe_updated_tracer * extensives.at(index).mass;
    }
};

class PrimitiveRecoveryProbeAcceleration final : public Acceleration3D
{
public:
    void operator()(Tessellation3D const& /*tess*/,
                    std::vector<ComputationalCell3D> const& /*cells*/,
                    std::vector<Conserved3D> const& /*fluxes*/,
                    double /*time*/,
                    std::vector<Vector3D>& /*acceleration*/) const override
    {
        throw std::logic_error(
            "Primitive recovery probe unexpectedly used full acceleration");
    }

    bool SupportsIndividualTargetEvaluation(void) const override
    {
        return true;
    }

    void EvaluateIndividualTargets(
        std::pair<Vector3D, Vector3D> const& /*bounds*/,
        std::vector<Vector3D> const& /*source_points*/,
        std::vector<double> const& /*source_masses*/,
        std::vector<std::uint64_t> const& /*source_ids*/,
        std::vector<Vector3D> const& target_points,
        std::vector<ComputationalCell3D> const& target_cells,
        double /*time*/,
        std::vector<Vector3D>& acceleration) const override
    {
        if(target_calls_ >= observed_tracers_.size())
            throw std::logic_error(
                "Primitive recovery probe received too many target evaluations");
        if(target_cells.size() != target_points.size() || target_cells.empty())
            throw std::logic_error(
                "Primitive recovery probe received inconsistent target cells");
        double const expected = target_calls_ == 0 ?
            0.0 : primitive_probe_updated_tracer;
        double const tolerance = 64 * std::numeric_limits<double>::epsilon();
        for(ComputationalCell3D const& cell : target_cells)
            if(std::abs(cell.tracers[1] - expected) > tolerance)
                throw std::runtime_error(
                    "Second-half source did not receive recovered tracer primitives");
        observed_tracers_[target_calls_] = target_cells.front().tracers[1];
        ++target_calls_;
        acceleration.assign(target_points.size(), Vector3D());
    }

    std::size_t targetCalls(void) const {return target_calls_;}
    double firstObservedTracer(void) const {return observed_tracers_[0];}
    double secondObservedTracer(void) const {return observed_tracers_[1];}

private:
    mutable std::size_t target_calls_ = 0;
    mutable std::array<double, 2> observed_tracers_ = {{-1, -1}};
};

struct ReducedMetrics
{
    std::uint64_t cells = 0;
    std::uint64_t minimum_owned_cells = 0;
    std::uint64_t maximum_owned_cells = 0;
    std::uint64_t zero_owned_ranks = 0;
    std::uint64_t active_updates = 0;
    std::uint64_t events = 0;
    std::uint64_t full_calls = 0;
    std::uint64_t target_calls = 0;
    std::uint64_t source_count = 0;
    std::uint64_t target_count = 0;
    std::array<std::uint64_t, 7> bin_counts;
    double wall_seconds = 0;
    double fmm_seconds = 0;
    double initial_mass = 0;
    double final_mass = 0;
    double initial_energy = 0;
    double final_energy = 0;

    ReducedMetrics()
    {
        bin_counts.fill(0);
    }
};

struct FmmParityProbe
{
    std::uint64_t targets = 0;
    double maximum_absolute_error = 0;
    double maximum_normalized_error = 0;
};

double SumMass(std::vector<Conserved3D> const& extensives,
               std::size_t owned_cells)
{
    double result = 0;
    for(std::size_t index = 0; index < owned_cells; ++index)
        result += extensives.at(index).mass;
    return result;
}

double SumEnergy(std::vector<Conserved3D> const& extensives,
                 std::size_t owned_cells)
{
    double result = 0;
    for(std::size_t index = 0; index < owned_cells; ++index)
        result += extensives.at(index).energy;
    return result;
}

FmmParityProbe ProbeFmmTargetParity(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    FmmGravityOptions const& options,
    double gravity_constant)
{
    std::size_t const owned_cells = tess.GetPointNo();
    if(cells.size() < owned_cells)
        throw std::logic_error("FMM parity probe cell array is smaller than the owned mesh");

    std::vector<Vector3D> source_points;
    std::vector<double> source_masses;
    std::vector<std::uint64_t> source_ids;
    std::vector<Vector3D> target_points;
    std::vector<ComputationalCell3D> target_cells;
    source_points.reserve(owned_cells);
    source_masses.reserve(owned_cells);
    source_ids.reserve(owned_cells);
    target_points.reserve(owned_cells);
    target_cells.reserve(owned_cells);
    for(std::size_t index = 0; index < owned_cells; ++index)
    {
        source_points.push_back(tess.GetCellCM(index));
        source_masses.push_back(cells[index].density * tess.GetVolume(index));
        source_ids.push_back(static_cast<std::uint64_t>(cells[index].ID));
        target_points.push_back(tess.GetCellCM(index));
        target_cells.push_back(cells[index]);
    }

    FastMultipoleAcceleration3D full_acceleration_provider(options, gravity_constant);
    FastMultipoleAcceleration3D target_acceleration_provider(options, gravity_constant);
    std::vector<Conserved3D> unused_extensives;
    std::vector<Vector3D> full_accelerations;
    std::vector<Vector3D> target_accelerations;
    full_acceleration_provider(
        tess, cells, unused_extensives, 0, full_accelerations);

    // Seed the retained target solver in canonical order, then reorder every
    // local source. Persistent cell IDs must keep the second solve identical.
    target_acceleration_provider.EvaluateIndividualTargets(
        tess.GetBoxCoordinates(), source_points, source_masses, source_ids,
        target_points, target_cells, 0, target_accelerations);
    std::reverse(source_points.begin(), source_points.end());
    std::reverse(source_masses.begin(), source_masses.end());
    std::reverse(source_ids.begin(), source_ids.end());
    target_acceleration_provider.EvaluateIndividualTargets(
        tess.GetBoxCoordinates(), source_points, source_masses, source_ids,
        target_points, target_cells, 0, target_accelerations);

    bool const local_sizes_match =
        full_accelerations.size() == owned_cells &&
        target_accelerations.size() == owned_cells;
#ifdef RICH_MPI
    int sizes_match = local_sizes_match ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &sizes_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(sizes_match == 0)
        throw std::runtime_error("FMM parity probe returned the wrong target count");
#else
    if(!local_sizes_match)
        throw std::runtime_error("FMM parity probe returned the wrong target count");
#endif

    FmmParityProbe result;
    result.targets = owned_cells;
    for(std::size_t index = 0; index < owned_cells; ++index)
    {
        Vector3D const difference =
            full_accelerations[index] - target_accelerations[index];
        double const absolute_error = fastabs(difference);
        double const scale = std::max(
            fastabs(full_accelerations[index]),
            fastabs(target_accelerations[index]));
        result.maximum_absolute_error =
            std::max(result.maximum_absolute_error, absolute_error);
        result.maximum_normalized_error = std::max(
            result.maximum_normalized_error,
            absolute_error / std::max(scale, 1e-300));
    }
#ifdef RICH_MPI
    unsigned long long local_targets =
        static_cast<unsigned long long>(result.targets);
    unsigned long long global_targets = 0;
    MPI_Allreduce(&local_targets, &global_targets, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.targets = global_targets;
    double global_error = 0;
    MPI_Allreduce(&result.maximum_absolute_error, &global_error, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    result.maximum_absolute_error = global_error;
    MPI_Allreduce(&result.maximum_normalized_error, &global_error, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    result.maximum_normalized_error = global_error;
#endif
    return result;
}

ReducedMetrics ReduceMetrics(
    std::uint64_t local_cells,
    std::uint64_t local_active_updates,
    std::uint64_t local_events,
    CountingFmmAcceleration const& acceleration,
    std::array<std::uint64_t, 7> const& local_bin_counts,
    double local_wall_seconds,
    double local_initial_mass,
    double local_final_mass,
    double local_initial_energy,
    double local_final_energy)
{
    ReducedMetrics result;
#ifdef RICH_MPI
    unsigned long long local_value = 0;
    unsigned long long global_value = 0;

    local_value = static_cast<unsigned long long>(local_cells);
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.cells = global_value;
    local_value = static_cast<unsigned long long>(local_cells);
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
    result.minimum_owned_cells = global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    result.maximum_owned_cells = global_value;
    local_value = local_cells == 0 ? 1 : 0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.zero_owned_ranks = global_value;
    local_value = static_cast<unsigned long long>(local_active_updates);
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.active_updates = global_value;
    local_value = static_cast<unsigned long long>(local_events);
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    result.events = global_value;
    local_value = static_cast<unsigned long long>(acceleration.fullCalls());
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    result.full_calls = global_value;
    local_value = static_cast<unsigned long long>(acceleration.targetCalls());
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    result.target_calls = global_value;
    local_value = static_cast<unsigned long long>(acceleration.sourceCount());
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.source_count = global_value;
    local_value = static_cast<unsigned long long>(acceleration.targetCount());
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    result.target_count = global_value;

    std::array<unsigned long long, 7> local_bins;
    std::array<unsigned long long, 7> global_bins;
    for(std::size_t index = 0; index < local_bins.size(); ++index)
        local_bins[index] = static_cast<unsigned long long>(local_bin_counts[index]);
    MPI_Allreduce(local_bins.data(), global_bins.data(),
                  static_cast<int>(local_bins.size()), MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    for(std::size_t index = 0; index < result.bin_counts.size(); ++index)
        result.bin_counts[index] = global_bins[index];

    MPI_Allreduce(&local_wall_seconds, &result.wall_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    double const local_fmm_seconds = acceleration.seconds();
    MPI_Allreduce(&local_fmm_seconds, &result.fmm_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_initial_mass, &result.initial_mass, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_final_mass, &result.final_mass, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_initial_energy, &result.initial_energy, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_final_energy, &result.final_energy, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
#else
    result.cells = local_cells;
    result.minimum_owned_cells = local_cells;
    result.maximum_owned_cells = local_cells;
    result.zero_owned_ranks = local_cells == 0 ? 1 : 0;
    result.active_updates = local_active_updates;
    result.events = local_events;
    result.full_calls = acceleration.fullCalls();
    result.target_calls = acceleration.targetCalls();
    result.source_count = acceleration.sourceCount();
    result.target_count = acceleration.targetCount();
    result.bin_counts = local_bin_counts;
    result.wall_seconds = local_wall_seconds;
    result.fmm_seconds = acceleration.seconds();
    result.initial_mass = local_initial_mass;
    result.final_mass = local_final_mass;
    result.initial_energy = local_initial_energy;
    result.final_energy = local_final_energy;
#endif
    return result;
}

void WriteState(std::string const& output_directory,
                std::string const& stem,
                int rank,
                Tessellation3D const& tess,
                std::vector<ComputationalCell3D> const& cells,
                std::vector<Conserved3D> const& extensives)
{
    std::size_t const owned_cells = tess.GetPointNo();
    if(cells.size() < owned_cells || extensives.size() < owned_cells)
        throw std::logic_error("Benchmark state arrays are smaller than the owned mesh");
    ActiveMeshView const view(tess, owned_cells);
    std::ostringstream name;
    name << output_directory << "/" << stem << "_rank_" << std::setw(5)
         << std::setfill('0') << rank << ".tsv";
    std::ofstream output(name.str().c_str());
    if(!output)
        throw std::runtime_error("Cannot write final benchmark state");
    output << "id\tx\ty\tz\tdensity\tpressure\tspecific_internal_energy"
              "\tvx\tvy\tvz\tmass\tmomx\tmomy\tmomz\ttotal_energy"
              "\textensive_internal_energy\tvolume\n";
    output << std::setprecision(17);
    for(std::size_t global = 0; global < owned_cells; ++global)
    {
        if(!view.containsGlobal(global))
            throw std::logic_error("Final benchmark mesh misses a canonical cell");
        std::size_t const local = view.globalToLocal(global);
        Vector3D const& point = tess.GetMeshPoint(local);
        ComputationalCell3D const& cell = cells[global];
        Conserved3D const& extensive = extensives[global];
        output << cell.ID << '\t' << point.x << '\t' << point.y << '\t'
               << point.z << '\t' << cell.density << '\t' << cell.pressure
               << '\t' << cell.internal_energy << '\t' << cell.velocity.x
               << '\t' << cell.velocity.y << '\t' << cell.velocity.z << '\t'
               << extensive.mass << '\t' << extensive.momentum.x << '\t'
               << extensive.momentum.y << '\t' << extensive.momentum.z << '\t'
               << extensive.energy << '\t' << extensive.internal_energy << '\t'
               << tess.GetVolume(local) << '\n';
    }
}

void WriteMetrics(std::string const& output_directory,
                  std::string const& scenario,
                  std::string const& mode,
                  int world_size,
                  std::size_t grid_side,
                  std::uint64_t final_tick,
                  double time_quantum,
                  double gravity_constant,
                  bool final_synchronized,
                  ReducedMetrics const& metrics,
                  FmmParityProbe const& fmm_parity)
{
    std::ofstream output((output_directory + "/metrics.txt").c_str());
    if(!output)
        throw std::runtime_error("Cannot write benchmark metrics");
    output << std::setprecision(17);
    output << "scenario " << scenario << '\n';
    output << "mode " << mode << '\n';
    output << "world_size " << world_size << '\n';
    output << "grid_side " << grid_side << '\n';
    output << "global_cells " << metrics.cells << '\n';
    output << "minimum_owned_cells " << metrics.minimum_owned_cells << '\n';
    output << "maximum_owned_cells " << metrics.maximum_owned_cells << '\n';
    output << "zero_owned_ranks " << metrics.zero_owned_ranks << '\n';
    output << "final_tick " << final_tick << '\n';
    output << "time_quantum " << time_quantum << '\n';
    output << "final_time " << time_quantum * static_cast<double>(final_tick) << '\n';
    output << "gravity_constant " << gravity_constant << '\n';
    output << "initial_state_profile "
           << (scenario == "performance" ? "cold_uniform" : "warm_nonuniform")
           << '\n';
    output << "point_layout "
           << (scenario == "performance" ? "cartesian" : "perturbed_cartesian")
           << '\n';
    output << "point_motion "
           << (scenario == "performance" ? "lagrangian" : "round_cells")
           << '\n';
    output << "timestep_layout "
           << (scenario == "performance" ? "localized_nested_cubes" :
                                             "uniform_bin_0")
           << '\n';
    output << "prescribed_maximum_bin "
           << (scenario == "performance" ? performance_maximum_bin : 0) << '\n';
    output << "prescribed_maximum_time_step "
           << time_quantum * static_cast<double>(
                  std::uint64_t(1) <<
                  (scenario == "performance" ? performance_maximum_bin : 0))
           << '\n';
    output << "performance_initial_density ";
    if(scenario == "performance")
        output << performance_initial_density;
    else
        output << "not_applicable";
    output << '\n';
    output << "performance_initial_pressure ";
    if(scenario == "performance")
        output << performance_initial_pressure;
    else
        output << "not_applicable";
    output << '\n';
    output << "events " << metrics.events << '\n';
    output << "active_updates " << metrics.active_updates << '\n';
    double const all_active_updates = static_cast<double>(metrics.cells) *
                                      static_cast<double>(metrics.events);
    output << "active_work_fraction "
           << (all_active_updates > 0 ? metrics.active_updates / all_active_updates : 0)
           << '\n';
    output << "wall_seconds " << metrics.wall_seconds << '\n';
    output << "fmm_seconds " << metrics.fmm_seconds << '\n';
    output << "fmm_full_calls " << metrics.full_calls << '\n';
    output << "fmm_target_calls " << metrics.target_calls << '\n';
    output << "fmm_source_count " << metrics.source_count << '\n';
    output << "fmm_target_count " << metrics.target_count << '\n';
    output << "fmm_probe_targets " << fmm_parity.targets << '\n';
    output << "fmm_probe_max_absolute_error "
           << fmm_parity.maximum_absolute_error << '\n';
    output << "fmm_probe_max_normalized_error "
           << fmm_parity.maximum_normalized_error << '\n';
    output << "initial_mass " << metrics.initial_mass << '\n';
    output << "final_mass " << metrics.final_mass << '\n';
    output << "initial_energy " << metrics.initial_energy << '\n';
    output << "final_energy " << metrics.final_energy << '\n';
    output << "final_synchronized " << (final_synchronized ? 1 : 0) << '\n';
    for(std::size_t bin = 0; bin < metrics.bin_counts.size(); ++bin)
        output << "initial_bin_" << bin << " " << metrics.bin_counts[bin] << '\n';
}

int RunPrimitiveRecoveryProbe(int rank,
                              int world_size,
                              std::string const& mode,
                              std::string const& output_directory)
{
    if(world_size != 1 || mode != "individual_sync")
        throw std::invalid_argument(
            "Primitive recovery probe requires one rank in individual_sync mode");
    if(EnvironmentSize("RICH_FMM_ITS_GRID_SIDE", 2) != 2 ||
       EnvironmentSize("RICH_FMM_ITS_FINAL_TICK", 1) != 1)
        throw std::invalid_argument(
            "Primitive recovery probe requires side=2 and final_tick=1");

    double const time_quantum = 2e-4;
    Vector3D const lower(0, 0, 0);
    Vector3D const upper(1, 1, 1);
    Voronoi3D tess(lower, upper);
    std::vector<Vector3D> points;
    if(rank == 0)
        points = MakePoints(2, false);
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif

    ComputationalCell3D::tracerNames = {"unused", "gravity_mask"};
    IdealGas eos(5.0 / 3.0);
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(ComputationalCell3D& cell : cells)
    {
        cell.density = 1;
        cell.pressure = 1;
        cell.velocity = Vector3D();
        cell.tracers[0] = 0;
        cell.tracers[1] = 0;
        cell.internal_energy = eos.dp2e(
            cell.density, cell.pressure, cell.tracers,
            ComputationalCell3D::tracerNames);
    }

    Simulation simulation(tess, cells, eos);
    std::unordered_map<std::size_t, double> individual_time_steps;
    for(ComputationalCell3D const& cell : simulation.getCells())
        individual_time_steps[cell.ID] = time_quantum;
    std::shared_ptr<BenchmarkTimeStep> time_step(
        new BenchmarkTimeStep(time_quantum, individual_time_steps));
    simulation.SetTimeStepFunction(time_step);

    Hllc3D riemann_solver;
    RigidWallGenerator3D ghost;
    LinearGauss3D reconstruction(eos, ghost);
    Lagrangian3D point_motion;
    DefaultCellUpdater cell_updater;
    RigidWallFlux3D rigid_flux(riemann_solver);
    RegularFlux3D regular_flux(riemann_solver);
    IsBoundaryFace3D boundary_face;
    IsBulkFace3D bulk_face;
    std::vector<std::pair<
        ConditionActionFlux1::Condition3D const*,
        ConditionActionFlux1::Action3D const*> > flux_conditions;
    flux_conditions.push_back(std::make_pair(&boundary_face, &rigid_flux));
    flux_conditions.push_back(std::make_pair(&bulk_face, &regular_flux));
    ConditionActionFlux1 flux_calculator(flux_conditions, reconstruction);

    ChooseAll choose_all;
    PrimitiveProbeTracerUpdate tracer_update;
    std::vector<std::pair<
        ConditionExtensiveUpdater3D::Condition3D const*,
        ConditionExtensiveUpdater3D::Action3D const*> > extensive_conditions;
    extensive_conditions.push_back(std::make_pair(&choose_all, &tracer_update));
    ConditionExtensiveUpdater3D extensive_updater(extensive_conditions);
    PrimitiveRecoveryProbeAcceleration acceleration;
    ConservativeForce3D force(acceleration, false);
    HDSim3D hydro(
        tess, simulation.getCells(), simulation.getExtensives(), eos,
        simulation.getTracker(), point_motion, *time_step, flux_calculator,
        cell_updater, extensive_updater, force,
        std::make_pair(ComputationalCell3D::tracerNames,
                       ComputationalCell3D::stickerNames));
    simulation.addPhysics(std::shared_ptr<HydroStep>(
        new HydroStep(hydro, HydroStep::TIMEADVANCE_2)));
    simulation.SetTimeStep(time_quantum);

    IndividualTimeStepOptions options;
    options.time_quantum = time_quantum;
    options.initial_bin = 0;
    options.maximum_bin = 0;
    options.maximum_neighbor_bin_difference = 0;
    options.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
    options.force_synchronized = true;
    simulation.EnableIndividualTimeSteps(options);
    std::vector<CellTimeState> states(simulation.getCells().size());
    for(std::size_t index = 0; index < states.size(); ++index)
    {
        states[index].cell_id = simulation.getCells()[index].ID;
        states[index].begin_tick = 0;
        states[index].end_tick = 1;
        states[index].last_primitive_tick = 0;
        states[index].time_bin = 0;
    }
    simulation.GetIndividualTimeStepScheduler()->restore(
        simulation.getCells(), 0, time_quantum, 0, states);
    simulation.GetIndividualTimeStepScheduler()->clampToTerminalTick(1);
    simulation.step();

    if(acceleration.targetCalls() != 2 ||
       std::abs(simulation.GetTime() - time_quantum) >
           64 * std::numeric_limits<double>::epsilon())
        throw std::runtime_error("Primitive recovery probe did not complete one event");
    std::ofstream receipt(
        (output_directory + "/primitive_recovery_receipt.txt").c_str());
    if(!receipt)
        throw std::runtime_error("Cannot write primitive recovery receipt");
    receipt << std::setprecision(17)
            << "active_targets " << simulation.getCells().size() << '\n'
            << "target_calls " << acceleration.targetCalls() << '\n'
            << "first_target_tracer " << acceleration.firstObservedTracer() << '\n'
            << "second_target_tracer " << acceleration.secondObservedTracer() << '\n'
            << "pass 1\n";
    return 0;
}

int RunBenchmark(int rank, int world_size)
{
    std::string const scenario = EnvironmentString("RICH_FMM_ITS_SCENARIO");
    std::string const mode = EnvironmentString("RICH_FMM_ITS_MODE");
    std::string const output_directory =
        EnvironmentString("RICH_FMM_ITS_OUTPUT");
    if(output_directory.empty())
        throw std::invalid_argument("RICH_FMM_ITS_OUTPUT is required");
    if(scenario == "primitive_recovery_probe")
        return RunPrimitiveRecoveryProbe(rank, world_size, mode, output_directory);
    if(scenario != "parity" && scenario != "performance")
        throw std::invalid_argument(
            "RICH_FMM_ITS_SCENARIO must be parity or performance");
    bool const individual = mode == "individual_sync" ||
                            mode == "individual_sparse";
    if(mode != "global" && !individual)
        throw std::invalid_argument(
            "RICH_FMM_ITS_MODE must be global, individual_sync, or individual_sparse");
    if(scenario == "performance" && mode == "individual_sync")
        throw std::invalid_argument("Benchmark scenario/mode combination is invalid");
    std::size_t const default_side = scenario == "parity" ? 20 : 48;
    std::size_t const grid_side =
        EnvironmentSize("RICH_FMM_ITS_GRID_SIDE", default_side);
    if(scenario == "performance" && (grid_side < 16 || grid_side % 8 != 0))
        throw std::invalid_argument(
            "Performance grid side must be at least 16 and divisible by 8");
    std::uint64_t const default_final_tick = scenario == "parity" ? 16 : 64;
    std::uint64_t const final_tick = static_cast<std::uint64_t>(
        EnvironmentSize("RICH_FMM_ITS_FINAL_TICK", default_final_tick));
    double const time_quantum = scenario == "parity" ? 1e-4 : 2e-4;
    double const gravity_constant = scenario == "parity" ? 0.1 : 1e-6;

    Vector3D const lower(0, 0, 0);
    Vector3D const upper(1, 1, 1);
    Voronoi3D tess(lower, upper);
    std::vector<Vector3D> points;
    if(rank == 0)
        points = MakePoints(grid_side, scenario == "parity");
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif

    IdealGas eos(5.0 / 3.0);
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        Vector3D const& point = tess.GetMeshPoint(index);
        if(scenario == "parity")
        {
            cells[index].density = 1.0 + 0.04 *
                std::sin(2 * pi * point.x) *
                std::sin(2 * pi * point.y) *
                std::sin(2 * pi * point.z);
            cells[index].pressure = 1.0 + 0.03 *
                std::cos(2 * pi * point.x) * std::cos(2 * pi * point.y);
            cells[index].velocity = Vector3D(
                0.01 * std::sin(2 * pi * point.y),
                -0.01 * std::sin(2 * pi * point.x),
                0.005 * std::sin(2 * pi * point.z));
        }
        else
        {
            cells[index].density = performance_initial_density;
            cells[index].pressure = performance_initial_pressure;
            cells[index].velocity = Vector3D();
        }
        cells[index].internal_energy = eos.dp2e(
            cells[index].density, cells[index].pressure,
            cells[index].tracers, ComputationalCell3D::tracerNames);
    }

    Simulation simulation(tess, cells, eos);
    std::size_t const local_owned_cells = tess.GetPointNo();
    std::unordered_map<std::size_t, std::uint8_t> target_bins;
    std::unordered_map<std::size_t, double> individual_time_steps;
    std::array<std::uint64_t, 7> local_bin_counts;
    local_bin_counts.fill(0);
    for(std::size_t index = 0; index < local_owned_cells; ++index)
    {
        std::uint8_t const bin = scenario == "performance" ?
            PerformanceBin(tess.GetMeshPoint(index), grid_side) : 0;
        std::size_t const cell_id = simulation.getCells()[index].ID;
        target_bins[cell_id] = bin;
        individual_time_steps[cell_id] =
            time_quantum * static_cast<double>(std::uint64_t(1) << bin);
        ++local_bin_counts[bin];
    }

    std::shared_ptr<BenchmarkTimeStep> time_step(
        new BenchmarkTimeStep(time_quantum, individual_time_steps));
    simulation.SetTimeStepFunction(time_step);

    Hllc3D riemann_solver;
    RigidWallGenerator3D ghost;
    LinearGauss3D reconstruction(eos, ghost);
    Lagrangian3D base_point_motion;
    RoundCells3D regularized_point_motion(base_point_motion, eos);
    PointMotion3D const& point_motion = scenario == "performance" ?
        static_cast<PointMotion3D const&>(base_point_motion) :
        static_cast<PointMotion3D const&>(regularized_point_motion);
    DefaultCellUpdater cell_updater;
    RigidWallFlux3D rigid_flux(riemann_solver);
    RegularFlux3D regular_flux(riemann_solver);
    IsBoundaryFace3D boundary_face;
    IsBulkFace3D bulk_face;
    std::vector<std::pair<
        ConditionActionFlux1::Condition3D const*,
        ConditionActionFlux1::Action3D const*> > flux_conditions;
    flux_conditions.push_back(std::make_pair(&boundary_face, &rigid_flux));
    flux_conditions.push_back(std::make_pair(&bulk_face, &regular_flux));
    ConditionActionFlux1 flux_calculator(flux_conditions, reconstruction);
    std::vector<std::pair<
        ConditionExtensiveUpdater3D::Condition3D const*,
        ConditionExtensiveUpdater3D::Action3D const*> > extensive_conditions;
    ConditionExtensiveUpdater3D extensive_updater(extensive_conditions);

    FmmGravityOptions fmm_options;
    fmm_options.expansionOrder = 3;
    fmm_options.thetaCritical = 0.9;
    fmm_options.leafCapacity = 128;
    FmmParityProbe const fmm_parity = ProbeFmmTargetParity(
        tess, simulation.getCells(), fmm_options, gravity_constant);
    CountingFmmAcceleration acceleration(fmm_options, gravity_constant);
    ConservativeForce3D force(acceleration);
    HDSim3D hydro(
        tess, simulation.getCells(), simulation.getExtensives(), eos,
        simulation.getTracker(), point_motion, *time_step, flux_calculator,
        cell_updater, extensive_updater, force,
        std::make_pair(ComputationalCell3D::tracerNames,
                       ComputationalCell3D::stickerNames));
    simulation.addPhysics(std::shared_ptr<HydroStep>(
        new HydroStep(hydro, HydroStep::TIMEADVANCE_2)));
    simulation.SetTimeStep(time_quantum);

    WriteState(output_directory, "initial_state", rank, tess,
               simulation.getCells(), simulation.getExtensives());

    std::uint64_t local_active_updates = 0;
    std::uint64_t local_events = 0;
    if(individual)
    {
        IndividualTimeStepOptions options;
        options.time_quantum = time_quantum;
        options.initial_bin = mode == "individual_sparse" ?
            performance_maximum_bin : 0;
        options.maximum_bin = mode == "individual_sparse" ?
            performance_maximum_bin : 0;
        options.maximum_neighbor_bin_difference = 2;
        options.mesh_build_policy = mode == "individual_sparse" ?
            IndividualMeshBuildPolicy::AutoPartial :
            IndividualMeshBuildPolicy::FullReference;
        options.partial_build_fraction = 0.6;
        options.force_synchronized = mode == "individual_sync";
        simulation.EnableIndividualTimeSteps(options);
        if(simulation.getCells().size() != local_owned_cells)
            throw std::logic_error(
                "Individual timestep state is not the canonical owned-cell state");

        std::vector<CellTimeState> states(local_owned_cells);
        for(std::size_t index = 0; index < states.size(); ++index)
        {
            std::size_t const cell_id = simulation.getCells()[index].ID;
            std::uint8_t const bin = target_bins.at(cell_id);
            states[index].cell_id = cell_id;
            states[index].begin_tick = 0;
            states[index].end_tick = std::uint64_t(1) << bin;
            states[index].last_primitive_tick = 0;
            states[index].time_bin = bin;
        }
        simulation.GetIndividualTimeStepScheduler()->restore(
            simulation.getCells(), 0, time_quantum, 0, states);
        simulation.GetIndividualTimeStepScheduler()->clampToTerminalTick(final_tick);
        simulation.SetIndividualPostPhysics(
            [&](IndividualStepContext const& context)
            {
                local_active_updates += context.active_indices.size();
            });
    }

    double const local_initial_mass =
        SumMass(simulation.getExtensives(), local_owned_cells);
    double const local_initial_energy =
        SumEnergy(simulation.getExtensives(), local_owned_cells);
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    double const wall_start = MPI_Wtime();
#else
    std::chrono::steady_clock::time_point const wall_start =
        std::chrono::steady_clock::now();
#endif
    double const final_time = time_quantum * static_cast<double>(final_tick);
    while(simulation.GetTime() < final_time - 0.25 * time_quantum)
    {
        simulation.step();
        ++local_events;
        if(!individual)
            local_active_updates += local_owned_cells;
    }
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    double const local_wall_seconds = MPI_Wtime() - wall_start;
#else
    double const local_wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
#endif
    if(std::abs(simulation.GetTime() - final_time) >
       64 * std::numeric_limits<double>::epsilon() * std::max(1.0, final_time))
        throw std::runtime_error("Benchmark did not stop at the requested endpoint");
    bool const final_synchronized = !individual ||
                                    simulation.IndividualStateSynchronized();
    if(!final_synchronized)
        throw std::runtime_error("Individual benchmark endpoint is not synchronized");
    if(tess.GetPointNo() != local_owned_cells)
        throw std::runtime_error("Benchmark changed rank ownership without load balancing");

    double const local_final_mass =
        SumMass(simulation.getExtensives(), local_owned_cells);
    double const local_final_energy =
        SumEnergy(simulation.getExtensives(), local_owned_cells);
    ReducedMetrics const metrics = ReduceMetrics(
        local_owned_cells, local_active_updates, local_events,
        acceleration, local_bin_counts, local_wall_seconds,
        local_initial_mass, local_final_mass,
        local_initial_energy, local_final_energy);

    WriteState(output_directory, "final_state", rank, tess, simulation.getCells(),
               simulation.getExtensives());
    if(rank == 0)
        WriteMetrics(output_directory, scenario, mode, world_size, grid_side,
                     final_tick, time_quantum, gravity_constant,
                     final_synchronized, metrics, fmm_parity);
    return 0;
}
}

int main(int argc, char** argv)
{
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    (void)argc;
    (void)argv;
#endif
    try
    {
        int const result = RunBenchmark(rank, world_size);
#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return result;
    }
    catch(std::exception const& error)
    {
        std::cerr << "FMM individual-timestep benchmark rank " << rank
                  << " failed: " << error.what() << std::endl;
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
