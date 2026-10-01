#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"

namespace
{
std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b, std::uint64_t c)
{
    std::uint64_t x = a + 0x9e3779b97f4a7c15ULL;
    x ^= b + 0xbf58476d1ce4e5b9ULL + (x << 6U) + (x >> 2U);
    x ^= c + 0x94d049bb133111ebULL + (x << 7U) + (x >> 3U);
    return x;
}

double rel_diff(double value, double reference)
{
    const double den = std::max(std::abs(reference), 1e-30);
    return std::abs(value - reference) / den;
}

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

bool EnvironmentEnabled(char const* name)
{
    char const* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

std::vector<long double> ConservedSums(std::vector<Conserved3D> const& values)
{
    std::vector<long double> sums(9 + MAX_TRACERS + ENERGY_GROUPS_NUM, 0);
    for(Conserved3D const& value : values) {
        sums[0] += value.mass;
        sums[1] += value.momentum.x;
        sums[2] += value.momentum.y;
        sums[3] += value.momentum.z;
        sums[4] += value.energy;
        sums[5] += value.internal_energy;
        sums[6] += value.Erad;
        sums[7] += value.Erad_dt;
        sums[8] += value.Erad_dt_dt;
        for(size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
            sums[9 + tracer] += value.tracers[tracer];
        for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            sums[9 + MAX_TRACERS + group] += value.Eg[group];
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()),
        MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    return sums;
}

double MaximumConservationError(std::vector<long double> const& before,
    std::vector<long double> const& after, size_t* worst_component = nullptr)
{
    double result = 0;
    for(size_t i = 0; i < before.size(); ++i) {
        long double const scale = std::max(static_cast<long double>(1),
            std::abs(before[i]));
        double const error = static_cast<double>(
            std::abs(after[i] - before[i]) / scale);
        if(error > result) {
            result = error;
            if(worst_component != nullptr)
                *worst_component = i;
        }
    }
    return result;
}

bool SameState(CellTimeState const& value, CellTimeState const& reference)
{
    return value.begin_tick == reference.begin_tick &&
        value.end_tick == reference.end_tick &&
        value.last_primitive_tick == reference.last_primitive_tick &&
        value.time_bin == reference.time_bin &&
        value.point_velocity.x == reference.point_velocity.x &&
        value.point_velocity.y == reference.point_velocity.y &&
        value.point_velocity.z == reference.point_velocity.z &&
        value.cached_acceleration.x == reference.cached_acceleration.x &&
        value.cached_acceleration.y == reference.cached_acceleration.y &&
        value.cached_acceleration.z == reference.cached_acceleration.z &&
        value.gravity_half_kick_pending == reference.gravity_half_kick_pending;
}

class RandomRefine3D : public CellsToRefine3D
{
public:
    RandomRefine3D(double probability, int rank):
        probability_(probability),
        rank_(rank),
        iter_(0),
        total_refined_(0) {}

    std::pair<std::vector<size_t>, std::vector<Vector3D> > ToRefine(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const&,
        double) const override
    {
        std::mt19937_64 gen(mix_seed(0xA1B2C3D4ULL, static_cast<std::uint64_t>(rank_), iter_));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<size_t> refine;
        refine.reserve(tess.GetPointNo() / 50);
        const size_t n = tess.GetPointNo();
        for(size_t i = 0; i < n; ++i) {
            if(abs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (0.2 * tess.GetWidth(i))) {
                continue;
            }
            if(dist(gen) < probability_) {
                refine.push_back(i);
            }
        }
        std::sort(refine.begin(), refine.end());
        refine.erase(std::unique(refine.begin(), refine.end()), refine.end());
        total_refined_ += refine.size();
        ++iter_;
        return std::make_pair(refine, std::vector<Vector3D>());
    }

    size_t getTotalRefined() const { return total_refined_; }

private:
    const double probability_;
    const int rank_;
    mutable std::uint64_t iter_;
    mutable size_t total_refined_;
};

class RandomRemove3D : public CellsToRemove3D
{
public:
    RandomRemove3D(double probability, int rank):
        probability_(probability),
        rank_(rank),
        iter_(0),
        total_removed_(0) {}

    std::pair<std::vector<size_t>, std::vector<double> > ToRemove(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const&,
        double) const override
    {
        std::mt19937_64 gen(mix_seed(0xF1E2D3C4ULL, static_cast<std::uint64_t>(rank_), iter_));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<size_t> remove;
        std::vector<double> merits;
        remove.reserve(tess.GetPointNo() / 50);
        merits.reserve(tess.GetPointNo() / 50);
        const size_t n = tess.GetPointNo();
        for(size_t i = 0; i < n; ++i) {
            if(abs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (0.2 * tess.GetWidth(i))) {
                continue;
            }
            if(dist(gen) < probability_) {
                remove.push_back(i);
                merits.push_back(dist(gen));
            }
        }
        if(!remove.empty()) {
            std::vector<std::pair<size_t, double> > zipped;
            zipped.reserve(remove.size());
            for(size_t i = 0; i < remove.size(); ++i) {
                zipped.push_back(std::make_pair(remove[i], merits[i]));
            }
            std::sort(
                zipped.begin(),
                zipped.end(),
                [](std::pair<size_t, double> const& a, std::pair<size_t, double> const& b) {
                    return a.first < b.first;
                });

            remove.clear();
            merits.clear();
            remove.reserve(zipped.size());
            merits.reserve(zipped.size());

            size_t current_index = zipped[0].first;
            double current_merit = zipped[0].second;
            for(size_t i = 1; i < zipped.size(); ++i) {
                if(zipped[i].first == current_index) {
                    current_merit = std::max(current_merit, zipped[i].second);
                }
                else {
                    remove.push_back(current_index);
                    merits.push_back(current_merit);
                    current_index = zipped[i].first;
                    current_merit = zipped[i].second;
                }
            }
            remove.push_back(current_index);
            merits.push_back(current_merit);
        }
        total_removed_ += remove.size();
        ++iter_;
        return std::make_pair(remove, merits);
    }

    size_t getTotalRemoved() const { return total_removed_; }

private:
    const double probability_;
    const int rank_;
    mutable std::uint64_t iter_;
    mutable size_t total_removed_;
};
}

int main()
{
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    try {
        Vector3D ll(-1.0, -1.0, -1.0);
        Vector3D ur(1.0, 1.0, 1.0);
        const size_t default_points = (world_size > 1) ? static_cast<size_t>(2e6) : static_cast<size_t>(1e4);
        const size_t target_points = EnvironmentSize("RICH_TEST_POINT_COUNT", default_points);
        const size_t amr_rounds = EnvironmentSize("RICH_TEST_AMR_ROUNDS", 1);
        const double amr_probability = EnvironmentDouble("RICH_TEST_AMR_PROBABILITY", 5e-2);
        const bool individual_amr = EnvironmentEnabled("RICH_TEST_INDIVIDUAL_AMR");
        const bool partial_before_amr =
            EnvironmentEnabled("RICH_TEST_PARTIAL_MESH_BEFORE_AMR");
        const bool rank_zero_only =
            EnvironmentEnabled("RICH_TEST_ACTIVE_RANK_ZERO_ONLY");
        const bool test_load_balance =
            EnvironmentEnabled("RICH_TEST_INDIVIDUAL_LOAD_BALANCE");
        const double active_fraction = EnvironmentDouble("RICH_TEST_ACTIVE_FRACTION", 0.25);
        if(!(active_fraction > 0 && active_fraction <= 1))
            throw UniversalError("RICH_TEST_ACTIVE_FRACTION must be in (0, 1]");

        std::vector<Vector3D> points;
        if(rank == 0) {
            points = RandRectangular(target_points, ll, ur);
        }
#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
        points = RoundGrid3D(points, ll, ur, 10);

        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        IdealGas eos(5.0 / 3.0);
        const ComputationalCell3D baseline = [&eos]() {
            ComputationalCell3D c;
            c.density = 1.0;
            c.internal_energy = 2.5;
            c.pressure = eos.de2p(c.density, c.internal_energy, c.tracers, ComputationalCell3D::tracerNames);
            c.velocity = Vector3D(0.0, 0.0, 0.0);
            c.Erad = 0.125;
            c.Erad_dt = 0.03125;
            c.Erad_dt_dt = 0.0078125;
            for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                c.Eg[group] = 0.01 * static_cast<double>(group + 1);
            return c;
        }();

        std::vector<ComputationalCell3D> cells(tess.GetPointNo(), baseline);

        Hllc3D rs;
        RigidWallGenerator3D ghost;
        LinearGauss3D interp(eos, ghost);
        std::vector<std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*> > sequence;
        IsBoundaryFace3D is_boundary;
        IsBulkFace3D is_bulk;
        RigidWallFlux3D rigid_flux(rs);
        RegularFlux3D regular_flux(rs);
        sequence.push_back(std::make_pair(&is_boundary, &rigid_flux));
        sequence.push_back(std::make_pair(&is_bulk, &regular_flux));
        ConditionActionFlux1 flux(sequence, interp);
        std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
        ConditionExtensiveUpdater3D eu(eu_sequence);
        DefaultCellUpdater cu;
        ZeroForce3D force;
        CourantFriedrichsLewy tsf(0.3, 1.0, force);
        Lagrangian3D bpm;
        RoundCells3D pm(bpm, eos);
        Simulation simulation(tess, cells, eos);
        HDSim3D sim(
            tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, tsf, flux, cu, eu, force,
            std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

        RandomRefine3D refine(amr_probability, rank);
        RandomRemove3D remove(amr_probability, rank);
        AMR3D amr(eos, refine, remove, interp);

        IndividualTimeStepOptions scheduler_options;
        scheduler_options.initial_bin = 4;
        scheduler_options.maximum_bin = 8;
        IndividualTimeStepScheduler scheduler(scheduler_options);
        if(individual_amr) {
#ifdef RICH_MPI
            simulation.getCells().resize(tess.GetPointNo());
            simulation.getExtensives().resize(tess.GetPointNo());
#endif
            scheduler.initialize(simulation.getCells(), simulation.GetTime(), 1.0);
            for(size_t i = 0; i < scheduler.states().size(); ++i) {
                CellTimeState& state = scheduler.states()[i];
                const double marker = static_cast<double>(state.cell_id + 1);
                state.point_velocity = Vector3D(marker, -2 * marker, 3 * marker);
                state.cached_acceleration = Vector3D(-4 * marker, 5 * marker, -6 * marker);
                state.gravity_half_kick_pending = (state.cell_id % 2) != 0;
            }
        }

        double max_drift_local = 0.0;
        double max_volume_growth = 0.0;
        double max_conservation_error = 0.0;
        size_t worst_conservation_component = 0;
        bool scheduler_remap_valid = true;
        bool load_balance_migration_valid = true;
        size_t migrated_cells_global = 0;
        size_t actual_refined_local = 0;
        size_t actual_removed_local = 0;
        size_t generators_moved_local = 0;
        for(size_t round = 0; round < amr_rounds; ++round) {
            std::unordered_map<size_t, double> old_volumes;
            old_volumes.reserve(tess.GetPointNo());
            for(size_t i = 0; i < tess.GetPointNo(); ++i) {
                old_volumes[simulation.getCells()[i].ID] = tess.GetVolume(i);
            }
            std::vector<long double> const conserved_before =
                ConservedSums(simulation.getExtensives());
            if(individual_amr) {
                IndividualStepContext context;
                const size_t cell_count = simulation.getCells().size();
                const bool rank_has_active_cells = !rank_zero_only || rank == 0;
                const size_t active_count = rank_has_active_cells ?
                    std::max<size_t>(1, static_cast<size_t>(
                        active_fraction * static_cast<double>(cell_count))) : 0;
                context.active_mask.assign(cell_count, 0);
                context.active_indices.reserve(active_count);
                std::unordered_set<size_t> active_ids;
                active_ids.reserve(active_count);
                for(size_t selection = 0; selection < active_count; ++selection) {
                    const size_t index = selection * cell_count / active_count;
                    context.active_indices.push_back(index);
                    context.active_mask[index] = 1;
                    active_ids.insert(simulation.getCells()[index].ID);
                }

                std::unordered_map<size_t, CellTimeState> old_states;
                old_states.reserve(scheduler.states().size());
                for(CellTimeState const& state : scheduler.states())
                    old_states.emplace(state.cell_id, state);

                // As the hydro step and Simulation fill it: committed
                // generators for AMR, cell centroids as gravity sources.  Every
                // cell that survives the pass must keep its generator exactly.
                std::vector<Vector3D> canonical_points = tess.getMeshPoints();
                canonical_points.resize(cell_count);
                context.generator_points = canonical_points;
                context.gravity_source_points.resize(cell_count);
                std::unordered_map<size_t, Vector3D> generator_before;
                generator_before.reserve(cell_count);
                for(size_t i = 0; i < cell_count; ++i) {
                    context.gravity_source_points[i] = tess.GetCellCM(i);
                    generator_before.emplace(simulation.getCells()[i].ID, canonical_points[i]);
                }
                if(partial_before_amr) {
#ifdef RICH_MPI
                    tess.BuildPartiallyParallel(canonical_points,
                        std::vector<double>(canonical_points.size(), 1.0),
                        context.active_indices, true, true);
#else
                    tess.BuildPartially(canonical_points, context.active_indices);
#endif
                }

                IndividualAMRChangeSet const changes = amr.ApplyIndividual(simulation, context);
                {
                    size_t const survivors = std::min(tess.GetPointNo(), simulation.getCells().size());
                    for(size_t i = 0; i < survivors; ++i) {
                        auto const before = generator_before.find(simulation.getCells()[i].ID);
                        Vector3D const& now = tess.GetMeshPoint(i);
                        if(before != generator_before.end() &&
                           !(now.x == before->second.x && now.y == before->second.y &&
                             now.z == before->second.z))
                            ++generators_moved_local;
                    }
                }
                actual_refined_local += changes.child_parent_ids.size();
                actual_removed_local += changes.removed_cell_ids.size();
                for(size_t id : changes.removed_cell_ids)
                    scheduler_remap_valid = scheduler_remap_valid && active_ids.count(id) != 0;
                for(auto const& child_parent : changes.child_parent_ids)
                    scheduler_remap_valid = scheduler_remap_valid &&
                        active_ids.count(child_parent.second) != 0;

                scheduler.applyAMRChangeSet(simulation.getCells(), changes, &tess);
                std::unordered_map<size_t, size_t> child_parents;
                child_parents.reserve(changes.child_parent_ids.size());
                for(auto const& entry : changes.child_parent_ids)
                    child_parents.emplace(entry.first, entry.second);
                for(size_t i = 0; i < simulation.getCells().size(); ++i) {
                    CellTimeState const& state = scheduler.states()[i];
                    const size_t id = simulation.getCells()[i].ID;
                    scheduler_remap_valid = scheduler_remap_valid && state.cell_id == id;
                    auto expected = old_states.find(id);
                    auto parent = child_parents.find(id);
                    if(parent != child_parents.end())
                        expected = old_states.find(parent->second);
                    scheduler_remap_valid = scheduler_remap_valid &&
                        expected != old_states.end() && SameState(state, expected->second);
                }
            }
            else {
                amr(simulation);
            }
            size_t round_worst_component = 0;
            double const round_conservation_error = MaximumConservationError(
                conserved_before, ConservedSums(simulation.getExtensives()),
                &round_worst_component);
            if(round_conservation_error > max_conservation_error) {
                max_conservation_error = round_conservation_error;
                worst_conservation_component = round_worst_component;
            }
            const std::vector<ComputationalCell3D>& current_cells = sim.getCells();
            const Tessellation3D& current_tess = sim.getTessellation();
            const size_t npoints = std::min(current_tess.GetPointNo(), current_cells.size());
            size_t real_local_points = 0;
            double max_density_drift_local = 0.0;
            double max_ie_drift_local = 0.0;
            double max_pressure_drift_local = 0.0;
            double max_vx_drift_local = 0.0;
            double max_vy_drift_local = 0.0;
            double max_vz_drift_local = 0.0;
            for(size_t i = 0; i < npoints; ++i) {
                if(current_tess.IsGhostPoint(i) || current_tess.IsPointOutsideBox(i)) {
                    continue;
                }
                ++real_local_points;
                const ComputationalCell3D& c = current_cells[i];
                auto old_volume = old_volumes.find(c.ID);
                if(old_volume != old_volumes.end()) {
                    max_volume_growth = std::max(max_volume_growth,
                        current_tess.GetVolume(i) / old_volume->second);
                }
                const double density_drift = rel_diff(c.density, baseline.density);
                const double ie_drift = rel_diff(c.internal_energy, baseline.internal_energy);
                const double pressure_drift = rel_diff(c.pressure, baseline.pressure);
                const double vx_drift = std::abs(c.velocity.x);
                const double vy_drift = std::abs(c.velocity.y);
                const double vz_drift = std::abs(c.velocity.z);

                max_density_drift_local = std::max(max_density_drift_local, density_drift);
                max_ie_drift_local = std::max(max_ie_drift_local, ie_drift);
                max_pressure_drift_local = std::max(max_pressure_drift_local, pressure_drift);
                max_vx_drift_local = std::max(max_vx_drift_local, vx_drift);
                max_vy_drift_local = std::max(max_vy_drift_local, vy_drift);
                max_vz_drift_local = std::max(max_vz_drift_local, vz_drift);
                max_drift_local = std::max(max_drift_local, density_drift);
                max_drift_local = std::max(max_drift_local, ie_drift);
                max_drift_local = std::max(max_drift_local, pressure_drift);
                max_drift_local = std::max(max_drift_local, vx_drift);
                max_drift_local = std::max(max_drift_local, vy_drift);
                max_drift_local = std::max(max_drift_local, vz_drift);
            }

            double max_density_drift = max_density_drift_local;
            double max_ie_drift = max_ie_drift_local;
            double max_pressure_drift = max_pressure_drift_local;
            double max_vx_drift = max_vx_drift_local;
            double max_vy_drift = max_vy_drift_local;
            double max_vz_drift = max_vz_drift_local;
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &max_density_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_ie_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_pressure_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vx_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vy_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &max_vz_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
            if(rank == 0) {
                std::cout << "AMR round " << round + 1 << "/" << amr_rounds
                          << " local_points=" << real_local_points
                          << " max_drift_local=" << max_drift_local
                          << " density=" << max_density_drift
                          << " ie=" << max_ie_drift
                          << " pressure=" << max_pressure_drift
                          << " vx=" << max_vx_drift
                          << " vy=" << max_vy_drift
                          << " vz=" << max_vz_drift
                          << std::endl;
            }
        }

#ifdef RICH_MPI
        if(individual_amr && test_load_balance) {
            std::unordered_set<size_t> previous_local_ids;
            previous_local_ids.reserve(simulation.getCells().size());
            std::vector<CellTimeState> migration_states(simulation.getCells().size());
            for(size_t i = 0; i < simulation.getCells().size(); ++i) {
                const size_t id = simulation.getCells()[i].ID;
                previous_local_ids.insert(id);
                const double marker = static_cast<double>(id + 1);
                migration_states[i].cell_id = id;
                migration_states[i].begin_tick = 0;
                migration_states[i].end_tick = 16;
                migration_states[i].last_primitive_tick = 0;
                migration_states[i].time_bin = 4;
                migration_states[i].point_velocity =
                    Vector3D(marker, -2 * marker, 3 * marker);
                migration_states[i].cached_acceleration =
                    Vector3D(-4 * marker, 5 * marker, -6 * marker);
                migration_states[i].gravity_half_kick_pending = (id % 2) != 0;
            }
            simulation.EnableIndividualTimeSteps(scheduler_options);
            simulation.GetIndividualTimeStepScheduler()->restore(
                simulation.getCells(), simulation.GetTime(), 1.0 / 16.0, 0,
                std::move(migration_states));

            std::vector<double> weights(tess.GetPointNo(), rank == 0 ? 100.0 : 1.0);
            tess.Rebalance(weights);
            simulation.buildDataTransfer();

            size_t migrated_cells_local = 0;
            std::vector<CellTimeState> const& migrated_states =
                simulation.GetIndividualTimeStepScheduler()->states();
            load_balance_migration_valid =
                migrated_states.size() == simulation.getCells().size();
            for(size_t i = 0; i < simulation.getCells().size(); ++i) {
                const size_t id = simulation.getCells()[i].ID;
                if(previous_local_ids.count(id) == 0)
                    ++migrated_cells_local;
                const double marker = static_cast<double>(id + 1);
                CellTimeState const& state = migrated_states[i];
                load_balance_migration_valid = load_balance_migration_valid &&
                    state.cell_id == id && state.begin_tick == 0 &&
                    state.end_tick == 16 && state.last_primitive_tick == 0 &&
                    state.time_bin == 4 &&
                    state.point_velocity.x == marker &&
                    state.point_velocity.y == -2 * marker &&
                    state.point_velocity.z == 3 * marker &&
                    state.cached_acceleration.x == -4 * marker &&
                    state.cached_acceleration.y == 5 * marker &&
                    state.cached_acceleration.z == -6 * marker &&
                    state.gravity_half_kick_pending == ((id % 2) != 0);
            }
            migrated_cells_global = migrated_cells_local;
            MPI_Allreduce(MPI_IN_PLACE, &migrated_cells_global, 1,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            int migration_valid = load_balance_migration_valid ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &migration_valid, 1, MPI_INT,
                MPI_MIN, MPI_COMM_WORLD);
            load_balance_migration_valid = migration_valid != 0 &&
                (world_size == 1 || migrated_cells_global > 0);
        }
#endif

        size_t total_refined_local = individual_amr ? actual_refined_local : refine.getTotalRefined();
        size_t total_removed_local = individual_amr ? actual_removed_local : remove.getTotalRemoved();
        size_t total_refined_global = total_refined_local;
        size_t total_removed_global = total_removed_local;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &total_refined_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &total_removed_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        size_t generators_moved_global = generators_moved_local;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &generators_moved_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(rank == 0) {
            std::cout << std::endl
                      << "Total points refined: " << total_refined_global
                      << ", Total points removed: " << total_removed_global
                      << std::endl << std::endl;
        }

        double max_drift = max_drift_local;
#ifdef RICH_MPI
        MPI_Allreduce(&max_drift_local, &max_drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_volume_growth, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        const bool mpi_mode = world_size > 1;
        const double threshold = mpi_mode ? 1e-6 : 1e-8;
        const double conservation_threshold = 1e-10;
        const bool individual_gate_passed = !individual_amr ||
            (scheduler_remap_valid && total_refined_global > 0 &&
             total_removed_global > 0 && generators_moved_global == 0 &&
             max_conservation_error <= conservation_threshold &&
             load_balance_migration_valid);
        const int passed = (max_drift <= threshold &&
            max_volume_growth <= 3.0 * (1.0 + 1e-8) &&
            individual_gate_passed) ? 1 : 0;
        int all_passed = passed;
#ifdef RICH_MPI
        MPI_Allreduce(&passed, &all_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

        if(rank == 0) {
            std::ofstream out("amr_random_metrics.txt");
            out.setf(std::ios::scientific);
            out.precision(16);
            out << "mode " << (individual_amr ?
                (mpi_mode ? "mpi-individual" : "serial-individual") :
                (mpi_mode ? "mpi" : "serial")) << "\n";
            out << "rounds " << amr_rounds << "\n";
            out << "target_points " << target_points << "\n";
            out << "max_drift " << max_drift << "\n";
            out << "threshold " << threshold << "\n";
            out << "max_volume_growth " << max_volume_growth << "\n";
            out << "volume_growth_limit 3.0\n";
            out << "actual_refined " << total_refined_global << "\n";
            out << "actual_removed " << total_removed_global << "\n";
            out << "max_conservation_error " << max_conservation_error << "\n";
            out << "worst_conservation_component " << worst_conservation_component << "\n";
            out << "conservation_threshold " << conservation_threshold << "\n";
            out << "scheduler_remap_valid " << scheduler_remap_valid << "\n";
            out << "individual_generators_moved " << generators_moved_global << "\n";
            out << "load_balance_migration_valid "
                << load_balance_migration_valid << "\n";
            out << "migrated_cells " << migrated_cells_global << "\n";
            out << "pass " << all_passed << "\n";
            out.close();
        }

#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return all_passed ? 0 : 1;
    }
    catch(UniversalError const& e) {
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }
}
